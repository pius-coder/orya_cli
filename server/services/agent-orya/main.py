"""
Agent Orya — Service de conversation per-user.

Reçoit un message utilisateur, construit le prompt avec persona + historique + facts,
appelle le LLM, et retourne la réponse.

Le but : que chaque réponse ressemble à un texto d'une vraie personne.
"""

import os
import time
from collections import defaultdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from persona import build_messages, NEGATIVE_EXAMPLES
from llm_router import call_llm
from feedback_store import record_feedback, get_good_examples

load_dotenv()

app = FastAPI(title="Agent Orya", version="0.1.0")

# In-memory conversation history per user (MVP — swap for Redis later)
histories: dict[str, list[dict[str, str]]] = defaultdict(list)

# Last response pending feedback per user
pending_responses: dict[str, dict] = {}

MEMORY_URL = os.getenv("MEMORY_URL", "http://localhost:5003")


# ── Models ─────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    userId: str
    text: str


class ChatResponse(BaseModel):
    reply: str
    provider: str | None = None
    facts: list[dict] = []


class FeedbackRequest(BaseModel):
    userId: str
    rating: str  # "good" | "bad"


class RetryResponse(BaseModel):
    reply: str
    provider: str | None = None


# ── Endpoints ──────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "agent-orya", "ts": time.time()}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Main conversation endpoint — called by Gateway for each user message."""
    user_id = req.userId
    text = req.text
    history = histories[user_id]

    # Fetch known facts from memory service (best effort)
    user_facts = await _get_user_facts(user_id)

    # Get good few-shot examples from feedback store
    few_shot = get_good_examples(n=3, exclude_user=user_id)

    # Build messages
    messages = build_messages(
        user_text=text,
        history=history,
        user_facts=user_facts,
        few_shot_good=few_shot,
    )

    # Call LLM
    result = await call_llm(messages, temperature=0.85, max_tokens=150)
    reply = result["text"]

    # Post-process: enforce short response (hard trim if needed)
    reply = _enforce_brevity(reply)

    # Update history
    history.append({"role": "user", "content": text})
    history.append({"role": "assistant", "content": reply})

    # Keep history bounded
    if len(history) > 30:
        histories[user_id] = history[-20:]

    # Store as pending for feedback
    pending_responses[user_id] = {
        "input": text,
        "response": reply,
        "ts": time.time(),
    }

    return ChatResponse(
        reply=reply,
        provider=result.get("provider"),
        facts=[],  # Facts are extracted by orchestrator asynchronously
    )


@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    """Rate the last Orya response. If 'bad', triggers a retry."""
    user_id = req.userId
    pending = pending_responses.get(user_id)
    if not pending:
        raise HTTPException(404, "No pending response to rate")

    record_feedback(
        user_input=pending["input"],
        orya_response=pending["response"],
        rating=req.rating,
        user_id=user_id,
    )

    if req.rating == "bad":
        # Retry with stronger constraints
        return await _retry(user_id, pending)

    # Clear pending
    del pending_responses[user_id]
    return {"ok": True, "message": "noted 👍"}


async def _retry(user_id: str, pending: dict) -> dict:
    """Regenerate response with the bad one as negative example."""
    history = histories[user_id]
    user_facts = await _get_user_facts(user_id)
    few_shot = get_good_examples(n=3, exclude_user=user_id)

    # Remove the bad response from history
    if history and history[-1]["content"] == pending["response"]:
        history.pop()  # Remove bad assistant response

    # Add the bad response as a negative in the prompt
    messages = build_messages(
        user_text=pending["input"],
        history=history[:-1] if history else [],  # Remove the user msg too (we re-add it)
        user_facts=user_facts,
        few_shot_good=few_shot,
    )

    # Inject negative: "ne dis PAS ça"
    messages.insert(-1, {
        "role": "user",
        "content": f"[NE DIS PAS ÇA] Ta dernière réponse était trop robotique : \"{pending['response']}\" — refais plus naturel, plus court"
    })
    messages.insert(-1, {"role": "assistant", "content": "ok je refais"})

    result = await call_llm(messages, temperature=0.9, max_tokens=100)
    new_reply = _enforce_brevity(result["text"])

    # Update history with new response
    history.append({"role": "assistant", "content": new_reply})

    # Store new pending
    pending_responses[user_id] = {
        "input": pending["input"],
        "response": new_reply,
        "ts": time.time(),
    }

    return {"ok": True, "retry_reply": new_reply, "provider": result.get("provider")}


def _enforce_brevity(text: str) -> str:
    """Hard limit: max 2 sentences or 50 words."""
    # Remove any bullet points or lists that slipped through
    lines = text.split("\n")
    lines = [l for l in lines if not l.strip().startswith(("-", "•", "*", "1.", "2.", "3."))]
    text = " ".join(l.strip() for l in lines if l.strip())

    # Word limit
    words = text.split()
    if len(words) > 50:
        text = " ".join(words[:50])
        if not text.endswith((".", "!", "?", "...")):
            text += "..."

    return text


async def _get_user_facts(user_id: str) -> list[str]:
    """Fetch known facts from Memory service (best effort, non-blocking)."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{MEMORY_URL}/facts/{user_id}")
            if resp.status_code == 200:
                data = resp.json()
                return data.get("facts", [])
    except:
        pass
    return []
