"""
Orchestrateur Async — Le cerveau en arrière-plan d'Orya.

Ce service reçoit chaque message utilisateur APRÈS que l'Agent Orya ait déjà répondu.
Il travaille en tâche de fond :

Pipeline :
1. EXTRACTION → Extrait passivement des facts (skills, ville, besoins, frustrations...)
2. INTENT DETECTION → Détecte si l'user cherche quelqu'un/quelque chose
3. SEARCH DECISION → Si oui, lance une recherche dans le graphe + vectoriel
4. DOUBLE OPT-IN → Quand un match est trouvé, demande aux DEUX parties
5. TUNNEL → Quand les deux acceptent, déclenche la mise en relation

Le tout est INVISIBLE pour l'utilisateur. Orya dit juste "attends je regarde"
et quand c'est prêt, elle notifie naturellement.
"""

import os
import time
import asyncio
from typing import Optional

from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from dotenv import load_dotenv

from extractor import extract_facts
from intent_detector import detect_intent, Intent
from opt_in_manager import OptInManager

load_dotenv()

app = FastAPI(title="Orchestrator", version="0.1.0")

MEMORY_URL = os.getenv("MEMORY_URL", "http://localhost:5003")
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:4001")

opt_in = OptInManager()


# ── Models ─────────────────────────────────────────────────────────
class ProcessRequest(BaseModel):
    userId: str
    text: str


class OptInResponse(BaseModel):
    userId: str
    matchId: str
    accept: bool


# ── Endpoints ──────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "orchestrator", "ts": time.time()}


@app.post("/process")
async def process_message(req: ProcessRequest, bg: BackgroundTasks):
    """
    Called by Gateway fire-and-forget after each user message.
    All heavy lifting happens in background.
    """
    bg.add_task(pipeline, req.userId, req.text)
    return {"ok": True, "queued": True}


@app.post("/opt-in")
async def handle_opt_in(req: OptInResponse):
    """User responds to a double opt-in request."""
    result = opt_in.respond(req.userId, req.matchId, req.accept)
    if result == "both_accepted":
        # Both sides said yes — trigger tunnel
        await _trigger_tunnel(req.matchId)
    elif result == "declined":
        # One side refused
        await _notify_decline(req.matchId, req.userId)
    return {"status": result}


# ── Background Pipeline ────────────────────────────────────────────
async def pipeline(user_id: str, text: str):
    """
    The main async pipeline. Runs after Orya has already replied.
    User doesn't wait for this.
    """
    import httpx

    # ─── STEP 1: Extract facts passively ───────────────────────────
    facts = await extract_facts(user_id, text)

    if facts:
        # Store in memory service
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(f"{MEMORY_URL}/facts", json={
                    "userId": user_id,
                    "facts": [f.dict() for f in facts],
                })
        except Exception as e:
            print(f"[orchestrator] memory store failed: {e}")

        # Notify user via gateway (trace event)
        for fact in facts:
            await _push_to_user(user_id, {
                "type": "fact_recorded",
                "fact": fact.dict(),
            })

    # ─── STEP 2: Detect intent ─────────────────────────────────────
    intent = await detect_intent(user_id, text)

    if intent and intent.type == "search":
        # ─── STEP 3: Search in graph + vector ──────────────────────
        await _handle_search(user_id, intent)

    elif intent and intent.type == "opt_in_reply":
        # User naturally replied about a pending match
        pass  # Handled via /opt-in endpoint


async def _handle_search(user_id: str, intent: Intent):
    """
    Orya detected that the user is looking for someone.
    Search memory + vector, then propose candidates.
    """
    import httpx

    # Notify user: "je regarde..."
    await _push_to_user(user_id, {
        "type": "reply",
        "text": "attends 2 sec je check si j'ai quelqu'un...",
        "meta": {"source": "orchestrator", "action": "search_started"},
    })

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(f"{MEMORY_URL}/search", json={
                "userId": user_id,
                "query": intent.query,
                "skills": intent.skills,
                "city": intent.city,
            })
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])

                if candidates:
                    await _push_to_user(user_id, {
                        "type": "candidates",
                        "sessionId": f"search-{user_id}-{int(time.time())}",
                        "candidates": candidates,
                        "tour": 1,
                        "reason": intent.query,
                    })
                else:
                    await _push_to_user(user_id, {
                        "type": "reply",
                        "text": "hmm j'ai personne en tête pour l'instant... je garde en mémoire, si quelqu'un se manifeste je te dis",
                        "meta": {"source": "orchestrator", "action": "no_results"},
                    })
    except Exception as e:
        print(f"[orchestrator] search failed: {e}")
        await _push_to_user(user_id, {
            "type": "reply",
            "text": "ah j'arrive pas à chercher là... réessaie dans 2 min",
            "meta": {"source": "orchestrator", "action": "search_error"},
        })


async def _trigger_tunnel(match_id: str):
    """Both users accepted — initiate the connection."""
    match = opt_in.get_match(match_id)
    if not match:
        return

    # Notify both users
    msg_seeker = "c'est bon ! je vous mets en contact. tu vas recevoir ses coordonnées"
    msg_provider = f"hey, quelqu'un cherche tes services — je lui file ton contact ok ?"

    await _push_to_user(match["seeker_id"], {
        "type": "reply",
        "text": msg_seeker,
        "meta": {"source": "orchestrator", "action": "tunnel_open", "matchId": match_id},
    })
    await _push_to_user(match["provider_id"], {
        "type": "reply",
        "text": f"c'est parti, {match['seeker_alias']} va te contacter 👋",
        "meta": {"source": "orchestrator", "action": "tunnel_open", "matchId": match_id},
    })


async def _notify_decline(match_id: str, declined_by: str):
    """One side declined the match."""
    match = opt_in.get_match(match_id)
    if not match:
        return

    seeker_id = match["seeker_id"]
    if declined_by != seeker_id:
        # Provider declined — tell seeker gently
        await _push_to_user(seeker_id, {
            "type": "reply",
            "text": "ah dommage, la personne est pas dispo en ce moment... je continue de chercher",
            "meta": {"source": "orchestrator", "action": "declined"},
        })


async def _push_to_user(user_id: str, payload: dict):
    """Push an event to a connected user via Gateway's internal API."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            await client.post(f"{GATEWAY_URL}/internal/push/{user_id}", json=payload)
    except:
        pass  # User might be offline
