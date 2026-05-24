"""Session summarizer for episodic memory.

Generates narrative summaries of conversation sessions.
Inspired by MemBrain's session summarizer.
"""
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from .models import SessionSummary

logger = logging.getLogger(__name__)

SUMMARIZER_PROMPT = """You are a session summarizer.
Summarize the conversation into a brief narrative (3-5 sentences).
Capture: who was discussed, what happened, key facts, user's mood/preferences.

Output ONLY valid JSON: {"subject": "Brief title", "content": "Summary text..."}"""


async def summarize_session(
    messages: list[dict[str, str]],
    session_number: int,
    llm: Runnable,
) -> SessionSummary:
    """Generate an episodic summary from conversation messages."""
    conversation = "\n".join(
        f"{m.get('speaker', 'User')}: {m.get('content', '')}" for m in messages
    )
    prompt = [
        SystemMessage(content=SUMMARIZER_PROMPT),
        HumanMessage(content=f"Conversation:\n{conversation}"),
    ]
    try:
        resp = await llm.ainvoke(prompt)
        data = _extract_json(str(getattr(resp, "content", "")))
        return SessionSummary(
            session_number=session_number,
            subject=data.get("subject", "Session summary"),
            content=data.get("content", ""),
        )
    except Exception as e:
        logger.error("Session summarization failed: %s", e)
        return SessionSummary(
            session_number=session_number,
            subject="Session summary",
            content=conversation[:500],
        )


def _extract_json(text: str) -> dict[str, Any]:
    import json

    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```json"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except Exception:
        return {}
