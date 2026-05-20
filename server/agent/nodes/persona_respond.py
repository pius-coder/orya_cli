"""Node: persona_respond.

Compose the prompt (system + facts + few-shot + history), invoke the LLM
router, and append the AI message to the conversation state.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable

from ..db import get_good_examples, get_user
from ..models import OryaState
from ..persona import build_messages

logger = logging.getLogger(__name__)


def make_persona_respond_node(llm: Runnable, small_llm: Runnable | None = None):
    async def persona_respond_node(state: OryaState) -> dict[str, Any]:
        user_id = state["user_id"]
        # Make sure the latest human turn is in messages
        history = list(state.get("messages") or [])
        last_text = state.get("last_user_text") or ""
        if (
            last_text
            and (not history or not isinstance(history[-1], HumanMessage)
                 or history[-1].content != last_text)
        ):
            history = history + [HumanMessage(content=last_text)]

        # Persona settings — read tutoyer flag from PG (default True)
        try:
            user_row = await get_user(user_id)
        except Exception:
            user_row = None
        tutoyer = bool(user_row.get("tutoyer", True)) if user_row else True
        alias = (user_row or {}).get("alias") or state.get("user_alias")

        # Positive few-shot examples from feedback (rating=1)
        try:
            good = await get_good_examples(exclude_user_id=user_id, limit=3)
        except Exception:
            good = []

        # Build dynamic user context prompt block under XML/Markdown structure
        from ..persona.user_prompt import build_user_prompt
        user_prompt_content = await build_user_prompt(
            user_id=user_id,
            user_alias=alias,
            last_user_text=last_text,
            facts_context=state.get("facts_context") or [],
            history=history,
            tutoyer=tutoyer,
        )

        prompt_messages = build_messages(
            history=history,
            facts_context=state.get("facts_context") or [],
            good_examples=good,
            tutoyer=tutoyer,
            user_alias=alias,
            user_prompt_content=user_prompt_content,
        )

        try:
            ai = await llm.ainvoke(prompt_messages)
        except Exception as e:
            logger.exception("LLM invoke failed")
            return {
                "messages": [AIMessage(content="Pardon, j'ai eu un souci. Tu peux répéter ?")],
                "last_assistant_reply": "Pardon, j'ai eu un souci. Tu peux répéter ?",
                "trace": _append_trace(state, "persona_respond", f"error: {e}"),
            }

        reply_text = _extract_text(ai)
        reply_text = _enforce_brevity(reply_text)
        return {
            "messages": [AIMessage(content=reply_text)],
            "last_assistant_reply": reply_text,
            "trace": _append_trace(state, "persona_respond", "ok"),
        }

    return persona_respond_node


def _extract_text(ai: Any) -> str:
    """Extract a plain string from an AIMessage / message-like object."""
    content = getattr(ai, "content", ai)
    if isinstance(content, list):
        # multi-part content (e.g. tool calls + text) — keep text only
        parts = [
            c.get("text", "") if isinstance(c, dict) else str(c)
            for c in content
        ]
        content = "".join(parts)
    return str(content).strip()


def _enforce_brevity(text: str, max_words: int = 70) -> str:
    """Hard cap on length to keep the persona consistent."""
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words]).rstrip(",;:.") + "…"
    return text


def _append_trace(state: OryaState, step: str, detail: str) -> list[dict[str, Any]]:
    existing = list(state.get("trace") or [])
    existing.append({"step": step, "detail": detail})
    return existing
