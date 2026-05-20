"""Memory Router: decides fast_think vs deep_think vs match.

Inspired by MemBrain's memory-router agent.
Analyzes the user's message to decide the retrieval strategy.
"""
import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from ..core.trace import append_trace
from ..models import OryaState

logger = logging.getLogger(__name__)

ROUTER_PROMPT = """You are a conversation router for Orya, a human-like chat agent.
Analyze the user's latest message and decide the response strategy.

STRATEGIES:
- "chat": Normal conversation. Recent context is sufficient.
- "deep_think": User mentions something from past conversations ("do you remember", "last time").
- "match": User is looking for someone/something ("tu connais", "cherche", "looking for", "need a").

Output ONLY valid JSON: {"strategy": "chat|deep_think|match", "query": "search query if match/deep_think"}"""


async def memory_router_node(state: OryaState, llm: Runnable) -> dict[str, Any]:
    """Route the conversation to the appropriate strategy."""
    text = state.get("last_user_text", "")

    # Fast heuristic first
    t_lower = text.lower()
    match_triggers = [
        "cherche", "recherche", "looking for", "need a", "besoin de",
        "tu connais", "connais-tu", "do you know", "quelqu'un",
        "une personne", "un pro", "un dev", "un plombier",
    ]
    deep_triggers = [
        "tu te souviens", "do you remember", "la dernière fois",
        "last time", "on avait parlé", "we talked about",
    ]

    strategy = "chat"
    query = ""

    if any(trig in t_lower for trig in match_triggers):
        strategy = "match"
        query = text
    elif any(trig in t_lower for trig in deep_triggers):
        strategy = "deep_think"
        query = text
    else:
        # LLM fallback for ambiguous cases
        try:
            prompt = [
                SystemMessage(content=ROUTER_PROMPT),
                HumanMessage(content=f"User message: {text}"),
            ]
            resp = await llm.ainvoke(prompt)
            data = _extract_json(str(getattr(resp, "content", "")))
            strategy = data.get("strategy", "chat")
            query = data.get("query", text)
        except Exception as e:
            logger.warning("Memory router LLM failed: %s", e)
            strategy = "chat"

    trace = append_trace(state, "memory_router", f"strategy={strategy}")
    return {
        "strategy": strategy,
        "match_query": query if strategy == "match" else "",
        "trace": trace,
    }


def _extract_json(text: str) -> dict[str, Any]:
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
