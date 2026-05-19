"""Node: retrieve_context.

Search Graphiti for facts relevant to the latest user message, scoped to the
user's own group_id. The resulting fact strings are exposed to the persona
prompt as recent context.
"""

from __future__ import annotations

import logging
from typing import Any

from graphiti_core import Graphiti

from ..models import OryaState
from ..settings import get_settings

logger = logging.getLogger(__name__)


def make_retrieve_context_node(graphiti: Graphiti):
    s = get_settings()

    async def retrieve_context_node(state: OryaState) -> dict[str, Any]:
        user_id = state["user_id"]
        text = state.get("last_user_text") or ""
        if not text:
            return {"facts_context": [], "trace": _append_trace(state, "retrieve_context", "no text")}

        try:
            edges = await graphiti.search(
                query=text,
                group_ids=[user_id],
                num_results=s.SEARCH_NUM_RESULTS_CONTEXT,
            )
        except Exception as e:  # pragma: no cover — boot-time degradation
            logger.warning("Graphiti.search failed: %s", e)
            return {
                "facts_context": [],
                "trace": _append_trace(
                    state, "retrieve_context", f"search failed: {e}"
                ),
            }

        facts: list[str] = []
        for edge in edges or []:
            fact = getattr(edge, "fact", None)
            if fact:
                facts.append(fact)
        return {
            "facts_context": facts,
            "trace": _append_trace(
                state, "retrieve_context", f"{len(facts)} facts"
            ),
        }

    return retrieve_context_node


def _append_trace(state: OryaState, step: str, detail: str) -> list[dict[str, Any]]:
    existing = list(state.get("trace") or [])
    existing.append({"step": step, "detail": detail})
    return existing



