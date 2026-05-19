"""LangGraph StateGraph definition for Orya.

Wires the nodes together. Compilation happens in `main.py` (so we can pass
the live checkpointer / dependencies).
"""

from __future__ import annotations

from typing import Literal

from graphiti_core import Graphiti
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph

from .models import OryaState
from .nodes import (
    extract_quick_node,
    make_detect_intent_node,
    make_persist_episode_node,
    make_persona_respond_node,
    make_retrieve_context_node,
    make_search_match_node,
    notify_user_node,
    opt_in_propose_node,
)


def _route_after_intent(state: OryaState) -> Literal["search_match", "notify_user"]:
    intent = state.get("intent") or {}
    if intent.get("action") == "search":
        return "search_match"
    return "notify_user"


def build_graph_builder(
    *,
    graphiti: Graphiti,
    llm_router: Runnable,
    small_llm: Runnable,
) -> StateGraph:
    """Construct the StateGraph builder. Caller is responsible for `.compile`."""

    builder = StateGraph(OryaState)

    builder.add_node(
        "retrieve_context", make_retrieve_context_node(graphiti)
    )
    builder.add_node("persona_respond", make_persona_respond_node(llm_router))
    builder.add_node(
        "persist_episode", make_persist_episode_node(graphiti)
    )
    builder.add_node("extract_quick", extract_quick_node)
    builder.add_node("detect_intent", make_detect_intent_node(small_llm))
    builder.add_node("search_match", make_search_match_node(graphiti))
    builder.add_node("opt_in_propose", opt_in_propose_node)
    builder.add_node("notify_user", notify_user_node)

    builder.add_edge(START, "retrieve_context")
    builder.add_edge("retrieve_context", "persona_respond")
    builder.add_edge("persona_respond", "persist_episode")
    builder.add_edge("persist_episode", "extract_quick")
    builder.add_edge("extract_quick", "detect_intent")

    builder.add_conditional_edges(
        "detect_intent",
        _route_after_intent,
        {"search_match": "search_match", "notify_user": "notify_user"},
    )
    builder.add_edge("search_match", "opt_in_propose")
    builder.add_edge("opt_in_propose", "notify_user")
    builder.add_edge("notify_user", END)

    return builder
