"""LangGraph StateGraph definition for Orya v3 — simplified.

Architecture:
  START → retrieve_context (reflections + Graphiti) → tool_agent (LLM with tools) → persist → END
                                                          ↓
                                                   cold_track (fire-and-forget)

Pas de Qualifier lourd. Pas de classification d'intent. L'agent LLM décide.
"""

from __future__ import annotations

import asyncio

from graphiti_core import Graphiti
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph

from agent.manifests.registry import ManifestRegistry
from agent.models import OryaState
from agent.nodes.v3.cold_track import run_cold_track
from agent.nodes.v3.tool_agent import make_tool_agent_node
from agent.tools.executor import ToolExecutor


def build_graph_builder_v3(
    *,
    graphiti: Graphiti,
    llm: Runnable,
    manifests: ManifestRegistry,
) -> StateGraph:
    """Construct the simplified Orya v3 StateGraph."""

    builder = StateGraph(OryaState)
    executor = ToolExecutor(graphiti)

    async def retrieve_context_node(state: OryaState) -> dict:
        """Load reflections + Graphiti facts + pending opt-ins."""
        from agent.db import get_reflections, list_pending_opt_ins

        user_id = state["user_id"]
        text = state.get("last_user_text", "")

        # 1. Reflections from PG
        user_ref, orya_ref = await get_reflections(user_id)

        # 2. Graphiti search (scoped to user) — si texte présent
        facts: list[str] = []
        if text:
            try:
                edges = await graphiti.search(
                    query=text,
                    group_ids=[user_id],
                    num_results=5,
                )
                for edge in edges or []:
                    fact = getattr(edge, "fact", None)
                    if fact:
                        facts.append(fact)
            except Exception as e:
                logger = __import__("logging").getLogger(__name__)
                logger.warning("Graphiti search failed: %s", e)

        # 3. Pending opt-ins
        opt_ins = await list_pending_opt_ins(user_id)
        if opt_ins:
            facts.append("--- MATCHINGS EN ATTENTE ---")
            for opt in opt_ins:
                facts.append(
                    f"Prestataire '{opt['provider_id']}' pour '{opt['need_summary']}'"
                )
            facts.append("----------------------------")

        return {
            "facts_context": facts,
            "user_reflection": user_ref,
            "orya_reflection": orya_ref,
            "trace": _append_trace(state, "retrieve_context", f"{len(facts)} items loaded"),
        }

    tool_agent = make_tool_agent_node(llm, manifests, executor)

    async def persist_node(state: OryaState) -> dict:
        """Fire-and-forget background tasks then end."""
        asyncio.create_task(run_cold_track(state, graphiti, manifests, llm))
        return {"trace": _append_trace(state, "persist", "background tasks launched")}

    builder.add_node("retrieve_context", retrieve_context_node)
    builder.add_node("tool_agent", tool_agent)
    builder.add_node("persist", persist_node)

    builder.add_edge(START, "retrieve_context")
    builder.add_edge("retrieve_context", "tool_agent")
    builder.add_edge("tool_agent", "persist")
    builder.add_edge("persist", END)

    return builder


def _append_trace(state: OryaState, step: str, detail: str) -> list[dict]:
    existing = list(state.get("trace") or [])
    existing.append({"step": step, "detail": detail})
    return existing
