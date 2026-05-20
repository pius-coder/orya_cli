"""LangGraph StateGraph builder for Orya v3 with MemBrain.

All async node wrappers are defined as named inner functions
so LangGraph can schedule them correctly (lambdas would return
coroutine objects instead of awaiting them).
"""
import logging
from typing import Any

from graphiti_core import Graphiti
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph

from ..manifests.registry import ManifestRegistry
from ..models import OryaState
from ..providers.embedder import HuggingFaceEmbedder
from ..tools.executor import ToolExecutor
from .cross_user_match import cross_user_match_node
from .memory_router import memory_router_node
from .persist_and_match import make_persist_and_match_node
from .retrieve_context import make_retrieve_context_node
from .tool_agent import make_tool_agent_node

logger = logging.getLogger(__name__)


def _conditional_match(state: OryaState) -> str:
    """Route to cross_user_match if strategy == 'match', else tool_agent."""
    if state.get("strategy") == "match":
        return "cross_user_match"
    return "tool_agent"


def build_graph(
    *,
    graphiti: Graphiti,
    llm: Runnable,
    manifests: ManifestRegistry,
    embedder: HuggingFaceEmbedder | None = None,
) -> StateGraph:
    """Compile and return the Orya v3+MemBrain LangGraph."""
    executor = ToolExecutor(graphiti)

    # ── Async wrappers (not lambdas) so LangGraph awaits them ──────

    async def _memory_router(state: OryaState) -> dict[str, Any]:
        return await memory_router_node(state, llm)

    async def _cross_user_match(state: OryaState) -> dict[str, Any]:
        return await cross_user_match_node(state, embedder)

    graph = StateGraph(OryaState)

    graph.add_node("memory_router", _memory_router)
    graph.add_node("retrieve_context", make_retrieve_context_node(graphiti))
    graph.add_node("cross_user_match", _cross_user_match)
    graph.add_node("tool_agent", make_tool_agent_node(llm, manifests, executor))
    graph.add_node("persist_and_match", make_persist_and_match_node(graphiti, llm, embedder))

    graph.add_edge(START, "memory_router")
    graph.add_edge("memory_router", "retrieve_context")
    graph.add_conditional_edges(
        "retrieve_context",
        _conditional_match,
        {"cross_user_match": "cross_user_match", "tool_agent": "tool_agent"},
    )
    graph.add_edge("cross_user_match", "tool_agent")
    graph.add_edge("tool_agent", "persist_and_match")
    graph.add_edge("persist_and_match", END)

    compiled = graph.compile()
    logger.info("Orya v3+MemBrain LangGraph compiled (5 nodes)")
    return compiled
