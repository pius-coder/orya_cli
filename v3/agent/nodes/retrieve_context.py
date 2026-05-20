"""Retrieve context from PKG (MemBrain) + Graphiti + opt-ins.

Updated to use the Personal Knowledge Graph as primary source,
with Graphiti as temporal fallback.
"""
from typing import Any

from graphiti_core import Graphiti

from ..core.config import get_settings
from ..core.trace import append_trace
from ..db import get_reflections, list_pending_opt_ins
from ..memory.retrieval import retrieve_from_pkg
from ..models import OryaState


def make_retrieve_context_node(graphiti: Graphiti):
    async def retrieve_context_node(state: OryaState) -> dict[str, Any]:
        user_id = state.get("user_id", "")
        text = state.get("last_user_text", "")
        settings = get_settings()

        facts: list[str] = []

        # 1. Retrieve from Personal Knowledge Graph (MemBrain layer)
        try:
            pkg_results = await retrieve_from_pkg(user_id, text, top_k=settings.SEARCH_NUM_RESULTS_CONTEXT)
            for f in pkg_results.get("facts", []):
                facts.append(str(f.get("text", "")))
            for ent in pkg_results.get("entities", []):
                facts.append(f"Entité connue: {ent.get('canonical_ref', '')}")
            for s in pkg_results.get("summaries", []):
                facts.append(f"Session: {s.get('subject', '')} — {s.get('content', '')[:100]}")
        except Exception as e:
            facts.append(f"[Erreur PKG: {e}]")

        # 2. Fallback: Graphiti search
        if len(facts) < 3:
            try:
                results = await graphiti.search(
                    query=text,
                    group_ids=[user_id],
                    num_results=settings.SEARCH_NUM_RESULTS_CONTEXT,
                )
                for r in results:
                    fact = getattr(r, "fact", None)
                    if fact:
                        facts.append(str(fact))
            except Exception as e:
                facts.append(f"[Erreur Graphiti: {e}]")

        # 3. Reflections from PostgreSQL
        try:
            reflections = await get_reflections(user_id)
            if reflections.get("user_reflection"):
                facts.append(f"Portrait: {reflections['user_reflection']}")
            if reflections.get("orya_reflection"):
                facts.append(f"Relation: {reflections['orya_reflection']}")
        except Exception as e:
            facts.append(f"[Erreur reflections: {e}]")

        # 4. Pending opt-ins
        try:
            pending = await list_pending_opt_ins(user_id)
            if pending:
                facts.append("--- MATCHINGS EN ATTENTE ---")
                for p in pending:
                    status = p.get("status", "?")
                    reason = (p.get("reason") or "")[:120]
                    facts.append(f"[{status}] {reason}")
        except Exception as e:
            facts.append(f"[Erreur opt-ins: {e}]")

        facts_context = "\n".join(facts)
        trace = append_trace(state, "retrieve_context", f"{len(facts)} facts loaded (PKG+Graphiti)")

        return {"facts_context": facts_context, "trace": trace}

    return retrieve_context_node
