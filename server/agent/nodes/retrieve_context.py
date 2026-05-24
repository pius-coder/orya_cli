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
        
        facts: list[str] = []

        # 1. Search semantic facts from Graphiti long-term memory
        if text:
            try:
                edges = await graphiti.search(
                    query=text,
                    group_ids=[user_id],
                    num_results=s.SEARCH_NUM_RESULTS_CONTEXT,
                )
                for edge in edges or []:
                    fact = getattr(edge, "fact", None)
                    if fact:
                        facts.append(fact)
            except Exception as e:  # pragma: no cover
                logger.warning("Graphiti.search failed: %s", e)

        # 2. Retrieve pending opt-in matches from PostgreSQL
        try:
            from ..db import list_pending_opt_ins, get_user
            opt_ins = await list_pending_opt_ins(user_id)
            if opt_ins:
                opt_ins_str = []
                for opt in opt_ins:
                    pid = opt["provider_id"]
                    try:
                        p_row = await get_user(pid)
                        p_alias = p_row.get("alias") if p_row else pid
                    except Exception:
                        p_alias = pid
                    opt_ins_str.append(
                        f"Prestataire '{p_alias}' disponible pour le besoin '{opt['need_summary']}' (Opt-in ID: {opt['opt_in_id']})"
                    )
                if opt_ins_str:
                    facts.append("--- MATCHINGS EN ATTENTE ---")
                    facts.append("Voici des profils du réseau qui correspondent aux besoins récents. Parles-en à l'utilisateur de façon naturelle s'il pose une question sur sa recherche ou si c'est le bon moment :")
                    facts.extend(opt_ins_str)
                    facts.append("----------------------------")
        except Exception:
            logger.exception("Failed to retrieve pending opt-ins for user=%s", user_id)

        return {
            "facts_context": facts,
            "trace": _append_trace(
                state, "retrieve_context", f"{len(facts)} facts/matches loaded"
            ),
        }

    return retrieve_context_node


def _append_trace(state: OryaState, step: str, detail: str) -> list[dict[str, Any]]:
    existing = list(state.get("trace") or [])
    existing.append({"step": step, "detail": detail})
    return existing



