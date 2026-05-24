"""Node: search_match.

Cross-group Graphiti search: looks across ALL users for facts that match the
seeker's need, then aggregates per-user candidates ranked by RRF.

The result is a list of candidate dicts ready for `opt_in_propose`.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from graphiti_core import Graphiti

from ..models import OryaState
from ..settings import get_settings

logger = logging.getLogger(__name__)


def make_search_match_node(graphiti: Graphiti):
    s = get_settings()

    async def search_match_node(state: OryaState) -> dict[str, Any]:
        seeker_id = state["user_id"]
        text = state.get("last_user_text") or ""
        intent = state.get("intent") or {}
        domain = intent.get("domain")
        location = intent.get("location")

        # Compose a richer query if we have signals from intent.
        bits: list[str] = [text]
        if domain:
            bits.append(domain)
        if location:
            bits.append(location)
        query = " ".join(bits).strip() or text

        try:
            edges = await graphiti.search(
                query=query,
                num_results=s.SEARCH_NUM_RESULTS_MATCH,
            )
        except Exception as e:
            logger.warning("Graphiti cross-group search failed: %s", e)
            return {
                "candidates": [],
                "trace": _append_trace(
                    state, "search_match", f"search failed: {e}"
                ),
            }

        # Group by group_id (= user_id) and skip the seeker.
        # Score = sum of inverse-rank weights (Reciprocal Rank Fusion style).
        per_user: dict[str, dict[str, Any]] = {}
        for rank, edge in enumerate(edges or [], start=1):
            group_id = getattr(edge, "group_id", None) or getattr(
                edge, "groupId", None
            )
            if not group_id or group_id == seeker_id:
                continue
            agg = per_user.setdefault(
                group_id,
                {
                    "user_id": group_id,
                    "score": 0.0,
                    "facts": [],
                    "candidate_uuid": getattr(edge, "uuid", "") or "",
                },
            )
            agg["score"] += 1.0 / rank
            fact = getattr(edge, "fact", None)
            if fact:
                agg["facts"].append(fact)
            if not agg.get("candidate_uuid"):
                agg["candidate_uuid"] = getattr(edge, "uuid", "") or ""

        candidates = sorted(
            per_user.values(), key=lambda c: c["score"], reverse=True
        )[:5]

        # Format a short summary for each candidate (top 3 facts)
        for c in candidates:
            facts = c.pop("facts", [])
            c["summary"] = " · ".join(facts[:3]) if facts else "(pas de résumé)"
            c["alias"] = None

        return {
            "candidates": candidates,
            "trace": _append_trace(
                state, "search_match", f"{len(candidates)} candidates"
            ),
        }

    return search_match_node


def _append_trace(state: OryaState, step: str, detail: str) -> list[dict[str, Any]]:
    existing = list(state.get("trace") or [])
    existing.append({"step": step, "detail": detail})
    return existing
