"""Cross-user matching node.

Triggers matching when the user is looking for someone.
Uses sequential reveal (1 candidate at a time).
"""
import logging
from typing import Any

from ..core.trace import append_trace
from ..db import create_opt_in
from ..matching import get_sequential_candidate
from ..models import OryaState
from ..providers.embedder import HuggingFaceEmbedder

logger = logging.getLogger(__name__)


async def cross_user_match_node(state: OryaState, embedder: HuggingFaceEmbedder | None = None) -> dict[str, Any]:
    user_id = state.get("user_id", "")
    query = state.get("match_query", "")

    if not query:
        trace = append_trace(state, "cross_user_match", "no query")
        return {"candidates": [], "pending_opt_in": None, "trace": trace}

    candidate = await get_sequential_candidate(user_id, query, embedder=embedder)

    if not candidate:
        trace = append_trace(state, "cross_user_match", "no candidates")
        return {
            "candidates": [],
            "pending_opt_in": None,
            "trace": trace,
        }

    try:
        row = await create_opt_in(
            seeker_id=user_id,
            provider_id=candidate["user_id"],
            reason=candidate["summary"],
            candidate_uuid=candidate["candidate_uuid"],
        )
        if not row:
            logger.info("Opt-in already exists for seeker=%s provider=%s", user_id, candidate["user_id"])
            trace = append_trace(state, "cross_user_match", "already proposed")
            return {"candidates": [], "pending_opt_in": None, "trace": trace}
    except Exception as e:
        logger.error("Failed to create opt-in: %s", e)
        trace = append_trace(state, "cross_user_match", f"opt_in_error: {e}")
        return {"candidates": [], "pending_opt_in": None, "trace": trace}

    trace = append_trace(
        state,
        "cross_user_match",
        f"proposed provider={candidate['user_id'][:8]} score={candidate['score']:.1f}",
    )

    return {
        "candidates": [candidate],
        "pending_opt_in": {
            "opt_in_id": str(row["id"]),
            "summary": candidate["summary"],
            "provider_alias": candidate["alias"],
        },
        "trace": trace,
    }
