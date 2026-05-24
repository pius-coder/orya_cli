"""Node: opt_in_propose.

Take the top candidate from `state.candidates`, persist a row in
`orya.opt_ins` (status=pending_seeker) and expose it to the next node so the
seeker can be notified.
"""

from __future__ import annotations

import logging
from typing import Any

from ..db import create_opt_in
from ..models import OryaState

logger = logging.getLogger(__name__)


async def opt_in_propose_node(state: OryaState) -> dict[str, Any]:
    candidates = state.get("candidates") or []
    if not candidates:
        return {
            "pending_opt_in": None,
            "trace": _append_trace(state, "opt_in_propose", "no candidates"),
        }

    top = candidates[0]
    seeker_id = state["user_id"]
    provider_id = top.get("user_id")
    candidate_uuid = top.get("candidate_uuid") or ""

    if not provider_id:
        return {
            "pending_opt_in": None,
            "trace": _append_trace(
                state, "opt_in_propose", "missing provider_id"
            ),
        }

    need_summary = (state.get("last_user_text") or "").strip()[:500]

    try:
        row = await create_opt_in(
            seeker_id=seeker_id,
            provider_id=provider_id,
            need_summary=need_summary or "(no summary)",
            candidate_uuid=candidate_uuid or "no-uuid",
        )
    except Exception as e:
        logger.exception("create_opt_in failed")
        return {
            "pending_opt_in": None,
            "trace": _append_trace(
                state, "opt_in_propose", f"db error: {e}"
            ),
        }

    if row is None:
        # Already proposed earlier — silently skip.
        return {
            "pending_opt_in": None,
            "trace": _append_trace(
                state, "opt_in_propose", "already proposed"
            ),
        }

    pending = {
        "opt_in_id": str(row["opt_in_id"]),
        "provider_id": provider_id,
        "summary": top.get("summary") or "",
        "score": top.get("score") or 0.0,
    }
    return {
        "pending_opt_in": pending,
        "trace": _append_trace(state, "opt_in_propose", "row created"),
    }


def _append_trace(state: OryaState, step: str, detail: str) -> list[dict[str, Any]]:
    existing = list(state.get("trace") or [])
    existing.append({"step": step, "detail": detail})
    return existing
