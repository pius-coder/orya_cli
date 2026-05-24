"""Node: notify_user.

Push asynchronous events to the gateway so they can be forwarded over the
WebSocket to the connected user. The gateway exposes
`POST /internal/push/:userId` and accepts the same JSON shape as a normal
WS message (see types.ts in the CLI).
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from ..models import OryaState
from ..settings import get_settings

logger = logging.getLogger(__name__)


async def notify_user_node(state: OryaState) -> dict[str, Any]:
    s = get_settings()
    user_id = state["user_id"]

    payloads: list[dict[str, Any]] = []

    # Live UI hint: the rule-based extracted facts
    for fact in state.get("extracted_facts") or []:
        payloads.append(
            {
                "type": "fact_recorded",
                "label": fact["label"],
                "value": fact["value"],
                "confidence": fact["confidence"],
            }
        )

    # Match candidates (one card per provider)
    candidates = state.get("candidates") or []
    if candidates:
        payloads.append(
            {
                "type": "candidates",
                "items": [
                    {
                        "user_id": c["user_id"],
                        "alias": c.get("alias"),
                        "summary": c["summary"],
                        "score": c["score"],
                        "candidate_uuid": c.get("candidate_uuid") or "",
                    }
                    for c in candidates
                ],
            }
        )

    # Trace events (only if anything happened — keeps client logs tidy)
    trace = state.get("trace") or []
    for ev in trace:
        payloads.append(
            {
                "type": "trace",
                "step": ev.get("step", ""),
                "detail": ev.get("detail"),
            }
        )

    if not payloads:
        return {"trace": _append_trace(state, "notify_user", "nothing to push")}

    base = s.GATEWAY_INTERNAL_URL.rstrip("/")
    url = f"{base}/internal/push/{user_id}"

    pushed = 0
    async with httpx.AsyncClient(timeout=5) as client:
        for p in payloads:
            try:
                await client.post(url, json=p)
                pushed += 1
            except Exception as e:
                logger.warning("push failed: %s", e)

    return {"trace": _append_trace(state, "notify_user", f"{pushed} pushed")}


def _append_trace(state: OryaState, step: str, detail: str) -> list[dict[str, Any]]:
    existing = list(state.get("trace") or [])
    existing.append({"step": step, "detail": detail})
    return existing
