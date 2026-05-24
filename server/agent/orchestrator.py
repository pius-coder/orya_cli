"""Orya v2 — Central background matching orchestrator.

Runs asynchronously after each episode ingestion (cold loop) to search
for matches, record opt-ins, and push candidates events.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
import httpx
from graphiti_core import Graphiti

from .db import create_opt_in, get_user
from .settings import get_settings

logger = logging.getLogger(__name__)

async def run_background_orchestrator(
    user_id: str,
    last_user_text: str,
    graphiti: Graphiti,
) -> None:
    """Perform cross-group matching and create opt-ins in the background."""
    s = get_settings()

    # 1. Skip if text is empty or too short to represent a search/need.
    if not last_user_text or len(last_user_text.split()) < 3:
        return

    logger.info("Cold Loop: starting background orchestrator match for user=%s", user_id)

    # 2. Query Graphiti cross-group search.
    try:
        edges = await graphiti.search(
            query=last_user_text,
            num_results=s.SEARCH_NUM_RESULTS_MATCH,
        )
    except Exception:
        logger.exception("Cold Loop: Graphiti cross-group search failed")
        return

    if not edges:
        logger.info("Cold Loop: no edges found for query")
        return

    # 3. Group by group_id (which is user_id) and filter out the seeker.
    per_user: dict[str, dict[str, any]] = {}
    for rank, edge in enumerate(edges, start=1):
        group_id = getattr(edge, "group_id", None) or getattr(edge, "groupId", None)
        if not group_id or group_id == user_id:
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

    candidates_list = sorted(
        per_user.values(), key=lambda c: c["score"], reverse=True
    )[:5]

    if not candidates_list:
        logger.info("Cold Loop: no matching candidates found")
        return

    # 4. Propose the top candidate.
    top = candidates_list[0]
    provider_id = top["user_id"]
    candidate_uuid = top["candidate_uuid"]
    score = top["score"]

    # Gather provider details.
    try:
        provider_row = await get_user(provider_id)
        provider_alias = provider_row.get("alias") if provider_row else provider_id
    except Exception:
        provider_alias = provider_id

    facts = top.pop("facts", [])
    summary = " · ".join(facts[:3]) if facts else "(pas de résumé)"

    need_summary = last_user_text.strip()[:500]

    # Create the opt-in in Postgres.
    try:
        row = await create_opt_in(
            seeker_id=user_id,
            provider_id=provider_id,
            need_summary=need_summary or "(no summary)",
            candidate_uuid=candidate_uuid or "no-uuid",
        )
    except Exception:
        logger.exception("Cold Loop: create_opt_in failed")
        return

    if row is None:
        logger.info("Cold Loop: opt-in already proposed earlier for user=%s and provider=%s", user_id, provider_id)
        return

    opt_in_id = str(row["opt_in_id"])
    logger.info("Cold Loop: Created opt_in=%s in state pending_seeker", opt_in_id)

    # 5. Notify the seeker about the candidate asynchronously (hybrid push).
    gateway_payload = {
        "type": "candidates",
        "items": [
            {
                "user_id": provider_id,
                "alias": provider_alias,
                "summary": summary,
                "score": score,
                "candidate_uuid": candidate_uuid,
                "opt_in_id": opt_in_id,
            }
        ],
        "sessionId": opt_in_id,
        "tour": 1,
    }

    base = s.GATEWAY_INTERNAL_URL.rstrip("/")
    url = f"{base}/internal/push/{user_id}"

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.post(url, json=gateway_payload)
            if resp.status_code == 200:
                logger.info("Cold Loop: pushed candidate event to user=%s via Gateway", user_id)
            else:
                logger.warning("Cold Loop: Gateway push returned %s", resp.status_code)
    except Exception:
        logger.exception("Cold Loop: failed to push candidate notification to Gateway")
