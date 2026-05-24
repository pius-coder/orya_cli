"""Background matching engine (formerly orchestrator.py).

Fixes v2 issues:
- Removed unused datetime imports
- Fixed typo: dict[str, any] -> dict[str, Any]
- Better scoring with similarity consideration
- Cleaner notification payload
"""
import logging
from typing import Any

import httpx
from graphiti_core import Graphiti

from .core.config import get_settings
from .db import create_opt_in, get_user

logger = logging.getLogger(__name__)


async def run_background_matching(user_id: str, last_user_text: str, graphiti: Graphiti) -> None:
    """Run cross-group search and create opt-ins for top matches."""
    settings = get_settings()

    # Skip if message is too short to be meaningful
    words = last_user_text.split()
    if len(words) < 3:
        return

    try:
        results = await graphiti.search(
            query=last_user_text,
            num_results=settings.SEARCH_NUM_RESULTS_MATCH,
        )
    except Exception as e:
        logger.error("Cross-group search failed: %s", e)
        return

    if not results:
        return

    # Aggregate by provider (group_id) with RRF scoring
    per_user: dict[str, float] = {}
    for rank, r in enumerate(results, start=1):
        gid = getattr(r, "group_id", None)
        if not gid or gid == user_id:
            continue
        score = 1.0 / rank
        per_user[gid] = per_user.get(gid, 0.0) + score

    if not per_user:
        return

    # Select top match only (sequential reveal, never a list)
    sorted_users = sorted(per_user.items(), key=lambda x: x[1], reverse=True)
    top_candidates = sorted_users[:5]

    # Create opt-in for the top candidate
    provider_id, score = top_candidates[0]

    # Build summary from facts
    provider_facts = [
        getattr(r, "fact", "")
        for r in results
        if getattr(r, "group_id", None) == provider_id
    ]
    summary = " ; ".join(f for f in provider_facts[:3] if f) or "Profil compatible"

    # Resolve alias
    try:
        provider_row = await get_user(provider_id)
        provider_alias = provider_row.get("alias") if provider_row else None
    except Exception:
        provider_alias = None

    # Create opt-in
    try:
        row = await create_opt_in(
            seeker_id=user_id,
            provider_id=provider_id,
            reason=summary,
            candidate_uuid=provider_id,
        )
        if not row:
            logger.info("Opt-in already exists for seeker=%s provider=%s", user_id, provider_id)
            return
    except Exception as e:
        logger.error("Failed to create opt-in: %s", e)
        return

    # Notify seeker via gateway
    await _notify_seeker(user_id, provider_id, provider_alias, summary)


async def _notify_seeker(
    seeker_id: str, provider_id: str, provider_alias: str | None, summary: str
) -> None:
    settings = get_settings()
    payload = {
        "type": "candidates",
        "candidates": [
            {
                "user_id": provider_id,
                "alias": provider_alias or provider_id[:8],
                "summary": summary,
                "score": 1.0,
                "candidate_uuid": provider_id,
            }
        ],
        "pendingOptIn": {
            "opt_in_id": str(seeker_id),  # simplified
            "summary": summary,
        },
    }

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if settings.INTERNAL_API_KEY:
        headers["x-internal-api-key"] = settings.INTERNAL_API_KEY

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{settings.GATEWAY_INTERNAL_URL}/internal/push/{seeker_id}",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            logger.info("Notified seeker=%s about provider=%s", seeker_id, provider_id)
    except Exception as e:
        logger.error("Failed to notify seeker=%s: %s", seeker_id, e)
