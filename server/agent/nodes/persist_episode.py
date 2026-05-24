"""Node: persist_episode.

Sends the full user↔Orya turn to Graphiti as an `EpisodeType.message`
episode, scoped to the user's group_id. Graphiti will then asynchronously
extract entities and edges using the typed Pydantic schemas.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

from ..models import OryaState
from ..models.entities import EDGE_TYPE_MAP, EDGE_TYPES, ENTITY_TYPES

logger = logging.getLogger(__name__)


def make_persist_episode_node(graphiti: Graphiti):
    async def persist_episode_node(state: OryaState) -> dict[str, Any]:
        user_id = state["user_id"]
        user_text = state.get("last_user_text") or ""
        assistant_reply = state.get("last_assistant_reply") or ""
        if not user_text:
            return {"trace": _append_trace(state, "persist_episode", "skipped (no text)")}

        speaker = state.get("user_alias") or user_id
        body = f"{speaker}: {user_text}\nOrya: {assistant_reply}".strip()

        # We fire-and-forget so the persona reply latency doesn't depend on
        # entity extraction. Errors are logged, never re-raised to the user.
        async def _ingest() -> None:
            try:
                await graphiti.add_episode(
                    name=f"chat:{user_id}:{int(datetime.now().timestamp())}",
                    episode_body=body,
                    source=EpisodeType.message,
                    source_description="orya chat turn",
                    reference_time=datetime.now(timezone.utc),
                    group_id=user_id,
                    entity_types=ENTITY_TYPES,
                    edge_types=EDGE_TYPES,
                    edge_type_map=EDGE_TYPE_MAP,
                )
                # Run the background orchestrator for matching
                from ..orchestrator import run_background_orchestrator
                await run_background_orchestrator(
                    user_id=user_id,
                    last_user_text=user_text,
                    graphiti=graphiti,
                )
            except Exception:
                logger.exception("Graphiti.add_episode or orchestrator failed for user=%s", user_id)

        asyncio.create_task(_ingest())
        return {"trace": _append_trace(state, "persist_episode", "queued")}

    return persist_episode_node


def _append_trace(state: OryaState, step: str, detail: str) -> list[dict[str, Any]]:
    existing = list(state.get("trace") or [])
    existing.append({"step": step, "detail": detail})
    return existing
