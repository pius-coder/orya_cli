"""Persist episode + trigger background matching, reflections, AND MemBrain ingest.

Updated to pass embedder for MemBrain ingestion.
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import HumanMessage

from graphiti_core import EpisodeType, Graphiti

from ..core.config import get_settings
from ..core.trace import append_trace
from ..db import get_user, save_reflections
from ..matching import run_background_matching
from ..memory import ingest_conversation
from ..models import OryaState
from ..models.entities import EDGE_TYPE_MAP, EDGE_TYPES, ENTITY_TYPES
from ..providers.embedder import HuggingFaceEmbedder

logger = logging.getLogger(__name__)

_background_tasks: set[asyncio.Task] = set()
_reflection_in_flight: set[str] = set()

REFLECTION_INTERVAL_TURNS = 5


def make_persist_and_match_node(graphiti: Graphiti, llm: Any, embedder: HuggingFaceEmbedder | None = None):
    async def persist_and_match_node(state: OryaState) -> dict[str, Any]:
        user_id = state.get("user_id", "")
        user_text = state.get("last_user_text", "")
        assistant_reply = state.get("last_assistant_reply", "")
        strategy = state.get("strategy", "chat")

        task = asyncio.create_task(
            _cold_track(user_id, user_text, assistant_reply, state, graphiti, llm, embedder, strategy)
        )
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
        task.add_done_callback(_log_task_error)

        trace = append_trace(state, "persist_and_match", "cold track queued")
        return {"trace": trace}

    return persist_and_match_node


def _log_task_error(task: asyncio.Task) -> None:
    if task.done() and not task.cancelled():
        exc = task.exception()
        if exc:
            logger.error("Background task failed: %s", exc)


async def _cold_track(
    user_id: str,
    user_text: str,
    assistant_reply: str,
    state: OryaState,
    graphiti: Graphiti,
    llm: Any,
    embedder: HuggingFaceEmbedder | None,
    strategy: str,
) -> None:
    await _persist_episode(graphiti, user_id, user_text, assistant_reply)

    messages = state.get("messages", [])
    ingest_msgs = []
    for msg in messages[-10:]:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        ingest_msgs.append({"speaker": role, "content": str(getattr(msg, "content", ""))})
    if ingest_msgs:
        try:
            await ingest_conversation(user_id, ingest_msgs, llm, embedder=embedder)
        except Exception as e:
            logger.error("MemBrain ingest failed: %s", e)

    if strategy != "match":
        try:
            await run_background_matching(user_id, user_text, graphiti)
        except Exception as e:
            logger.error("Background matching failed: %s", e)

    await _maybe_trigger_reflection(state, user_id, llm)


async def _persist_episode(
    graphiti: Graphiti, user_id: str, user_text: str, assistant_reply: str
) -> None:
    body = f"User: {user_text}\nOrya: {assistant_reply}"
    try:
        await graphiti.add_episode(
            name=f"turn-{datetime.now(timezone.utc).isoformat()}",
            episode_body=body,
            source=EpisodeType.message,
            reference_time=datetime.now(timezone.utc),
            group_id=user_id,
            entity_types=ENTITY_TYPES,
            edge_types=EDGE_TYPES,
            edge_type_map=EDGE_TYPE_MAP,
        )
        logger.info("Episode persisted for user=%s", user_id)
    except Exception as e:
        logger.error("Failed to persist episode: %s", e)


async def _maybe_trigger_reflection(state: OryaState, user_id: str, llm: Any) -> None:
    if user_id in _reflection_in_flight:
        return

    messages = state.get("messages", [])
    msg_count = sum(1 for m in messages if isinstance(m, HumanMessage))

    if msg_count > 0 and msg_count % REFLECTION_INTERVAL_TURNS == 0:
        _reflection_in_flight.add(user_id)
        try:
            await _update_reflections(state, user_id, llm)
        finally:
            _reflection_in_flight.discard(user_id)


async def _update_reflections(state: OryaState, user_id: str, llm: Any) -> None:
    from ..providers import build_llm

    settings = get_settings()
    reflection_llm = build_llm(temperature=0.2, max_tokens=512)

    from ..db import get_reflections

    existing = await get_reflections(user_id)
    history_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Orya'}: {m.content}"
        for m in state.get("messages", [])[-20:]
    )

    user_prompt = (
        f"Mets à jour le portrait de cet utilisateur.\n"
        f"Portrait existant:\n{existing.get('user_reflection') or '(vide)'}\n\n"
        f"Nouvelle conversation:\n{history_text}\n"
    )
    try:
        user_ref_raw = await reflection_llm.ainvoke([HumanMessage(content=user_prompt)])
        user_reflection = str(getattr(user_ref_raw, "content", "")).strip()
    except Exception as e:
        logger.error("User reflection failed: %s", e)
        user_reflection = existing.get("user_reflection")

    orya_prompt = (
        f"Mets à jour les notes sur la relation avec cet utilisateur.\n"
        f"Notes existantes:\n{existing.get('orya_reflection') or '(vide)'}\n\n"
        f"Nouvelle conversation:\n{history_text}\n"
    )
    try:
        orya_ref_raw = await reflection_llm.ainvoke([HumanMessage(content=orya_prompt)])
        orya_reflection = str(getattr(orya_ref_raw, "content", "")).strip()
    except Exception as e:
        logger.error("Orya reflection failed: %s", e)
        orya_reflection = existing.get("orya_reflection")

    await save_reflections(user_id, user_reflection, orya_reflection)
    logger.info("Reflections updated for user=%s", user_id)
