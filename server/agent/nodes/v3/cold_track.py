"""Cold Track — fire-and-forget background tasks.

Runs after the hot track responds to the user. Includes:
- Persist episode to Graphiti
- Background orchestrator (cross-group matching)
- Reflection update (if threshold reached)

None of these block the response to the user.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

from agent.db import get_user
from agent.manifests.registry import ManifestRegistry
from agent.models import OryaState
from agent.settings import get_settings

logger = logging.getLogger(__name__)

# In-flight tracking to avoid duplicate reflection runs
_reflection_in_flight: set[str] = set()
_background_tasks: set[asyncio.Task] = set()


async def run_cold_track(
    state: OryaState,
    graphiti: Graphiti,
    manifests: ManifestRegistry,
    llm: Any,  # Runnable — for reflection generation
) -> None:
    """Fire-and-forget all background tasks."""
    user_id = state["user_id"]
    user_text = state.get("last_user_text") or ""
    assistant_reply = state.get("last_assistant_reply") or ""

    # 1. Persist to Graphiti (always)
    task1 = asyncio.create_task(
        _persist_episode(graphiti, user_id, user_text, assistant_reply, state)
    )
    _background_tasks.add(task1)
    task1.add_done_callback(_background_tasks.discard)

    # 2. Background orchestrator (matching) — only if user expressed a need
    if len(user_text.split()) >= 3:
        task2 = asyncio.create_task(
            _run_background_match(user_id, user_text, graphiti)
        )
        _background_tasks.add(task2)
        task2.add_done_callback(_background_tasks.discard)

    # 3. Reflection update — every N turns
    task3 = asyncio.create_task(
        _maybe_trigger_reflection(state, manifests, llm)
    )
    _background_tasks.add(task3)
    task3.add_done_callback(_background_tasks.discard)


# ── Episode persistence ──────────────────────────────────────

async def _persist_episode(
    graphiti: Graphiti,
    user_id: str,
    user_text: str,
    assistant_reply: str,
    state: OryaState,
) -> None:
    if not user_text:
        return

    body = f"{user_id}: {user_text}\nOrya: {assistant_reply}".strip()

    try:
        await graphiti.add_episode(
            name=f"chat:{user_id}:{int(datetime.now().timestamp())}",
            episode_body=body,
            source=EpisodeType.message,
            source_description="orya v3 chat turn",
            reference_time=datetime.now(timezone.utc),
            group_id=user_id,
        )
        logger.info("Graphiti episode persisted user=%s", user_id)
    except Exception:
        logger.exception("Graphiti persist failed user=%s", user_id)


# ── Background matching ──────────────────────────────────────

async def _run_background_match(
    user_id: str, last_text: str, graphiti: Graphiti
) -> None:
    """Delegated to the existing orchestrator."""
    try:
        from ..orchestrator import run_background_orchestrator
        await run_background_orchestrator(
            user_id=user_id,
            last_user_text=last_text,
            graphiti=graphiti,
        )
    except Exception:
        logger.exception("Background orchestrator failed")


# ── Reflection service ───────────────────────────────────────

async def _maybe_trigger_reflection(
    state: OryaState,
    manifests: ManifestRegistry,
    llm: Any,
) -> None:
    """Update reflection documents every REFLECTION_INTERVAL_TURNS."""
    s = get_settings()
    interval = getattr(s, "REFLECTION_INTERVAL_TURNS", 5)

    user_id = state["user_id"]
    if user_id in _reflection_in_flight:
        return

    # For simplicity in v3, we count messages in the current state.
    # In production this should query PG for turn count since last reflection.
    msg_count = len([m for m in state.get("messages", []) if isinstance(m, HumanMessage)])
    if msg_count < interval:
        return

    _reflection_in_flight.add(user_id)
    try:
        await _update_reflections(state, manifests, llm)
    except Exception:
        logger.exception("Reflection update failed")
    finally:
        _reflection_in_flight.discard(user_id)


async def _update_reflections(
    state: OryaState,
    manifests: ManifestRegistry,
    llm: Any,
) -> None:
    """Run reflection-user and reflection-orya agents in parallel."""
    from langchain_core.messages import HumanMessage

    user_id = state["user_id"]
    history = state.get("messages", [])

    # Format recent conversation as text
    convo_text = "\n".join(
        f"{'Utilisateur' if isinstance(m, HumanMessage) else 'Orya'}: {m.content}"
        for m in history[-20:]
    )

    # Load existing reflections
    from ..db import get_reflections
    existing_user_ref, existing_orya_ref = await get_reflections(user_id)

    # Render prompts
    user_prompt = manifests.render("reflection-user")
    orya_prompt = manifests.render("reflection-orya")

    # Build full prompts with context
    user_task = f"{user_prompt}\n\n[Document existant]\n{existing_user_ref or '(vide)'}\n\n[Nouvelle conversation]\n{convo_text}"
    orya_task = f"{orya_prompt}\n\n[Document existant]\n{existing_orya_ref or '(vide)'}\n\n[Nouvelle conversation]\n{convo_text}"

    # Run both in parallel
    user_result, orya_result = await asyncio.gather(
        _run_reflection_agent(llm, user_task),
        _run_reflection_agent(llm, orya_task),
        return_exceptions=True,
    )

    new_user_ref = user_result if not isinstance(user_result, Exception) else existing_user_ref
    new_orya_ref = orya_result if not isinstance(orya_result, Exception) else existing_orya_ref

    # Save to PG
    from ..db import save_reflections
    await save_reflections(
        user_id=user_id,
        user_reflection=new_user_ref,
        orya_reflection=new_orya_ref,
    )
    logger.info("Reflections updated user=%s", user_id)


async def _run_reflection_agent(llm: Any, task_text: str) -> str:
    """Run a single reflection agent."""
    from langchain_core.messages import SystemMessage, HumanMessage
    response = await llm.ainvoke([
        SystemMessage(content="Tu es un agent de synthèse. Sois concis."),
        HumanMessage(content=task_text),
    ])
    return _extract_text(response)


def _extract_text(ai: Any) -> str:
    content = getattr(ai, "content", ai)
    return str(content).strip()
