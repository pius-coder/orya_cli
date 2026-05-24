"""MemBrain-style memory ingestion node.

Fire-and-forget ingestion of the conversation into the user's PKG.
"""
import asyncio
import logging
from typing import Any

from langchain_core.runnables import Runnable

from ..core.trace import append_trace
from ..memory import ingest_conversation
from ..models import OryaState

logger = logging.getLogger(__name__)

_background_tasks: set[asyncio.Task] = set()


def make_ingest_memory_node(llm: Runnable):
    async def ingest_memory_node(state: OryaState) -> dict[str, Any]:
        user_id = state.get("user_id", "")
        messages = state.get("messages", [])

        # Build message list for ingestion
        ingest_msgs = []
        for msg in messages[-10:]:  # Last 10 messages
            role = "user" if getattr(msg, "type", "") == "human" else "assistant"
            ingest_msgs.append({"speaker": role, "content": str(getattr(msg, "content", ""))})

        if not ingest_msgs:
            trace = append_trace(state, "ingest_memory", "no messages")
            return {"trace": trace}

        # Fire-and-forget
        task = asyncio.create_task(
            _safe_ingest(user_id, ingest_msgs, llm)
        )
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
        task.add_done_callback(_log_error)

        trace = append_trace(state, "ingest_memory", "queued")
        return {"trace": trace}

    return ingest_memory_node


async def _safe_ingest(user_id: str, messages: list[dict[str, str]], llm: Runnable) -> None:
    try:
        result = await ingest_conversation(user_id, messages, llm)
        logger.info("MemBrain ingest result: %s", result)
    except Exception as e:
        logger.error("MemBrain ingest failed: %s", e)


def _log_error(task: asyncio.Task) -> None:
    if task.done() and not task.cancelled():
        exc = task.exception()
        if exc:
            logger.error("Ingest background task failed: %s", exc)
