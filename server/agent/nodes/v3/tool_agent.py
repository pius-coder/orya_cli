"""Tool-Augmented Agent Node — the core of Orya v3.

Receives:
  - conversation history
  - facts from Graphiti
  - reflection documents
  - pending opt-ins

The LLM decides which tools to call (ReAct cycle).
No rigid pipeline — the agent coordinates.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable

from agent.db import get_good_examples, get_user
from agent.manifests.registry import ManifestRegistry
from agent.models import OryaState
from agent.settings import get_settings
from agent.tools.definitions import AGENT_TOOLS
from agent.tools.executor import ToolExecutor

logger = logging.getLogger(__name__)


def make_tool_agent_node(llm: Runnable, manifests: ManifestRegistry, executor: ToolExecutor):
    """Build the tool agent node.

    The LLM receives tools and can decide to call them. The node handles
    the full ReAct cycle: LLM → tool calls → execution → LLM → final answer.
    """

    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(AGENT_TOOLS)

    async def tool_agent_node(state: OryaState) -> dict[str, Any]:
        user_id = state["user_id"]
        last_text = state.get("last_user_text") or ""
        history = list(state.get("messages") or [])

        # Ensure latest human message is in history
        if (
            last_text
            and (not history or not isinstance(history[-1], HumanMessage)
                 or history[-1].content != last_text)
        ):
            history = history + [HumanMessage(content=last_text)]

        # Load user row (tutoyer, alias)
        try:
            user_row = await get_user(user_id)
        except Exception:
            user_row = None
        tutoyer = bool(user_row.get("tutoyer", True)) if user_row else True
        alias = (user_row or {}).get("alias") or state.get("user_alias")

        # Load good examples from feedback
        try:
            good = await get_good_examples(exclude_user_id=user_id, limit=3)
        except Exception:
            good = []

        # Build dynamic system prompt
        system_prompt = _build_system_prompt(
            manifests=manifests,
            user_text=last_text,
            facts_context=state.get("facts_context") or [],
            user_reflection=state.get("user_reflection"),
            orya_reflection=state.get("orya_reflection"),
            tutoyer=tutoyer,
            alias=alias,
            good_examples=good,
        )

        # ── Step 1: First LLM call with tools ───────────────────
        prompt_messages = [SystemMessage(content=system_prompt)]
        prompt_messages.extend(history[-10:])

        try:
            ai_response = await llm_with_tools.ainvoke(prompt_messages)
        except Exception as e:
            logger.exception("LLM tool agent failed (first call)")
            return _error_response(state, e)

        # ── Step 2: Check if tools were called ──────────────────
        tool_calls = getattr(ai_response, "tool_calls", None) or []

        if not tool_calls:
            # No tools needed — return the response directly
            reply_text = _extract_text(ai_response)
            reply_text = _enforce_brevity(reply_text, tutoyer)
            return _success_response(state, reply_text, tool_calls)

        # ── Step 3: Execute tools ────────────────────────────────
        logger.info("Agent calling %d tool(s): %s", len(tool_calls), [tc.get("name") for tc in tool_calls])

        tool_results = []
        for tc in tool_calls:
            tool_name = tc.get("name")
            tool_args = tc.get("args", {})
            tool_id = tc.get("id", "")

            # Execute the tool via executor
            result_str = await _execute_tool(executor, tool_name, user_id, tool_args)
            tool_results.append({
                "tool_call_id": tool_id,
                "name": tool_name,
                "result": result_str,
            })

        # ── Step 4: Second LLM call with tool results ────────────
        # Add the AI message with tool calls
        prompt_messages.append(ai_response)

        # Add tool result messages
        for tr in tool_results:
            prompt_messages.append(
                ToolMessage(content=tr["result"], tool_call_id=tr["tool_call_id"])
            )

        try:
            final_response = await llm.ainvoke(prompt_messages)
        except Exception as e:
            logger.exception("LLM tool agent failed (second call)")
            return _error_response(state, e)

        reply_text = _extract_text(final_response)
        reply_text = _enforce_brevity(reply_text, tutoyer)

        return _success_response(state, reply_text, tool_results)

    return tool_agent_node


# ── Tool execution ───────────────────────────────────────────

async def _execute_tool(
    executor: ToolExecutor, tool_name: str, user_id: str, args: dict[str, Any]
) -> str:
    """Dispatch tool execution to the executor."""
    try:
        if tool_name == "search_user_memory":
            return await executor.search_user_memory(
                user_id=user_id,
                query=args.get("query", ""),
            )
        elif tool_name == "search_providers":
            return await executor.search_providers(
                query=args.get("query", ""),
                location=args.get("location"),
            )
        elif tool_name == "get_pending_matchings":
            return await executor.get_pending_matchings(user_id=user_id)
        elif tool_name == "get_user_profile":
            return await executor.get_user_profile(user_id=user_id)
        elif tool_name == "record_event":
            return await executor.record_event(
                user_id=user_id,
                event_type=args.get("event_type", ""),
                description=args.get("description", ""),
            )
        else:
            return f"Outil inconnu : {tool_name}"
    except Exception as e:
        logger.exception("Tool execution failed: %s", tool_name)
        return f"Erreur exécution outil {tool_name}: {e}"


# ── Prompt builder ─────────────────────────────────────────────

def _build_system_prompt(
    manifests: ManifestRegistry,
    user_text: str,
    facts_context: list[str],
    user_reflection: str | None,
    orya_reflection: str | None,
    tutoyer: bool,
    alias: str | None,
    good_examples: list[dict[str, Any]],
) -> str:
    """Compose the dynamic system prompt by rendering the manifest template
    and injecting live context."""

    # 1. Base persona from manifest
    base = manifests.render("writer")

    # 2. Dynamic context block
    ctx_lines = ["\n## Contexte dynamique"]

    ctx_lines.append(f"- Utilisateur : {alias or 'inconnu'}")
    ctx_lines.append(f"- Mode : {'TUTOIEMENT' if tutoyer else 'VOUVOIEMENT'}")
    ctx_lines.append(f"- Message actuel : {user_text}")

    # Facts from Graphiti
    if facts_context:
        ctx_lines.append("\n### Faits récupérés du graphe")
        for f in facts_context:
            ctx_lines.append(f"- {f}")

    # Reflection documents
    if user_reflection:
        ctx_lines.append(f"\n### Profil utilisateur (mémoire longue)\n{user_reflection}")
    if orya_reflection:
        ctx_lines.append(f"\n### Notes d'Orya sur cette relation\n{orya_reflection}")

    # Good examples
    if good_examples:
        ctx_lines.append("\n### Exemples de bonnes réponses (style attendu)")
        for ex in good_examples[:2]:
            ctx_lines.append(f"Utilisateur : {ex['user_text']}")
            ctx_lines.append(f"Orya : {ex['assistant_reply']}")

    ctx_block = "\n".join(ctx_lines)

    return f"{base}\n{ctx_block}"


# ── Helpers ──────────────────────────────────────────────────

def _extract_text(ai: Any) -> str:
    """Extract plain text from AIMessage."""
    content = getattr(ai, "content", ai)
    if isinstance(content, list):
        parts = [
            c.get("text", "") if isinstance(c, dict) else str(c)
            for c in content
        ]
        content = "".join(parts)
    return str(content).strip()


def _enforce_brevity(text: str, tutoyer: bool = True) -> str:
    """Hard cap on length."""
    max_words = 25 if tutoyer else 40
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words]).rstrip(",;:.!") + "…"
    return text


def _success_response(state: OryaState, reply_text: str, tool_calls: list[dict]) -> dict:
    return {
        "messages": [AIMessage(content=reply_text)],
        "last_assistant_reply": reply_text,
        "tool_calls": tool_calls,
        "trace": _append_trace(
            state, "tool_agent", f"tools={len(tool_calls)} reply_len={len(reply_text)}"
        ),
    }


def _error_response(state: OryaState, e: Exception) -> dict:
    return {
        "messages": [AIMessage(content="Pardon, j'ai eu un souci. Tu peux répéter ?")],
        "last_assistant_reply": "Pardon, j'ai eu un souci. Tu peux répéter ?",
        "tool_calls": [],
        "trace": _append_trace(state, "tool_agent", f"error: {e}"),
    }


def _append_trace(state: OryaState, step: str, detail: str) -> list[dict]:
    existing = list(state.get("trace") or [])
    existing.append({"step": step, "detail": detail})
    return existing
