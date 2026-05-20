"""ReAct tool agent node: the brain of Orya v3.

Fixes v2 issues:
- Uses ToolExecutor.TOOL_MAP for dispatch (no manual if/elif)
- HumanMessage is properly imported
- Brevity caps are consistent
- get_settings is actually used
"""
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable

from ..core.config import get_settings
from ..core.text import enforce_brevity, extract_text
from ..core.trace import append_trace
from ..db import get_good_examples, get_user
from ..manifests.registry import ManifestRegistry
from ..models import OryaState
from ..tools.executor import ToolExecutor


def make_tool_agent_node(llm: Runnable, manifests: ManifestRegistry, executor: ToolExecutor):
    settings = get_settings()

    async def tool_agent_node(state: OryaState) -> dict[str, Any]:
        user_id = state.get("user_id", "")
        user_alias = state.get("user_alias", "")
        facts_context = state.get("facts_context", "")
        history = list(state.get("messages", []))

        # Ensure last user text is in history
        last_text = state.get("last_user_text", "")
        if not history or getattr(history[-1], "content", None) != last_text:
            history.append(HumanMessage(content=last_text))

        # Load user profile
        try:
            user_row = await get_user(user_id)
            tutoyer = user_row.get("tutoyer", True) if user_row else True
        except Exception:
            tutoyer = True

        # Load good examples
        try:
            good_examples = await get_good_examples(user_id, limit=3)
        except Exception:
            good_examples = []

        # Build system prompt from manifest
        writer_prompt = manifests.render(
            "writer",
            alias=user_alias,
            facts=facts_context,
            tutoyer=tutoyer,
        )

        # Add few-shot examples
        prompt_messages = [SystemMessage(content=writer_prompt)]
        for ex in good_examples[:3]:
            prompt_messages.append(HumanMessage(content=ex["user_input"]))
            prompt_messages.append(AIMessage(content=ex["orya_response"]))
        prompt_messages.extend(history)

        # Bind tools to LLM
        llm_with_tools = llm.bind_tools([t for t in executor.TOOL_MAP.keys()])

        # First LLM call (decision)
        try:
            ai_response = await llm_with_tools.ainvoke(prompt_messages)
        except Exception as e:
            trace = append_trace(state, "tool_agent_error", str(e))
            return {
                "last_assistant_reply": "Oups, j'ai bugué. Tu peux répéter ?",
                "trace": trace,
            }

        tool_calls = getattr(ai_response, "tool_calls", None) or []

        # If no tool calls, direct response
        if not tool_calls:
            reply_text = extract_text(ai_response)
            reply_text = enforce_brevity(reply_text, max_words=70)
            trace = append_trace(state, "tool_agent_direct", f"words={len(reply_text.split())}")
            return {
                "last_assistant_reply": reply_text,
                "trace": trace,
            }

        # Execute tools
        executed = []
        for tc in tool_calls:
            name = tc.get("name") or tc.get("function", {}).get("name", "")
            args = tc.get("args") or tc.get("function", {}).get("arguments", {}) or {}
            if isinstance(args, str):
                import json
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            result = await executor.execute(name, user_id, args)
            executed.append({"name": name, "result": result})
            prompt_messages.append(ToolMessage(content=result, tool_call_id=tc.get("id", "")))

        # Second LLM call with tool results
        prompt_messages.append(ai_response)
        try:
            final_response = await llm.ainvoke(prompt_messages)
        except Exception as e:
            trace = append_trace(state, "tool_agent_final_error", str(e))
            return {
                "last_assistant_reply": "Hmm, j'ai eu un souci. Répète ?",
                "trace": trace,
            }

        reply_text = extract_text(final_response)
        reply_text = enforce_brevity(reply_text, max_words=70)
        trace = append_trace(state, "tool_agent_tools", f"tools={len(executed)}")

        return {
            "last_assistant_reply": reply_text,
            "tool_calls": executed,
            "trace": trace,
        }

    return tool_agent_node
