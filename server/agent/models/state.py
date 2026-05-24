"""LangGraph state schema for Orya v3 — simplified.

No heavy qualifier. The Tool Agent LLM decides what to do.
"""

from __future__ import annotations

from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class OryaState(TypedDict, total=False):
    """Typed dict for the Orya v3 LangGraph state.

    v3 changes:
    - reflections: user_reflection + orya_reflection documents
    - tool_calls: record of which tools the agent decided to invoke
    """

    # Conversation history (managed by LangGraph reducer; survives checkpoints)
    messages: Annotated[list[AnyMessage], add_messages]

    # Identity
    user_id: str
    user_alias: Optional[str]

    # Convenience snapshots of last turn
    last_user_text: str
    last_assistant_reply: str

    # v3: Reflection documents (loaded from PG or empty)
    user_reflection: Optional[str]
    orya_reflection: Optional[str]

    # v3: Tool call trace (which tools the agent used)
    tool_calls: list[dict[str, Any]]

    # Memory retrieval (populated by tools or retrieve_context)
    facts_context: list[str]

    # Match candidates (populated by search_providers tool)
    candidates: list[dict[str, Any]]

    # Active opt-in awaiting decision
    pending_opt_in: Optional[dict[str, Any]]

    # Inbound opt-in decision (set by main.py when the user responds)
    opt_in_response: Optional[dict[str, Any]]

    # Trace events accumulated for the response payload
    trace: list[dict[str, Any]]
