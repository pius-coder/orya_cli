"""LangGraph state schema for Orya agent."""

from __future__ import annotations

from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class OryaState(TypedDict, total=False):
    """Typed dict for the Orya LangGraph state.

    All fields are optional at the type level (`total=False`) but most are
    populated at graph entry by `main.py` before invocation.
    """

    # Conversation history (managed by LangGraph reducer; survives checkpoints)
    messages: Annotated[list[AnyMessage], add_messages]

    # Identity
    user_id: str
    user_alias: Optional[str]

    # Convenience snapshots of last turn (for nodes that don't want to walk
    # the whole `messages` list)
    last_user_text: str
    last_assistant_reply: str

    # Memory retrieval
    facts_context: list[str]

    # Quick-extraction (rule-based) — emitted to client for live UI feedback
    extracted_facts: list[dict[str, Any]]

    # Intent detection
    intent: Optional[dict[str, Any]]

    # Match candidates (cross-group Graphiti search)
    candidates: list[dict[str, Any]]

    # Active opt-in awaiting decision
    pending_opt_in: Optional[dict[str, Any]]

    # Inbound opt-in decision (set by main.py when the user responds)
    opt_in_response: Optional[dict[str, Any]]

    # Trace events accumulated for the response payload
    trace: list[dict[str, Any]]
