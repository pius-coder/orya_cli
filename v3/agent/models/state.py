"""LangGraph state definition for Orya v3.

Cleaned up from v2: removes ambiguity between v2 and v3 state shapes.
Uses total=True for stronger type safety within nodes.
"""
from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class OryaState(TypedDict):
    """Shared state flowing through the LangGraph."""

    messages: Annotated[list[AnyMessage], add_messages]
    user_id: str
    user_alias: str
    last_user_text: str
    last_assistant_reply: str

    # Memory
    user_reflection: Optional[str]
    orya_reflection: Optional[str]
    facts_context: Optional[str]

    # Tooling
    tool_calls: Annotated[list[dict[str, Any]], list]
    candidates: Annotated[list[dict[str, Any]], list]

    # Matching
    pending_opt_in: Optional[dict[str, Any]]
    opt_in_response: Optional[dict[str, Any]]

    # Observability
    trace: Annotated[list[dict[str, str]], list]
