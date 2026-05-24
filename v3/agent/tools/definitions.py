"""Tool definitions for the ReAct agent.

Schemas are declared as StructuredTool with no-op functions.
The real execution lives in tools.executor.ToolExecutor.
This separation prevents circular imports.
"""
from typing import Any, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class SearchUserMemoryInput(BaseModel):
    query: str = Field(description="Search query for the user's own memory graph")


class SearchProvidersInput(BaseModel):
    query: str = Field(description="What the user is looking for")
    location: Optional[str] = Field(None, description="Optional location filter")


class GetPendingMatchingsInput(BaseModel):
    pass


class GetUserProfileInput(BaseModel):
    pass


class RecordEventInput(BaseModel):
    event_type: str = Field(description="Type of event to record")
    description: str = Field(description="Description of the event")


def _noop(**kwargs: Any) -> str:
    return "[tool called]"


AGENT_TOOLS = [
    StructuredTool.from_function(
        name="search_user_memory",
        description="Search the user's own facts and history in the knowledge graph.",
        args_schema=SearchUserMemoryInput,
        func=_noop,
    ),
    StructuredTool.from_function(
        name="search_providers",
        description="Search for other users who might match the current need. Use ONLY when the user is looking for someone or something.",
        args_schema=SearchProvidersInput,
        func=_noop,
    ),
    StructuredTool.from_function(
        name="get_pending_matchings",
        description="Get the list of pending matchings (opt-ins) for this user.",
        args_schema=GetPendingMatchingsInput,
        func=_noop,
    ),
    StructuredTool.from_function(
        name="get_user_profile",
        description="Get a summary of the user's known profile.",
        args_schema=GetUserProfileInput,
        func=_noop,
    ),
    StructuredTool.from_function(
        name="record_event",
        description="Record a notable event in the user's timeline.",
        args_schema=RecordEventInput,
        func=_noop,
    ),
]
