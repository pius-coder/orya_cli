"""Tool definitions for the Tool-Augmented Agent.

These are LangChain-compatible tool schemas. The LLM receives these schemas
and decides which ones to invoke. The actual execution happens in the agent node
via ToolExecutor.
"""

from __future__ import annotations

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class SearchUserMemoryInput(BaseModel):
    query: str = Field(description="Search query for user's memory / past conversations")
    user_id: str = Field(description="ID of the user whose memory to search")


class SearchProvidersInput(BaseModel):
    query: str = Field(description="What the user is looking for (skill, service, need)")
    location: str | None = Field(default=None, description="Optional location constraint")


class GetPendingMatchingsInput(BaseModel):
    user_id: str = Field(description="ID of the user to check matchings for")


class GetUserProfileInput(BaseModel):
    user_id: str = Field(description="ID of the user whose profile to retrieve")


class RecordEventInput(BaseModel):
    user_id: str = Field(description="ID of the user this event concerns")
    event_type: str = Field(description="Type of event (e.g. 'search', 'preference', 'contact')")
    description: str = Field(description="Description of what happened")


# Dummy functions — real execution is in ToolExecutor
def _noop(**kwargs) -> str:
    return "[tool called]"


AGENT_TOOLS = [
    StructuredTool.from_function(
        name="search_user_memory",
        description="Search the user's long-term memory (Graphiti) for facts, past conversations, and relationships relevant to the query. Use when the user references the past or asks 'do you remember'.",
        args_schema=SearchUserMemoryInput,
        func=_noop,
        return_direct=False,
    ),
    StructuredTool.from_function(
        name="search_providers",
        description="Search for people across the network matching the given criteria. Use when the user is looking for a specific skill, service, or professional.",
        args_schema=SearchProvidersInput,
        func=_noop,
        return_direct=False,
    ),
    StructuredTool.from_function(
        name="get_pending_matchings",
        description="Get any pending matchings (opt-ins) awaiting the user's decision. Use when the user asks about previous proposals or contacts.",
        args_schema=GetPendingMatchingsInput,
        func=_noop,
        return_direct=False,
    ),
    StructuredTool.from_function(
        name="get_user_profile",
        description="Get the user's stored profile including alias, preferences, and reflection documents. Use to personalize the response.",
        args_schema=GetUserProfileInput,
        func=_noop,
        return_direct=False,
    ),
    StructuredTool.from_function(
        name="record_event",
        description="Record a significant event or fact about the user for future reference. Use when the user shares important personal info.",
        args_schema=RecordEventInput,
        func=_noop,
        return_direct=False,
    ),
]
