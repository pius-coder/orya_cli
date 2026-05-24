"""HTTP request/response Pydantic schemas for the Agent FastAPI server."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class OptInResponseInput(BaseModel):
    """Embedded in `ChatRequest` when the user is responding to a previously
    proposed opt-in."""

    opt_in_id: str
    decision: Literal["accept", "reject"]


class ChatRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=128)
    alias: Optional[str] = Field(None, max_length=128)
    text: str = Field(..., min_length=1, max_length=4000)
    opt_in_response: Optional[OptInResponseInput] = None


class ExtractedFactOut(BaseModel):
    label: str
    value: str
    confidence: float


class CandidateOut(BaseModel):
    user_id: str
    alias: Optional[str] = None
    summary: str
    score: float
    candidate_uuid: str


class TraceEvent(BaseModel):
    step: str
    detail: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    facts: list[ExtractedFactOut] = Field(default_factory=list)
    candidates: list[CandidateOut] = Field(default_factory=list)
    pending_opt_in: Optional[dict[str, Any]] = None
    trace: list[TraceEvent] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    user_id: str
    user_text: str
    assistant_reply: str
    rating: Literal[-1, 1]


class FeedbackResponse(BaseModel):
    ok: bool = True


class HealthResponse(BaseModel):
    ok: bool = True
    services: dict[str, bool]


class OptInDecision(BaseModel):
    """Standalone endpoint response when an opt_in is processed (out-of-graph)."""

    opt_in_id: str
    new_status: str
    notified_provider: bool = False
    notified_seeker: bool = False
