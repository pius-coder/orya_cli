"""Pydantic schemas for FastAPI request/response validation.

Eliminates dead models (OptInDecision) and fixes typing issues from v2.
"""
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class OptInResponseInput(BaseModel):
    opt_in_id: str
    decision: Literal["accept", "reject"]


class ChatRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=128)
    alias: str = Field(..., min_length=1, max_length=100)
    text: str = Field(..., min_length=1, max_length=4000)
    opt_in_response: Optional[OptInResponseInput] = None


class CandidateOut(BaseModel):
    user_id: str
    alias: Optional[str] = None
    summary: str
    score: float
    candidate_uuid: Optional[str] = None


class TraceEvent(BaseModel):
    step: str
    detail: str


class ChatResponse(BaseModel):
    reply: str = ""
    candidates: list[CandidateOut] = Field(default_factory=list)
    pending_opt_in: Optional[dict[str, Any]] = None
    trace: list[TraceEvent] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    user_id: str
    user_input: str
    orya_response: str
    rating: Literal["good", "bad"]


class FeedbackResponse(BaseModel):
    ok: bool = True


class HealthResponse(BaseModel):
    ok: bool
    services: dict[str, bool]
