"""MemBrain-style Personal Knowledge Graph models.

Lightweight Pydantic models for facts, entities, and trees.
No SQLAlchemy here — we use raw asyncpg for performance.
"""
from typing import Any, Optional

from pydantic import BaseModel, Field


class ExtractedEntity(BaseModel):
    ref: str = Field(description="Canonical reference name, 1-4 words")
    aliases: list[str] = Field(default_factory=list)
    description: str = ""


class NaturalFact(BaseModel):
    text: str = Field(description="Natural language fact with [Entity] references")
    entities: list[str] = Field(description="Entity refs appearing in this fact")
    time_raw: Optional[str] = None
    time_resolved: Optional[str] = None


class EntityTreeNode(BaseModel):
    node_type: str = Field(..., pattern=r"^(root|aspect|leaf)$")
    entity_id: str
    parent_id: Optional[int] = None
    fact_id: Optional[int] = None
    description: Optional[str] = None
    support: int = 0
    fresh_count: int = 0


class SessionSummary(BaseModel):
    session_number: int
    subject: str
    content: str


class MatchIndexEntry(BaseModel):
    user_id: str
    entity_id: str
    canonical_ref: str
    fact_summary: str
    category: Optional[str] = None
