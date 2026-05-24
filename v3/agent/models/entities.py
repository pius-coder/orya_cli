"""Typed entity and edge definitions for Graphiti extraction.

Ensures Pydantic-typed extraction is passed to graphiti.add_episode.
This was a regression in v2 cold_track.py which omitted these schemas.
"""
from typing import Optional

from pydantic import BaseModel, Field


# ── Entity Types ──────────────────────────────────────────────────
class Person(BaseModel):
    occupation: Optional[str] = Field(None, description="Job or role")
    location_label: Optional[str] = Field(None, description="City or region")
    age_range: Optional[str] = Field(None, description="Approximate age")


class Skill(BaseModel):
    domain: Optional[str] = Field(None, description="Skill domain")
    seniority: Optional[str] = Field(None, description="Junior / Senior / Expert")


class Need(BaseModel):
    domain: Optional[str] = Field(None, description="What the person needs")
    urgency: Optional[str] = Field(None, description="urgent / moderate / flexible")


class City(BaseModel):
    country: Optional[str] = Field(None, description="Country name")


class Company(BaseModel):
    industry: Optional[str] = Field(None, description="Industry sector")


# ── Edge Types ────────────────────────────────────────────────────
class HasSkill(BaseModel):
    since: Optional[str] = Field(None, description="Year or duration")
    seniority: Optional[str] = Field(None, description="Level")


class Wants(BaseModel):
    expressed_at: Optional[str] = Field(None, description="ISO timestamp")
    urgency: Optional[str] = Field(None, description="Urgency level")


class LocatedIn(BaseModel):
    since: Optional[str] = Field(None, description="Since when")


class WorksAt(BaseModel):
    since: Optional[str] = Field(None, description="Since when")
    role: Optional[str] = Field(None, description="Role at company")


# ── Registries ────────────────────────────────────────────────────
ENTITY_TYPES = [Person, Skill, Need, City, Company]

EDGE_TYPES = [HasSkill, Wants, LocatedIn, WorksAt]

EDGE_TYPE_MAP: dict[tuple[str, str], list[str]] = {
    ("Person", "Skill"): ["HAS_SKILL"],
    ("Person", "Need"): ["WANTS"],
    ("Person", "City"): ["LOCATED_IN"],
    ("Person", "Company"): ["WORKS_AT"],
    ("Entity", "Entity"): ["RELATES_TO"],
}
