"""Pydantic entity & edge type definitions for Graphiti.

These types are passed to `graphiti.add_episode(entity_types=…, edge_types=…,
edge_type_map=…)` so the LLM extracts a typed graph that matches the Orya
domain (people offering / seeking help in a community).

Constraints:
- Attributes must NOT collide with EntityNode protected names
  (`uuid`, `name`, `group_id`, `labels`, `created_at`, `summary`, `attributes`,
  `name_embedding`).
- Each attribute should be `Optional[…]` because Graphiti will populate them
  only when present in the source text.
- Descriptions are read by the LLM at extraction time — keep them sharp.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ============================================================
# Entity types
# ============================================================


class Person(BaseModel):
    """A real person (user, contact, public figure)."""

    occupation: Optional[str] = Field(
        None, description="Current professional role or job title"
    )
    location_label: Optional[str] = Field(
        None, description="Free-text current city / region label"
    )
    age_range: Optional[str] = Field(
        None,
        description="Coarse age bucket: 'student', 'young_pro', 'adult', 'senior'",
    )


class Skill(BaseModel):
    """A professional or technical skill."""

    domain: Optional[str] = Field(
        None,
        description="High-level domain, e.g. 'software_dev', 'plumbing', 'fiscal_law'",
    )
    seniority: Optional[str] = Field(
        None, description="'junior', 'intermediate', 'senior', 'expert'"
    )


class Need(BaseModel):
    """A need expressed by a user (looking for help, a service, or a contact)."""

    domain: Optional[str] = Field(
        None, description="Domain of the need (mirrors Skill.domain)"
    )
    urgency: Optional[str] = Field(
        None,
        description="'immediate', 'days', 'weeks', 'exploratory'",
    )
    location_label: Optional[str] = Field(
        None, description="Where the need must be solved"
    )


class City(BaseModel):
    """A city, town, or named geographical area."""

    country: Optional[str] = Field(
        None, description="Country (ISO name or common name)"
    )


class Company(BaseModel):
    """A business, organization, or NGO."""

    industry: Optional[str] = Field(None, description="Sector / industry")


# ============================================================
# Edge types
# ============================================================


class HasSkill(BaseModel):
    """Person possesses a Skill."""

    seniority: Optional[str] = Field(None, description="Skill level")
    years: Optional[int] = Field(
        None, description="Years of practice in this skill"
    )


class Wants(BaseModel):
    """Person expressed a Need."""

    expressed_at: Optional[str] = Field(
        None, description="ISO timestamp at which the need was expressed"
    )


class LocatedIn(BaseModel):
    """Person / Need / Company is located in a City."""


class WorksAt(BaseModel):
    """Person works at a Company."""

    role: Optional[str] = Field(None, description="Role at the company")


# ============================================================
# Registry — passed to add_episode
# ============================================================


ENTITY_TYPES: dict[str, type[BaseModel]] = {
    "Person": Person,
    "Skill": Skill,
    "Need": Need,
    "City": City,
    "Company": Company,
}


EDGE_TYPES: dict[str, type[BaseModel]] = {
    "HasSkill": HasSkill,
    "Wants": Wants,
    "LocatedIn": LocatedIn,
    "WorksAt": WorksAt,
}


EDGE_TYPE_MAP: dict[tuple[str, str], list[str]] = {
    ("Person", "Skill"): ["HasSkill"],
    ("Person", "Need"): ["Wants"],
    ("Person", "City"): ["LocatedIn"],
    ("Person", "Company"): ["WorksAt"],
    ("Need", "City"): ["LocatedIn"],
    ("Company", "City"): ["LocatedIn"],
    # Fallback for any other combination so Graphiti still records facts.
    ("Entity", "Entity"): ["RELATES_TO"],
}
