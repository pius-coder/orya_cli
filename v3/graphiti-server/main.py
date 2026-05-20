"""Graphiti REST API server (FastAPI).

Minimal REST layer over Graphiti for external tools and admin.
"""
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from .client import init_graphiti

_state: dict[str, Any] = {}


class IngestEpisodeRequest(BaseModel):
    name: str
    episode_body: str
    source: str = "message"
    reference_time: Optional[str] = None
    group_id: str


class SearchRequest(BaseModel):
    query: str
    group_ids: Optional[list[str]] = None
    num_results: int = 10


@asynccontextmanager
async def lifespan(app: FastAPI):
    graphiti = await init_graphiti()
    _state["graphiti"] = graphiti
    yield
    _state.clear()


app = FastAPI(title="Graphiti Server", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"ok": _state.get("graphiti") is not None}


@app.post("/ingest/episodes")
async def ingest_episode(req: IngestEpisodeRequest) -> dict[str, Any]:
    graphiti = _state.get("graphiti")
    if not graphiti:
        return {"error": "Graphiti not initialized"}

    from graphiti_core.nodes import EpisodeType

    ref_time = _parse_iso(req.reference_time) if req.reference_time else datetime.utcnow()
    episode_type = EpisodeType.message if req.source == "message" else EpisodeType.text

    await graphiti.add_episode(
        name=req.name,
        episode_body=req.episode_body,
        source=episode_type,
        reference_time=ref_time,
        group_id=req.group_id,
    )
    return {"ok": True}


@app.post("/retrieve/search")
async def search(req: SearchRequest) -> dict[str, Any]:
    graphiti = _state.get("graphiti")
    if not graphiti:
        return {"error": "Graphiti not initialized"}

    results = await graphiti.search(
        query=req.query,
        group_ids=req.group_ids,
        num_results=req.num_results,
    )
    serialized = []
    for r in results:
        serialized.append({
            "uuid": getattr(r, "uuid", None),
            "fact": getattr(r, "fact", str(r)),
            "group_id": getattr(r, "group_id", None),
            "valid_at": getattr(r, "valid_at", None),
            "invalid_at": getattr(r, "invalid_at", None),
        })
    return {"results": serialized}


@app.post("/admin/build-indices")
async def build_indices() -> dict[str, Any]:
    graphiti = _state.get("graphiti")
    if not graphiti:
        return {"error": "Graphiti not initialized"}
    await graphiti.build_indices_and_constraints()
    return {"ok": True}


def _parse_iso(value: Optional[str]) -> datetime:
    if not value:
        return datetime.utcnow()
    # Handle both ISO 8601 and simple formats
    value = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.utcnow()
