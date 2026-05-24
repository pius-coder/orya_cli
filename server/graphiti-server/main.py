"""Standalone Graphiti REST server.

Exposes the most useful Graphiti operations over HTTP for inspection,
external tools, and admin tasks. The agent itself uses Graphiti directly
via Python, not through this server.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from graphiti_core.nodes import EpisodeType
from pydantic import BaseModel, Field

from client import init_graphiti

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("orya.graphiti")

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    g = await init_graphiti()
    _state["graphiti"] = g
    try:
        yield
    finally:
        _state.pop("graphiti", None)


app = FastAPI(title="Orya Graphiti Server", version="2.0.0", lifespan=lifespan)


class IngestEpisodeRequest(BaseModel):
    name: str = Field(..., min_length=1)
    body: str = Field(..., min_length=1)
    source: str = Field(
        "message",
        pattern="^(message|text|json)$",
        description="One of: message, text, json",
    )
    source_description: str = ""
    reference_time: Optional[str] = Field(
        None, description="ISO 8601; defaults to now"
    )
    group_id: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    group_ids: Optional[list[str]] = None
    center_node_uuid: Optional[str] = None
    num_results: int = Field(10, ge=1, le=50)


def _epoch_to_dt(value: Optional[str]) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(value)
    except ValueError as e:
        raise HTTPException(400, f"Invalid reference_time: {e}") from e


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"ok": _state.get("graphiti") is not None}


@app.post("/ingest/episodes")
async def ingest_episode(req: IngestEpisodeRequest) -> dict[str, Any]:
    g = _state.get("graphiti")
    if g is None:
        raise HTTPException(503, "graphiti not ready")
    src = {
        "message": EpisodeType.message,
        "text": EpisodeType.text,
        "json": EpisodeType.json,
    }[req.source]
    await g.add_episode(
        name=req.name,
        episode_body=req.body,
        source=src,
        source_description=req.source_description,
        reference_time=_epoch_to_dt(req.reference_time),
        group_id=req.group_id,
    )
    return {"ok": True}


@app.post("/retrieve/search")
async def search(req: SearchRequest) -> dict[str, Any]:
    g = _state.get("graphiti")
    if g is None:
        raise HTTPException(503, "graphiti not ready")
    edges = await g.search(
        query=req.query,
        center_node_uuid=req.center_node_uuid,
        group_ids=req.group_ids,
        num_results=req.num_results,
    )
    return {
        "results": [
            {
                "uuid": getattr(e, "uuid", None),
                "fact": getattr(e, "fact", None),
                "group_id": getattr(e, "group_id", None),
                "valid_at": getattr(e, "valid_at", None) and str(e.valid_at),
                "invalid_at": getattr(e, "invalid_at", None) and str(e.invalid_at),
            }
            for e in (edges or [])
        ]
    }


@app.post("/admin/build-indices")
async def build_indices() -> dict[str, Any]:
    g = _state.get("graphiti")
    if g is None:
        raise HTTPException(503, "graphiti not ready")
    await g.build_indices_and_constraints()
    return {"ok": True}
