"""
Memory Service — Graphiti + FalkorDB + Qdrant.

Responsabilités :
1. Stocker les facts extraits dans le Knowledge Graph (Graphiti/FalkorDB)
2. Rechercher des profils/prestataires par skills + localisation
3. Fournir les facts connus d'un user pour enrichir le contexte Orya
4. Embedding vectoriel via Qdrant pour la recherche sémantique

Le graphe modélise :
- Nodes: User, Skill, City, Need, Fact
- Edges: HAS_SKILL, LIVES_IN, NEEDS, KNOWS, PROVIDES
"""

import os
import time
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from graph_store import GraphStore
from vector_store import VectorStore

load_dotenv()

app = FastAPI(title="Memory Service", version="0.1.0")

# Init stores
graph = GraphStore()
vectors = VectorStore()


# ── Models ─────────────────────────────────────────────────────────
class FactInput(BaseModel):
    kind: str
    value: str
    confidence: float
    source: str = "inline"
    ts: float = 0


class StoreFactsRequest(BaseModel):
    userId: str
    facts: list[FactInput]


class SearchRequest(BaseModel):
    userId: str
    query: str
    skills: list[str] = []
    city: Optional[str] = None


class Candidate(BaseModel):
    userId: str
    alias: str
    bio: str
    skills: list[str]
    city: str
    scoreGraph: float
    scoreVector: float
    scoreFused: float
    rank: int


# ── Endpoints ──────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "memory", "ts": time.time()}


@app.on_event("startup")
async def startup():
    """Initialize connections."""
    await graph.connect()
    await vectors.connect()


@app.get("/facts/{user_id}")
async def get_facts(user_id: str):
    """Return all known facts for a user (used by Agent Orya for context)."""
    facts = await graph.get_user_facts(user_id)
    return {"userId": user_id, "facts": facts}


@app.post("/facts")
async def store_facts(req: StoreFactsRequest):
    """Store extracted facts into graph + vector stores."""
    stored = 0
    for fact in req.facts:
        try:
            # Store in graph (relationships)
            await graph.add_fact(req.userId, fact.kind, fact.value, fact.confidence)

            # If it's a skill → also index in vector for search
            if fact.kind in ("skill", "need"):
                await vectors.index_user_attribute(
                    user_id=req.userId,
                    attribute_type=fact.kind,
                    value=fact.value,
                )
            stored += 1
        except Exception as e:
            print(f"[memory] failed to store fact: {e}")

    return {"stored": stored, "total": len(req.facts)}


@app.post("/search")
async def search(req: SearchRequest):
    """
    Hybrid search: graph traversal + vector similarity.
    Returns ranked candidates.
    """
    # Step 1: Graph search (people with matching skills in same city)
    graph_results = await graph.search_providers(
        skills=req.skills,
        city=req.city,
        exclude_user=req.userId,
    )

    # Step 2: Vector search (semantic similarity to query)
    vector_results = await vectors.search_similar(
        query=req.query,
        skills=req.skills,
        limit=10,
        exclude_user=req.userId,
    )

    # Step 3: Fuse scores
    candidates = _fuse_results(graph_results, vector_results)

    return {"candidates": candidates}


@app.post("/register-profile")
async def register_profile(
    userId: str,
    alias: str = "",
    skills: list[str] = [],
    city: str = "",
    bio: str = "",
    isProvider: bool = False,
):
    """Register or update a user profile in the graph."""
    await graph.upsert_user(userId, alias, skills, city, bio, isProvider)
    # Also index in vector
    if skills:
        for skill in skills:
            await vectors.index_user_attribute(userId, "skill", skill)
    return {"ok": True}


# ── Score Fusion ───────────────────────────────────────────────────
def _fuse_results(
    graph_results: list[dict],
    vector_results: list[dict],
    graph_weight: float = 0.4,
    vector_weight: float = 0.6,
) -> list[dict]:
    """
    Reciprocal Rank Fusion (RRF) of graph and vector results.
    Returns ranked candidates.
    """
    scores: dict[str, dict] = {}

    # Graph results
    for i, r in enumerate(graph_results):
        uid = r["userId"]
        if uid not in scores:
            scores[uid] = {**r, "scoreGraph": 0.0, "scoreVector": 0.0}
        scores[uid]["scoreGraph"] = 1.0 / (i + 1 + 60)  # RRF k=60

    # Vector results
    for i, r in enumerate(vector_results):
        uid = r["userId"]
        if uid not in scores:
            scores[uid] = {**r, "scoreGraph": 0.0, "scoreVector": 0.0}
        scores[uid]["scoreVector"] = 1.0 / (i + 1 + 60)

    # Fuse
    for uid, data in scores.items():
        data["scoreFused"] = (
            data["scoreGraph"] * graph_weight +
            data["scoreVector"] * vector_weight
        )

    # Rank
    ranked = sorted(scores.values(), key=lambda x: x["scoreFused"], reverse=True)
    candidates = []
    for i, r in enumerate(ranked[:5]):  # Top 5
        candidates.append(Candidate(
            userId=r.get("userId", ""),
            alias=r.get("alias", r.get("userId", "")),
            bio=r.get("bio", ""),
            skills=r.get("skills", []),
            city=r.get("city", ""),
            scoreGraph=r.get("scoreGraph", 0),
            scoreVector=r.get("scoreVector", 0),
            scoreFused=r.get("scoreFused", 0),
            rank=i + 1,
        ).dict())

    return candidates
