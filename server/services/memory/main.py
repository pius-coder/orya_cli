"""
Memory Service — Powered by Graphiti (graphiti-core-falkordb).

Graphiti handles EVERYTHING automatically:
- Entity extraction from conversations
- Relationship building between entities
- Temporal tracking (facts change over time)
- Hybrid search (semantic + keyword + graph traversal)
- Episode management (conversation turns)

We just feed it conversations and query it. No manual graph code needed.

Qdrant role: Graphiti uses its OWN vector embeddings internally (stored in FalkorDB).
Qdrant is used as a SEPARATE semantic search layer for quick profile matching
when Graphiti's graph search isn't sufficient (e.g. fuzzy skill matching).
"""

import os
import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Memory Service", version="0.2.0")

# Graphiti instance (initialized on startup)
graphiti = None


# ── Models ─────────────────────────────────────────────────────────
class EpisodeInput(BaseModel):
    userId: str
    text: str
    role: str = "user"  # "user" or "assistant"
    source: str = "conversation"


class SearchRequest(BaseModel):
    userId: str
    query: str
    num_results: int = 10


class FactInput(BaseModel):
    kind: str
    value: str
    confidence: float
    source: str = "inline"
    ts: float = 0


class StoreFactsRequest(BaseModel):
    userId: str
    facts: list[FactInput]


# ── Graphiti Setup ─────────────────────────────────────────────────
async def init_graphiti():
    """Initialize Graphiti with FalkorDB driver and our free LLM providers."""
    global graphiti

    from graphiti_core import Graphiti
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
    from graphiti_core.llm_client import LLMConfig
    from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
    from graphiti_core.driver.falkordb_driver import FalkorDriver

    # FalkorDB connection
    falkor_host = os.getenv("FALKORDB_HOST", "127.0.0.1")
    falkor_port = int(os.getenv("FALKORDB_PORT", "6379"))

    driver = FalkorDriver(
        host=falkor_host,
        port=falkor_port,
        database="orya_memory",
    )

    # LLM: Use Nvidia (OpenAI-compatible) for entity extraction & reasoning
    nvidia_key = os.getenv("NVIDIA_API_KEY", "")
    groq_key = os.getenv("GROQ_API_KEY", "")
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")

    # Pick best available provider for Graphiti's LLM
    if nvidia_key:
        llm_config = LLMConfig(
            api_key=nvidia_key,
            model="meta/llama-4-maverick-17b-128e-instruct",
            small_model="meta/llama-4-maverick-17b-128e-instruct",
            base_url="https://integrate.api.nvidia.com/v1",
        )
    elif groq_key:
        llm_config = LLMConfig(
            api_key=groq_key,
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            small_model="meta-llama/llama-4-scout-17b-16e-instruct",
            base_url="https://api.groq.com/openai/v1",
        )
    elif openrouter_key:
        llm_config = LLMConfig(
            api_key=openrouter_key,
            model="meta-llama/llama-3.3-70b-instruct:free",
            small_model="meta-llama/llama-3.3-70b-instruct:free",
            base_url="https://openrouter.ai/api/v1",
        )
    else:
        print("[memory] WARNING: No LLM API key found, Graphiti will not work!")
        return

    llm_client = OpenAIGenericClient(config=llm_config)

    # Embedder: Use HuggingFace via a compatible endpoint or fallback
    # Graphiti needs embeddings — we use OpenAI-compatible endpoint
    hf_key = os.getenv("HUGGINGFACE_API_KEY", "")
    if nvidia_key:
        # Nvidia has embedding models too
        embedder_config = OpenAIEmbedderConfig(
            api_key=nvidia_key,
            embedding_model="nvidia/nv-embedqa-e5-v5",
            embedding_dim=1024,
            base_url="https://integrate.api.nvidia.com/v1",
        )
    elif openrouter_key:
        # Fallback: use a smaller model
        embedder_config = OpenAIEmbedderConfig(
            api_key=openrouter_key,
            embedding_model="openai/text-embedding-3-small",
            embedding_dim=1536,
            base_url="https://openrouter.ai/api/v1",
        )
    else:
        print("[memory] WARNING: No embedding provider available!")
        return

    embedder = OpenAIEmbedder(config=embedder_config)

    # Initialize Graphiti — pass cross_encoder=None to avoid default OpenAI reranker
    graphiti = Graphiti(
        graph_driver=driver,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=None,
        store_raw_episode_content=True,
    )

    # Build indices on first run
    try:
        await graphiti.build_indices_and_constraints()
        print(f"[memory] Graphiti initialized with FalkorDB at {falkor_host}:{falkor_port}")
    except Exception as e:
        print(f"[memory] Graphiti index build warning: {e}")


# ── Endpoints ──────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "memory",
        "graphiti": graphiti is not None,
        "ts": time.time(),
    }


@app.on_event("startup")
async def startup():
    await init_graphiti()


@app.post("/episode")
async def add_episode(req: EpisodeInput):
    """
    Add a conversation turn to Graphiti.
    Graphiti automatically extracts entities, builds relationships, and updates the graph.
    """
    if not graphiti:
        return {"error": "Graphiti not initialized", "stored": False}

    from graphiti_core.nodes import EpisodeType

    try:
        episode_body = f"{req.role}: {req.text}"

        await graphiti.add_episode(
            name=f"conversation_{req.userId}_{int(time.time())}",
            episode_body=episode_body,
            source_description=f"Conversation with user {req.userId}",
            reference_time=datetime.now(timezone.utc),
            source=EpisodeType.message,
            group_id=req.userId,  # Each user has their own graph partition
        )
        return {"stored": True, "userId": req.userId}
    except Exception as e:
        print(f"[memory] add_episode error: {e}")
        return {"stored": False, "error": str(e)}


@app.post("/facts")
async def store_facts(req: StoreFactsRequest):
    """
    Store extracted facts as episodes in Graphiti.
    Each fact becomes a mini-episode that Graphiti processes into the graph.
    """
    if not graphiti:
        return {"stored": 0, "error": "Graphiti not initialized"}

    from graphiti_core.nodes import EpisodeType

    stored = 0
    for fact in req.facts:
        try:
            # Convert fact to a natural language statement for Graphiti
            fact_text = f"User fact: {fact.kind} is {fact.value}"

            await graphiti.add_episode(
                name=f"fact_{req.userId}_{fact.kind}_{int(time.time())}",
                episode_body=fact_text,
                source_description=f"Extracted fact from conversation ({fact.confidence:.0%} confidence)",
                reference_time=datetime.now(timezone.utc),
                source=EpisodeType.text,
                group_id=req.userId,
            )
            stored += 1
        except Exception as e:
            print(f"[memory] store fact error: {e}")

    return {"stored": stored, "total": len(req.facts)}


@app.get("/facts/{user_id}")
async def get_facts(user_id: str):
    """
    Search Graphiti for all known facts about a user.
    Uses hybrid search (semantic + graph traversal).
    """
    if not graphiti:
        return {"userId": user_id, "facts": []}

    try:
        # Search for facts related to this user
        results = await graphiti.search(
            query=f"What do we know about this person?",
            group_ids=[user_id],
            num_results=20,
        )

        facts = [edge.fact for edge in results if edge.fact]
        return {"userId": user_id, "facts": facts}
    except Exception as e:
        print(f"[memory] get_facts error: {e}")
        return {"userId": user_id, "facts": []}


@app.post("/search")
async def search(req: SearchRequest):
    """
    Hybrid search across the knowledge graph.
    Combines semantic similarity, keyword (BM25), and graph traversal.
    Used by the orchestrator to find matching users/providers.
    """
    if not graphiti:
        return {"candidates": []}

    try:
        # Search across ALL users' graphs (not just one group)
        results = await graphiti.search(
            query=req.query,
            num_results=req.num_results,
        )

        # Extract candidate information from graph edges
        candidates = []
        seen_users = set()

        for edge in results:
            # Try to identify the user from the edge's group_id or node info
            # Graphiti stores group_id which maps to userId
            if hasattr(edge, 'group_id') and edge.group_id:
                uid = edge.group_id
                if uid == req.userId or uid in seen_users:
                    continue
                seen_users.add(uid)
                candidates.append({
                    "userId": uid,
                    "alias": uid,
                    "bio": edge.fact or "",
                    "skills": [],
                    "city": "",
                    "scoreGraph": 1.0 / (len(candidates) + 1),
                    "scoreVector": 0.5,
                    "scoreFused": 0.7 / (len(candidates) + 1),
                    "rank": len(candidates) + 1,
                })

        return {"candidates": candidates[:5]}
    except Exception as e:
        print(f"[memory] search error: {e}")
        return {"candidates": []}


@app.get("/episodes/{user_id}")
async def get_episodes(user_id: str, last_n: int = 10):
    """Retrieve recent episodes for a user."""
    if not graphiti:
        return {"episodes": []}

    try:
        episodes = await graphiti.retrieve_episodes(
            reference_time=datetime.now(timezone.utc),
            last_n=last_n,
            group_ids=[user_id],
        )
        return {
            "episodes": [
                {"name": ep.name, "content": ep.content, "created_at": str(ep.created_at)}
                for ep in episodes
            ]
        }
    except Exception as e:
        print(f"[memory] get_episodes error: {e}")
        return {"episodes": []}
