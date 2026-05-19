"""
Vector Store — Qdrant interface for semantic search.

Stores user attributes (skills, needs) as embeddings for fuzzy matching.
E.g. searching "someone who fixes pipes" matches users with skill "plomberie".

Uses HuggingFace or Nvidia embeddings (free tier).
"""

import os
import hashlib
from typing import Optional

import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchExcept,
)


COLLECTION_NAME = "orya_profiles"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 (fast, free)


class VectorStore:
    def __init__(self):
        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY", "")
        self.hf_api_key = os.getenv("HUGGINGFACE_API_KEY", "")
        self.client: Optional[QdrantClient] = None

    async def connect(self):
        """Connect to Qdrant and ensure collection exists."""
        try:
            self.client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key if self.qdrant_api_key else None,
            )
            # Create collection if not exists
            collections = self.client.get_collections().collections
            names = [c.name for c in collections]
            if COLLECTION_NAME not in names:
                self.client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=EMBEDDING_DIM,
                        distance=Distance.COSINE,
                    ),
                )
            print(f"[vector_store] connected to Qdrant at {self.qdrant_url}")
        except Exception as e:
            print(f"[vector_store] connection failed: {e} — running in degraded mode")
            self.client = None

    async def index_user_attribute(self, user_id: str, attribute_type: str, value: str):
        """Embed and store a user attribute (skill, need, etc.)."""
        if not self.client:
            return

        embedding = await self._embed(value)
        if not embedding:
            return

        # Deterministic point ID from user + attribute
        point_id = self._make_id(f"{user_id}:{attribute_type}:{value}")

        try:
            self.client.upsert(
                collection_name=COLLECTION_NAME,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "userId": user_id,
                            "type": attribute_type,
                            "value": value,
                        },
                    )
                ],
            )
        except Exception as e:
            print(f"[vector_store] upsert failed: {e}")

    async def search_similar(
        self,
        query: str,
        skills: list[str] = [],
        limit: int = 10,
        exclude_user: str = "",
    ) -> list[dict]:
        """Search for users with similar attributes to the query."""
        if not self.client:
            return []

        # Embed the query (combine query + skills)
        search_text = query
        if skills:
            search_text += " " + " ".join(skills)

        embedding = await self._embed(search_text)
        if not embedding:
            return []

        try:
            # Build filter to exclude the requesting user
            search_filter = None
            if exclude_user:
                search_filter = Filter(
                    must_not=[
                        FieldCondition(
                            key="userId",
                            match=MatchExcept(value=[exclude_user]),  # type: ignore
                        )
                    ]
                )

            results = self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=embedding,
                limit=limit * 2,  # Get more to deduplicate per user
                query_filter=search_filter,
            )

            # Deduplicate by user (keep highest score per user)
            seen_users: dict[str, dict] = {}
            for hit in results:
                uid = hit.payload.get("userId", "")  # type: ignore
                if uid == exclude_user:
                    continue
                if uid not in seen_users or hit.score > seen_users[uid].get("score", 0):
                    seen_users[uid] = {
                        "userId": uid,
                        "alias": uid,  # Will be enriched by graph
                        "bio": "",
                        "skills": [],
                        "city": "",
                        "score": hit.score,
                    }
                # Collect skills
                if hit.payload.get("type") == "skill":  # type: ignore
                    if hit.payload["value"] not in seen_users[uid].get("skills", []):  # type: ignore
                        seen_users[uid].setdefault("skills", []).append(hit.payload["value"])  # type: ignore

            # Sort by score and limit
            ranked = sorted(seen_users.values(), key=lambda x: x.get("score", 0), reverse=True)
            return ranked[:limit]

        except Exception as e:
            print(f"[vector_store] search failed: {e}")
            return []

    async def _embed(self, text: str) -> Optional[list[float]]:
        """Get embedding using HuggingFace Inference API (free)."""
        if not self.hf_api_key:
            return None

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2",
                    headers={"Authorization": f"Bearer {self.hf_api_key}"},
                    json={"inputs": text, "options": {"wait_for_model": True}},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    # HF returns nested array for single input
                    if isinstance(data, list) and len(data) > 0:
                        if isinstance(data[0], list):
                            return data[0]
                        return data
                else:
                    print(f"[vector_store] HF embed failed: {resp.status_code}")
        except Exception as e:
            print(f"[vector_store] embed error: {e}")

        return None

    @staticmethod
    def _make_id(text: str) -> str:
        """Create a deterministic UUID-like ID from a string."""
        h = hashlib.md5(text.encode()).hexdigest()
        # Qdrant accepts string IDs
        return h
