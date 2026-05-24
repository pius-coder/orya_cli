"""Qdrant vector database client for Orya.

Used for storing and searching fact embeddings across all users.
Much faster than PostgreSQL JSONB for cosine similarity queries.

Collections:
- orya_facts : fact embeddings (one point per fact)
- orya_match_index : match index embeddings for cross-user search
"""
import logging
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from ..core.config import get_settings

logger = logging.getLogger(__name__)

_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    """Get or create the global Qdrant client."""
    global _client
    if _client is None:
        settings = get_settings()
        kwargs = {"url": settings.qdrant_url}
        if settings.QDRANT_API_KEY:
            kwargs["api_key"] = settings.QDRANT_API_KEY
        _client = QdrantClient(**kwargs)
        _ensure_collections()
    return _client


def _ensure_collections() -> None:
    """Create Qdrant collections if they don't exist."""
    settings = get_settings()
    dim = settings.EMBEDDING_DIM

    collections = ["orya_facts", "orya_match_index"]
    for name in collections:
        if not _client.collection_exists(name):
            _client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection: %s", name)


def upsert_fact(
    fact_id: int,
    user_id: str,
    text: str,
    embedding: list[float],
    entity_ids: list[str],
    category: str = "general",
) -> None:
    """Upsert a fact embedding into Qdrant."""
    client = get_qdrant_client()
    client.upsert(
        collection_name="orya_facts",
        points=[
            PointStruct(
                id=fact_id,
                vector=embedding,
                payload={
                    "user_id": user_id,
                    "text": text,
                    "entity_ids": entity_ids,
                    "category": category,
                },
            )
        ],
    )


def search_facts(
    user_id: str,
    query_embedding: list[float],
    top_k: int = 10,
    min_score: float = 0.5,
) -> list[dict[str, Any]]:
    """Search facts for a specific user using cosine similarity.

    Returns list of {id, text, score, entity_ids, category}
    """
    client = get_qdrant_client()
    results = client.search(
        collection_name="orya_facts",
        query_vector=query_embedding,
        query_filter={
            "must": [
                {"key": "user_id", "match": {"value": user_id}}
            ]
        },
        limit=top_k,
        score_threshold=min_score,
    )
    return [
        {
            "id": r.id,
            "text": r.payload.get("text", ""),
            "score": r.score,
            "entity_ids": r.payload.get("entity_ids", []),
            "category": r.payload.get("category", "general"),
        }
        for r in results
    ]


def search_facts_cross_user(
    query_embedding: list[float],
    exclude_user_ids: list[str],
    top_k: int = 10,
    min_score: float = 0.5,
) -> list[dict[str, Any]]:
    """Search facts across ALL users (for matching).

    Returns list of {id, user_id, text, score, entity_ids, category}
    """
    client = get_qdrant_client()
    results = client.search(
        collection_name="orya_facts",
        query_vector=query_embedding,
        query_filter={
            "must_not": [
                {"key": "user_id", "match": {"value": uid}}
                for uid in exclude_user_ids
            ]
        },
        limit=top_k,
        score_threshold=min_score,
    )
    return [
        {
            "id": r.id,
            "user_id": r.payload.get("user_id", ""),
            "text": r.payload.get("text", ""),
            "score": r.score,
            "entity_ids": r.payload.get("entity_ids", []),
            "category": r.payload.get("category", "general"),
        }
        for r in results
    ]


def delete_facts_by_user(user_id: str) -> None:
    """Delete all facts for a user (e.g., on account deletion)."""
    client = get_qdrant_client()
    client.delete(
        collection_name="orya_facts",
        points_selector={
            "filter": {
                "must": [
                    {"key": "user_id", "match": {"value": user_id}}
                ]
            }
        },
    )


# ── Match Index (cross-user) ─────────────────────────────────────


def upsert_match_index(
    user_id: str,
    entity_id: str,
    fact_summary: str,
    embedding: list[float],
    category: str = "general",
) -> None:
    """Upsert a match index entry into Qdrant."""
    client = get_qdrant_client()
    point_id = f"{user_id}_{entity_id}"
    client.upsert(
        collection_name="orya_match_index",
        points=[
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "user_id": user_id,
                    "entity_id": entity_id,
                    "fact_summary": fact_summary,
                    "category": category,
                },
            )
        ],
    )


def search_match_index(
    query_embedding: list[float],
    exclude_user_ids: list[str],
    top_k: int = 10,
    min_score: float = 0.5,
) -> list[dict[str, Any]]:
    """Search match index across all users (for cross-user matching).

    Returns list of {user_id, entity_id, fact_summary, score, category}
    """
    client = get_qdrant_client()
    results = client.search(
        collection_name="orya_match_index",
        query_vector=query_embedding,
        query_filter={
            "must_not": [
                {"key": "user_id", "match": {"value": uid}}
                for uid in exclude_user_ids
            ]
        },
        limit=top_k,
        score_threshold=min_score,
    )
    return [
        {
            "user_id": r.payload.get("user_id", ""),
            "entity_id": r.payload.get("entity_id", ""),
            "fact_summary": r.payload.get("fact_summary", ""),
            "score": r.score,
            "category": r.payload.get("category", "general"),
        }
        for r in results
    ]


def search_match_index_by_category(
    query_embedding: list[float],
    category: str,
    exclude_user_ids: list[str],
    top_k: int = 10,
    min_score: float = 0.5,
) -> list[dict[str, Any]]:
    """Search match index filtered by category (e.g., 'offering' vs 'seeking')."""
    client = get_qdrant_client()
    results = client.search(
        collection_name="orya_match_index",
        query_vector=query_embedding,
        query_filter={
            "must": [
                {"key": "category", "match": {"value": category}}
            ],
            "must_not": [
                {"key": "user_id", "match": {"value": uid}}
                for uid in exclude_user_ids
            ]
        },
        limit=top_k,
        score_threshold=min_score,
    )
    return [
        {
            "user_id": r.payload.get("user_id", ""),
            "entity_id": r.payload.get("entity_id", ""),
            "fact_summary": r.payload.get("fact_summary", ""),
            "score": r.score,
            "category": r.payload.get("category", "general"),
        }
        for r in results
    ]
