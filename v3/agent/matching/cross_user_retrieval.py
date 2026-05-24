"""Cross-user matching engine.

ADAPTED from MemBrain's retrieval patterns.
Uses Qdrant for vector-based cross-user matching.
Implements sequential reveal (1 candidate at a time).
"""
import logging
from typing import Any

from ..db.postgres import get_pool
from ..infra.qdrant import search_match_index, search_match_index_by_category
from ..providers.embedder import HuggingFaceEmbedder

logger = logging.getLogger(__name__)

_RRF_K = 60


async def find_matches(
    seeker_id: str,
    query: str,
    embedder: HuggingFaceEmbedder | None = None,
    top_k: int = 5,
    excluded_user_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Find matching users across all PKGs using Qdrant vector search.

    Multi-path cross-user retrieval:
      Path 1: Qdrant vector search on match_index (cosine similarity)
      Path 2: Category complementarity via Qdrant filtered search
      Path 3: Entity overlap via PostgreSQL
    """
    pool = get_pool()
    excluded = excluded_user_ids or []
    excluded.append(seeker_id)

    candidates: dict[str, dict[str, Any]] = {}

    # Path 1: Qdrant vector search on match_index
    if embedder:
        try:
            query_vec = await embedder.create(query)
            qdrant_results = search_match_index(
                query_embedding=query_vec,
                exclude_user_ids=excluded,
                top_k=top_k * 3,
                min_score=0.3,
            )
            for r in qdrant_results:
                uid = r["user_id"]
                if uid not in candidates:
                    candidates[uid] = {
                        "user_id": uid,
                        "score": 0.0,
                        "matches": [],
                        "entities": set(),
                        "categories": set(),
                    }
                candidates[uid]["matches"].append(r["fact_summary"])
                candidates[uid]["entities"].add(r["entity_id"])
                candidates[uid]["categories"].add(r["category"])
                candidates[uid]["score"] += r["score"] * 2.0
        except Exception as e:
            logger.warning("Path 1 Qdrant vector search failed: %s", e)

    # Path 2: Category complementarity via Qdrant
    category_hint = _detect_category(query)
    if category_hint and embedder:
        opposite = "offering" if category_hint == "seeking" else "seeking"
        try:
            query_vec = await embedder.create(query)
            qdrant_results = search_match_index_by_category(
                query_embedding=query_vec,
                category=opposite,
                exclude_user_ids=excluded,
                top_k=top_k * 2,
                min_score=0.3,
            )
            for r in qdrant_results:
                uid = r["user_id"]
                if uid not in candidates:
                    candidates[uid] = {
                        "user_id": uid,
                        "score": 0.0,
                        "matches": [],
                        "entities": set(),
                        "categories": set(),
                    }
                candidates[uid]["matches"].append(r["fact_summary"])
                candidates[uid]["entities"].add(r["entity_id"])
                candidates[uid]["categories"].add(r["category"])
                candidates[uid]["score"] += r["score"] * 3.0  # Higher weight for complementarity
        except Exception as e:
            logger.warning("Path 2 category complementarity failed: %s", e)

    # Path 3: Entity overlap via PostgreSQL
    async with pool.acquire() as conn:
        seeker_entities = await conn.fetch(
            "SELECT DISTINCT entity_id FROM orya.mb_match_index WHERE user_id = $1",
            seeker_id,
        )
        seeker_eids = [r["entity_id"] for r in seeker_entities]
        if seeker_eids:
            overlap_rows = await conn.fetch(
                """SELECT user_id, entity_id, canonical_ref, fact_summary
                   FROM orya.mb_match_index
                   WHERE user_id != ALL($1) AND entity_id = ANY($2)
                   LIMIT $3""",
                excluded,
                seeker_eids,
                top_k * 3,
            )
            for r in overlap_rows:
                uid = r["user_id"]
                if uid not in candidates:
                    candidates[uid] = {
                        "user_id": uid,
                        "score": 0.0,
                        "matches": [],
                        "entities": set(),
                        "categories": set(),
                    }
                candidates[uid]["matches"].append(r["fact_summary"])
                candidates[uid]["entities"].add(r["canonical_ref"])
                candidates[uid]["score"] += 1.5

    # Rank by score
    ranked = sorted(candidates.values(), key=lambda x: x["score"], reverse=True)

    # Build result
    result = []
    for c in ranked[:top_k]:
        uid = c["user_id"]
        async with pool.acquire() as conn:
            alias = await conn.fetchval(
                "SELECT alias FROM orya.users WHERE id = $1", uid
            )

        unique_matches = list(dict.fromkeys(c["matches"]))[:3]
        summary = " ; ".join(unique_matches) if unique_matches else "Profil compatible"

        result.append({
            "user_id": uid,
            "alias": alias or uid[:8],
            "summary": summary,
            "score": c["score"],
            "entities": list(c["entities"])[:5],
            "candidate_uuid": uid,
        })

    return result


def _detect_category(text: str) -> str | None:
    t = text.lower()
    if any(w in t for w in ["cherche", "looking for", "besoin", "need", "veux", "want", " recherche "]):
        return "seeking"
    if any(w in t for w in ["suis", "am a", "work as", "fais", "do", "offre", "offer"]):
        return "offering"
    return None


async def get_sequential_candidate(seeker_id: str, query: str, embedder: HuggingFaceEmbedder | None = None) -> dict[str, Any] | None:
    """Get the top 1 candidate for sequential reveal.

    Returns None if no match or if all candidates already proposed.
    """
    from ..db import list_pending_opt_ins

    pending = await list_pending_opt_ins(seeker_id)
    excluded = [p["provider_id"] for p in pending if p.get("provider_id")]

    candidates = await find_matches(seeker_id, query, embedder=embedder, top_k=1, excluded_user_ids=excluded)
    return candidates[0] if candidates else None
