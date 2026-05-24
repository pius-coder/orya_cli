"""Multi-path retrieval from Personal Knowledge Graph.

ADAPTED from MemBrain's retrieval/application/retrieval.py.
Paths:
  A  — PostgreSQL full-text search (tsvector) on facts
  B  — Cosine similarity on fact embeddings
  B2 — Cosine similarity on HyDE query embedding
  B3 — Cosine similarity on event-focused query embedding
  C  — Entity tree traversal (simplified)
  D  — Keyword-conjunctive search on facts

Fusion: RRF (Reciprocal Rank Fusion) or direct scoring.
"""
import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from ..db.postgres import get_pool
from ..infra.qdrant import search_facts
from ..providers.embedder import HuggingFaceEmbedder

logger = logging.getLogger(__name__)

_RRF_K = 60

# Multi-query prompts (from MemBrain)
_MULTI_QUERY_SYSTEM = """Generate EXACTLY 3 complementary search queries for memory retrieval.
Query 1 — Event-focused (removes temporal constraints, focuses on the event)
Query 2 — HyDE declarative (a hypothetical passage that would exist in memory)
Query 3 — BM25 keyword strip (core entities and verbs only, no filler words)
Output ONLY valid JSON: {"queries": ["...", "...", "..."]}"""

_REWRITE_SYSTEM = """Rewrite the user's question for memory retrieval.
Extract 3-6 key terms that would match stored facts.
Output ONLY the rewritten query, no explanation."""


async def retrieve_from_pkg(
    user_id: str,
    question: str,
    llm: Runnable | None = None,
    embedder: HuggingFaceEmbedder | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    """Multi-path retrieval from user's PKG.

    Returns dict with "facts", "entities", "summaries", "packed_context".
    """
    pool = get_pool()

    # ── 1. Query expansion ────────────────────────────────────────
    rewritten = question
    q_event = q_hyde = ""
    if llm:
        try:
            rewritten = await _rewrite_query(question, llm)
            extra = await _generate_multi_queries(question, llm)
            q_event = extra[0] if len(extra) > 0 else ""
            q_hyde = extra[1] if len(extra) > 1 else ""
        except Exception as e:
            logger.warning("Query expansion failed: %s", e)

    # ── 2. Embeddings ─────────────────────────────────────────────
    orig_vec = await _embed_question(question, embedder)
    hyde_vec = await _embed_question(q_hyde, embedder) if q_hyde else orig_vec
    event_vec = await _embed_question(q_event, embedder) if q_event else orig_vec

    # ── 3. Six retrieval paths ────────────────────────────────────
    path_a = await _path_a_tsvector(user_id, rewritten, pool, top_k * 2)
    path_b = await _path_b_embedding(user_id, orig_vec, pool, top_k * 2)
    path_b2 = await _path_b_embedding(user_id, hyde_vec, pool, top_k * 2) if hyde_vec is not orig_vec else []
    path_b3 = await _path_b_embedding(user_id, event_vec, pool, top_k * 2) if event_vec is not orig_vec else []
    path_c = await _path_c_entity_tree(user_id, question, orig_vec, pool, top_k)
    path_d = await _path_d_keyword_conjunctive(user_id, rewritten, pool, top_k)

    # ── 4. Dedup into pool ────────────────────────────────────────
    ranked_lists = [
        [f["id"] for f in path_a],
        [f["id"] for f in path_b],
        [f["id"] for f in path_b2],
        [f["id"] for f in path_b3],
        [f["id"] for f in path_c],
        [f["id"] for f in path_d],
    ]
    seen: dict[int, dict] = {}
    for lst in (path_a, path_b, path_b2, path_b3, path_c, path_d):
        for f in lst:
            fid = f["id"]
            if fid not in seen:
                seen[fid] = f
    pool_facts = list(seen.values())

    # ── 5. RRF Fusion ─────────────────────────────────────────────
    _fuse_rrf(pool_facts, ranked_lists)
    pool_facts.sort(key=lambda f: f.get("rrf_score", 0.0), reverse=True)
    top_facts = pool_facts[:top_k]

    # ── 6. Load entities for context ──────────────────────────────
    entities = []
    summaries = []
    if top_facts:
        fact_ids = [f["id"] for f in top_facts]
        async with pool.acquire() as conn:
            ent_rows = await conn.fetch(
                """SELECT DISTINCT fr.entity_id, e.canonical_ref
                   FROM orya.mb_fact_refs fr
                   JOIN orya.mb_entities e ON fr.entity_id = e.entity_id AND fr.user_id = e.user_id
                   WHERE fr.fact_id = ANY($1) AND fr.user_id = $2""",
                fact_ids,
                user_id,
            )
            entities = [dict(r) for r in ent_rows]

            sum_rows = await conn.fetch(
                """SELECT session_number, subject, content
                   FROM orya.mb_session_summaries
                   WHERE user_id = $1
                   ORDER BY session_number DESC
                   LIMIT 3""",
                user_id,
            )
            summaries = [dict(r) for r in sum_rows]

    # ── 7. Pack context ───────────────────────────────────────────
    context_lines = []
    for f in top_facts:
        line = f"- {f['text']}"
        if f.get("time_info"):
            line += f" ({f['time_info']})"
        context_lines.append(line)
    packed = "\n".join(context_lines)

    return {
        "facts": top_facts,
        "entities": entities,
        "summaries": summaries,
        "packed_context": packed,
    }


# ── Path A: PostgreSQL full-text search ──────────────────────────


async def _path_a_tsvector(
    user_id: str, query: str, pool, limit: int
) -> list[dict]:
    """Path A: Full-text search using PostgreSQL tsvector on search_text."""
    try:
        async with pool.acquire() as conn:
            # Use plainto_tsquery for natural language query
            rows = await conn.fetch(
                """SELECT id, text, ts_rank(to_tsvector('french', search_text), plainto_tsquery('french', $1)) AS score
                   FROM orya.mb_facts
                   WHERE user_id = $2 AND status = 'active'
                     AND to_tsvector('french', search_text) @@ plainto_tsquery('french', $1)
                   ORDER BY score DESC
                   LIMIT $3""",
                query,
                user_id,
                limit,
            )
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning("Path A tsvector failed: %s", e)
        return []


# ── Path B: Cosine similarity on embeddings ──────────────────────


async def _path_b_embedding(
    user_id: str, query_vec: list[float] | None, pool, limit: int
) -> list[dict]:
    """Path B: Cosine similarity via Qdrant vector search."""
    if not query_vec:
        return []
    try:
        results = search_facts(
            user_id=user_id,
            query_embedding=query_vec,
            top_k=limit,
            min_score=0.3,
        )
        return [
            {
                "id": r["id"],
                "text": r["text"],
                "score": r["score"],
                "source": "embed",
            }
            for r in results
        ]
    except Exception as e:
        logger.warning("Path B Qdrant embedding failed: %s", e)
        return []


# ── Path C: Entity tree traversal ────────────────────────────────


async def _path_c_entity_tree(
    user_id: str, query: str, query_vec: list[float] | None, pool, limit: int
) -> list[dict]:
    """Path C: Find matching entities, then traverse tree to collect facts."""
    try:
        async with pool.acquire() as conn:
            # Match entities by text similarity on canonical_ref
            ent_rows = await conn.fetch(
                """SELECT entity_id FROM orya.mb_entities
                   WHERE user_id = $1
                     AND (canonical_ref ILIKE $2 OR description ILIKE $2)
                   LIMIT 10""",
                user_id,
                f"%{query}%",
            )
            if not ent_rows:
                return []

            # Collect fact_ids from entity tree leaves
            all_fact_ids = []
            for er in ent_rows:
                eid = er["entity_id"]
                tree_rows = await conn.fetch(
                    """SELECT fact_id FROM orya.mb_entity_trees
                       WHERE user_id = $1 AND entity_id = $2 AND node_type = 'leaf'
                         AND fact_id IS NOT NULL""",
                    user_id,
                    eid,
                )
                all_fact_ids.extend([tr["fact_id"] for tr in tree_rows])

            if not all_fact_ids:
                return []

            # Fetch facts, rank by embedding similarity if available
            unique_ids = list(dict.fromkeys(all_fact_ids))[:limit * 2]
            fact_rows = await conn.fetch(
                """SELECT id, text, embedding FROM orya.mb_facts
                   WHERE id = ANY($1) AND status = 'active'""",
                unique_ids,
            )

            if query_vec:
                scored = []
                for r in fact_rows:
                    emb = r["embedding"]
                    if isinstance(emb, str):
                        import json
                        emb = json.loads(emb)
                    sim = _cosine_sim(query_vec, emb) if emb else 0.0
                    scored.append((sim, dict(r)))
                scored.sort(key=lambda x: x[0], reverse=True)
                return [f for _, f in scored[:limit]]
            return [dict(r) for r in fact_rows[:limit]]
    except Exception as e:
        logger.warning("Path C entity tree failed: %s", e)
        return []


# ── Path D: Keyword conjunctive search ───────────────────────────


async def _path_d_keyword_conjunctive(
    user_id: str, query: str, pool, limit: int
) -> list[dict]:
    """Path D: AND-based keyword search on search_text.

    Breaks query into words and requires ALL words to match.
    """
    words = [w for w in query.split() if len(w) > 2]
    if not words:
        return []
    try:
        # Build AND condition with ILIKE
        conditions = " AND ".join([f"search_text ILIKE ${i+3}" for i in range(len(words))])
        params = [user_id, limit] + [f"%{w}%" for w in words]
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"""SELECT id, text FROM orya.mb_facts
                    WHERE user_id = $1 AND status = 'active' AND {conditions}
                    ORDER BY created_at DESC
                    LIMIT $2""",
                *params,
            )
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning("Path D keyword search failed: %s", e)
        return []


# ── Helpers ──────────────────────────────────────────────────────


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _fuse_rrf(pool: list[dict], ranked_lists: list[list[int]]) -> None:
    """Reciprocal Rank Fusion — assigns rrf_score to each fact in pool."""
    rank_maps = []
    for lst in ranked_lists:
        rank_maps.append({fid: i for i, fid in enumerate(lst)})
    for fact in pool:
        score = 0.0
        for rm in rank_maps:
            rank = rm.get(fact["id"])
            if rank is not None:
                score += 1.0 / (_RRF_K + rank + 1)
        fact["rrf_score"] = score


async def _embed_question(text: str, embedder: HuggingFaceEmbedder | None) -> list[float] | None:
    if not text or not embedder:
        return None
    try:
        return await embedder.create(text)
    except Exception as e:
        logger.warning("Embedding failed: %s", e)
        return None


async def _rewrite_query(question: str, llm: Runnable) -> str:
    prompt = [
        SystemMessage(content=_REWRITE_SYSTEM),
        HumanMessage(content=question),
    ]
    resp = await llm.ainvoke(prompt)
    return str(getattr(resp, "content", "")).strip() or question


async def _generate_multi_queries(question: str, llm: Runnable) -> list[str]:
    prompt = [
        SystemMessage(content=_MULTI_QUERY_SYSTEM),
        HumanMessage(content=question),
    ]
    resp = await llm.ainvoke(prompt)
    data = _extract_json(str(getattr(resp, "content", "")))
    return data.get("queries", [])


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```json"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except Exception:
        return {}
