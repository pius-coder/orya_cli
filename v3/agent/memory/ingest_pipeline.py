"""Ingest pipeline orchestrating MemBrain-style memory creation.

ADAPTED from MemBrain's ingest_workflow.py.
Stages:
1. Extract entities (2 passes)
2. Generate facts
3. Resolve entities (3 layers: exact, MinHash LSH, LLM)
4. Generate embeddings
5. Persist to PostgreSQL
6. Update entity trees
7. Update match index (cross-user)
"""
import json
import logging
from typing import Any

from langchain_core.runnables import Runnable

from ..db.postgres import get_pool
from ..infra.qdrant import upsert_fact, upsert_match_index
from ..providers.embedder import HuggingFaceEmbedder
from .entity_resolver import ResolverDecision, resolve_entities_membrain
from .entity_tree import EntityTree, TreeNode, attach_all, batch_recompute_centroids, route_facts
from .extractor import extract_entities
from .fact_generator import generate_facts
from .models import MatchIndexEntry, NaturalFact

logger = logging.getLogger(__name__)


async def ingest_conversation(
    user_id: str,
    messages: list[dict[str, str]],
    llm: Runnable,
    embedder: HuggingFaceEmbedder | None = None,
    session_number: int = 0,
) -> dict[str, Any]:
    """Ingest a conversation batch into the user's PKG.

    Args:
        user_id: The user identifier.
        messages: List of {"speaker": "user|assistant", "content": "..."}
        llm: LLM for extraction and resolution.
        embedder: HuggingFace embedder for generating embeddings.
        session_number: Optional session identifier.

    Returns:
        Summary of what was ingested.
    """
    # Build conversation text
    user_texts = [m["content"] for m in messages if m.get("speaker") == "user"]
    all_text = "\n".join(m["content"] for m in messages)
    user_text = "\n".join(user_texts)

    # Stage 1: Load existing entities for this user
    existing_entities, existing_aliases = await _load_existing_entities(user_id)

    # Stage 2: Extract entities (2 passes)
    known = list(existing_entities.values()) if existing_entities else None
    entity_names = await extract_entities(user_text, llm, known_entities=known)
    if not entity_names:
        logger.info("No entities extracted for user=%s", user_id)
        return {"ingested": False, "reason": "no_entities"}

    # Stage 3: Generate facts
    facts = await generate_facts(all_text, entity_names, llm)
    if not facts:
        logger.info("No facts generated for user=%s", user_id)
        return {"ingested": False, "reason": "no_facts"}

    # Stage 4: Resolve entities (3 layers)
    resolver_decisions = await resolve_entities_membrain(
        entity_names, existing_entities, existing_aliases, llm=llm
    )

    # Stage 5: Generate embeddings for facts
    fact_embeddings = {}
    if embedder:
        texts_to_embed = [f.text for f in facts]
        try:
            # Batch embed
            embeddings = await embedder.create_batch(texts_to_embed)
            for fact, emb in zip(facts, embeddings):
                fact_embeddings[fact.text] = emb
        except Exception as e:
            logger.error("Embedding generation failed: %s", e)

    # Stage 6: Persist batch
    fact_ids = await _persist_batch(user_id, facts, resolver_decisions, fact_embeddings, session_number)

    # Stage 7: Update entity trees
    await _update_entity_trees(user_id, resolver_decisions, facts, fact_ids)

    # Stage 8: Update cross-user match index
    await _update_match_index(user_id, resolver_decisions, facts, fact_embeddings)

    logger.info(
        "Ingested user=%s entities=%d facts=%d",
        user_id,
        len(entity_names),
        len(facts),
    )
    return {
        "ingested": True,
        "entities": len(entity_names),
        "facts": len(facts),
        "fact_ids": fact_ids,
    }


async def _load_existing_entities(user_id: str) -> tuple[dict[str, dict], dict[str, list[str]]]:
    """Load existing entities and aliases for a user.

    Returns:
        existing_entities: {entity_id: {"canonical_ref": ..., "desc": ...}}
        existing_aliases: {entity_id: [alias1, alias2, ...]}
    """
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT entity_id, canonical_ref, description FROM orya.mb_entities WHERE user_id = $1",
            user_id,
        )
        entities = {
            r["entity_id"]: {"canonical_ref": r["canonical_ref"], "desc": r["description"] or ""}
            for r in rows
        }

        alias_rows = await conn.fetch(
            "SELECT DISTINCT entity_id, alias_text FROM orya.mb_fact_refs WHERE user_id = $1",
            user_id,
        )
        aliases: dict[str, list[str]] = {}
        for r in alias_rows:
            aliases.setdefault(r["entity_id"], []).append(r["alias_text"])

    return entities, aliases


async def _persist_batch(
    user_id: str,
    facts: list[NaturalFact],
    decisions: list[ResolverDecision],
    fact_embeddings: dict[str, list[float]],
    session_number: int,
) -> list[int]:
    """Persist entities, facts, and refs to PostgreSQL."""
    pool = get_pool()
    fact_ids: list[int] = []

    # Build entity_id mapping from resolver decisions
    entity_id_map: dict[str, str] = {}  # ref -> entity_id
    for d in decisions:
        eid = d.target_entity_id or d.new_entity_ref.replace(" ", "_").lower()[:64]
        entity_id_map[d.new_entity_ref] = eid

    async with pool.acquire() as conn:
        async with conn.transaction():
            # Persist new entities (for "keep" decisions)
            for d in decisions:
                if d.action == "keep":
                    eid = d.new_entity_ref.replace(" ", "_").lower()[:64]
                    existing = await conn.fetchval(
                        "SELECT id FROM orya.mb_entities WHERE user_id = $1 AND entity_id = $2",
                        user_id,
                        eid,
                    )
                    if not existing:
                        await conn.execute(
                            """INSERT INTO orya.mb_entities (user_id, entity_id, canonical_ref)
                               VALUES ($1, $2, $3)
                               ON CONFLICT DO NOTHING""",
                            user_id,
                            eid,
                            d.new_entity_ref,
                        )

            # Persist facts
            for fact in facts:
                search_text = fact.text
                for ent in fact.entities:
                    search_text = search_text.replace(f"[{ent}]", ent)
                search_text = search_text.replace("[", "").replace("]", "")

                fact_id = await conn.fetchval(
                    """INSERT INTO orya.mb_facts (user_id, text, search_text, session_number, fact_ts)
                       VALUES ($1, $2, $3, $4, $5) RETURNING id""",
                    user_id,
                    fact.text,
                    search_text,
                    session_number,
                    fact.time_resolved,
                )
                fact_ids.append(fact_id)

                # Store embedding in Qdrant (vector DB)
                embedding = fact_embeddings.get(fact.text)
                if embedding:
                    try:
                        eids = [entity_id_map.get(ent) for ent in fact.entities if entity_id_map.get(ent)]
                        category = _categorize_fact(fact.text)
                        upsert_fact(
                            fact_id=fact_id,
                            user_id=user_id,
                            text=fact.text,
                            embedding=embedding,
                            entity_ids=eids,
                            category=category,
                        )
                    except Exception as e:
                        logger.warning("Failed to upsert fact to Qdrant: %s", e)

                # Persist fact_refs
                for ent_ref in fact.entities:
                    eid = entity_id_map.get(ent_ref)
                    if eid:
                        await conn.execute(
                            """INSERT INTO orya.mb_fact_refs (fact_id, entity_id, alias_text, user_id)
                               VALUES ($1, $2, $3, $4)
                               ON CONFLICT DO NOTHING""",
                            fact_id,
                            eid,
                            ent_ref,
                            user_id,
                        )

                # Persist time annotations
                if fact.time_raw and fact.time_resolved:
                    await conn.execute(
                        """INSERT INTO orya.mb_time_annotations (fact_id, time_raw, time_resolved)
                           VALUES ($1, $2, $3)""",
                        fact_id,
                        fact.time_raw,
                        fact.time_resolved,
                    )

    return fact_ids


async def _update_entity_trees(
    user_id: str,
    decisions: list[ResolverDecision],
    facts: list[NaturalFact],
    fact_ids: list[int],
) -> None:
    """Update entity trees for touched entities."""
    pool = get_pool()
    touched = {d.target_entity_id or entity_id_map.get(d.new_entity_ref) for d in decisions}
    # Rebuild entity_id_map for touched entities
    entity_id_map = {}
    for d in decisions:
        eid = d.target_entity_id or d.new_entity_ref.replace(" ", "_").lower()[:64]
        entity_id_map[d.new_entity_ref] = eid
    touched = {entity_id_map.get(d.new_entity_ref) for d in decisions}
    touched.discard(None)

    for eid in touched:
        if not eid:
            continue
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT id, parent_id, node_type, fact_id, description
                   FROM orya.mb_entity_trees
                   WHERE user_id = $1 AND entity_id = $2
                   ORDER BY id""",
                user_id,
                eid,
            )

        if not rows:
            async with pool.acquire() as conn:
                root_id = await conn.fetchval(
                    """INSERT INTO orya.mb_entity_trees (user_id, entity_id, node_type, description)
                       VALUES ($1, $2, 'root', $3)
                       ON CONFLICT DO NOTHING
                       RETURNING id""",
                    user_id,
                    eid,
                    f"Entity: {eid}",
                )
                if not root_id:
                    root_id = await conn.fetchval(
                        "SELECT id FROM orya.mb_entity_trees WHERE user_id = $1 AND entity_id = $2 AND node_type = 'root'",
                        user_id,
                        eid,
                    )
        else:
            root_id = rows[0]["id"]

        for fid in fact_ids:
            async with pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO orya.mb_entity_trees (user_id, entity_id, parent_id, node_type, fact_id)
                       VALUES ($1, $2, $3, 'leaf', $4)
                       ON CONFLICT DO NOTHING""",
                    user_id,
                    eid,
                    root_id,
                    fid,
                )


async def _update_match_index(
    user_id: str,
    decisions: list[ResolverDecision],
    facts: list[NaturalFact],
    fact_embeddings: dict[str, list[float]],
) -> None:
    """Update the cross-user match index in PostgreSQL + Qdrant."""
    pool = get_pool()
    for fact in facts:
        for ent in fact.entities:
            category = _categorize_fact(fact.text)
            eid = ent.replace(" ", "_").lower()[:64]
            
            # PostgreSQL (relational metadata)
            async with pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO orya.mb_match_index (user_id, entity_id, canonical_ref, fact_summary, category)
                       VALUES ($1, $2, $3, $4, $5)
                       ON CONFLICT DO NOTHING""",
                    user_id,
                    eid,
                    ent,
                    fact.text[:500],
                    category,
                )
            
            # Qdrant (vector search)
            embedding = fact_embeddings.get(fact.text)
            if embedding:
                try:
                    upsert_match_index(
                        user_id=user_id,
                        entity_id=eid,
                        fact_summary=fact.text[:500],
                        embedding=embedding,
                        category=category,
                    )
                except Exception as e:
                    logger.warning("Failed to upsert match index to Qdrant: %s", e)


def _categorize_fact(text: str) -> str:
    """Simple keyword-based categorization for matching."""
    t = text.lower()
    if any(w in t for w in ["cherche", "looking for", "besoin", "need", "want"]):
        return "seeking"
    if any(w in t for w in ["suis", "am a", "work as", "plombier", "dev", "developer"]):
        return "offering"
    if any(w in t for w in ["habite", "live in", "lyon", "paris", "ville"]):
        return "location"
    return "general"
