"""Entity resolution with three-layer deduplication.

ADAPTED from MemBrain's entity_resolver.py.
Uses MinHash LSH + Jaccard for fuzzy matching, LLM fallback for ambiguities.
"""
from __future__ import annotations

import json
import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from hashlib import blake2b
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

logger = logging.getLogger(__name__)

# ── Config (from MemBrain settings) ──────────────────────────────
RESOLVER_MINHASH_PERMUTATIONS = 32
RESOLVER_MINHASH_BAND_SIZE = 4
RESOLVER_JACCARD_THRESHOLD = 0.9
RESOLVER_MIN_NAME_LENGTH = 6
RESOLVER_MIN_TOKEN_COUNT = 2
RESOLVER_ENTROPY_THRESHOLD = 1.5


# ── Normalization ────────────────────────────────────────────────


def _normalize(ref: str) -> str:
    """Lowercase and collapse whitespace."""
    return re.sub(r"\s+", " ", ref.lower()).strip()


def _normalize_fuzzy(ref: str) -> str:
    """Produce alphanumeric-only form for n-gram shingles."""
    cleaned = re.sub(r"[^a-z0-9' ]", " ", _normalize(ref))
    return re.sub(r"\s+", " ", cleaned).strip()


# ── Shingles + MinHash ───────────────────────────────────────────


def _shingles(fuzzy: str) -> set[str]:
    """3-gram shingles from space-stripped fuzzy form."""
    s = fuzzy.replace(" ", "")
    if not s:
        return set()
    if len(s) < 3:
        return {s}
    return {s[i : i + 3] for i in range(len(s) - 2)}


def _hash_shingle(shingle: str, seed: int) -> int:
    digest = blake2b(f"{seed}:{shingle}".encode(), digest_size=8)
    return int.from_bytes(digest.digest(), "big")


def _minhash_signature(shingles: set[str]) -> tuple[int, ...]:
    if not shingles:
        return tuple()
    n = RESOLVER_MINHASH_PERMUTATIONS
    return tuple(min(_hash_shingle(sh, seed) for sh in shingles) for seed in range(n))


def _lsh_bands(sig: tuple[int, ...]) -> list[tuple[int, ...]]:
    band_size = RESOLVER_MINHASH_BAND_SIZE
    sig_list = list(sig)
    bands = []
    for start in range(0, len(sig_list), band_size):
        band = tuple(sig_list[start : start + band_size])
        if len(band) == band_size:
            bands.append(band)
    return bands


# ── Entropy filter ───────────────────────────────────────────────


def _has_high_entropy(fuzzy: str) -> bool:
    token_count = len(fuzzy.split())
    if len(fuzzy) < RESOLVER_MIN_NAME_LENGTH and token_count < RESOLVER_MIN_TOKEN_COUNT:
        return False
    stripped = fuzzy.replace(" ", "")
    if not stripped:
        return False
    counts: dict[str, int] = {}
    for ch in stripped:
        counts[ch] = counts.get(ch, 0) + 1
    total = sum(counts.values())
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
    return entropy >= RESOLVER_ENTROPY_THRESHOLD


# ── Jaccard ──────────────────────────────────────────────────────


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ── Data structures ──────────────────────────────────────────────


@dataclass
class CandidateEntry:
    entity_id: str
    name: str
    shingles: set[str]


@dataclass
class ResolverIndexes:
    entries: list[CandidateEntry]
    by_entity_id: dict[str, Any]  # entity_id -> {canonical_ref, desc}
    normalized_map: defaultdict[str, list[CandidateEntry]]
    lsh_buckets: defaultdict[tuple, list[str]]  # (band_idx, band) -> [entity_id]
    aliases_by_entity: dict[str, list[str]]


@dataclass
class ResolverDecision:
    new_entity_ref: str
    action: Literal["keep", "merge"]
    target_entity_id: str | None = None
    resolved_via: Literal["exact", "minhash", "llm"] | None = None


# Sentinel: Layer 1 found multiple entities for the same normalized name
LAYER1_AMBIGUOUS = "AMBIGUOUS"


def build_resolver_indexes(
    entries: list[CandidateEntry],
    by_entity_id: dict[str, Any],
    aliases_by_entity: dict[str, list[str]],
) -> ResolverIndexes:
    normalized_map: defaultdict[str, list[CandidateEntry]] = defaultdict(list)
    lsh_buckets: defaultdict[tuple, list[str]] = defaultdict(list)

    for entry in entries:
        norm = _normalize(entry.name)
        normalized_map[norm].append(entry)

        sig = _minhash_signature(entry.shingles)
        for band_idx, band in enumerate(_lsh_bands(sig)):
            lsh_buckets[(band_idx, band)].append(entry.entity_id)

    return ResolverIndexes(
        entries=entries,
        by_entity_id=by_entity_id,
        normalized_map=normalized_map,
        lsh_buckets=lsh_buckets,
        aliases_by_entity=aliases_by_entity,
    )


# ── Layer 1: exact normalized match ──────────────────────────────


def layer1_exact(
    new_ref: str, indexes: ResolverIndexes
) -> ResolverDecision | str | None:
    """Return ResolverDecision on match, LAYER1_AMBIGUOUS on multi-entity hit, None on miss."""
    norm = _normalize(new_ref)
    matches = indexes.normalized_map.get(norm, [])
    if not matches:
        return None
    unique_eids = {e.entity_id for e in matches}
    if len(unique_eids) > 1:
        return LAYER1_AMBIGUOUS
    return ResolverDecision(
        new_entity_ref=new_ref,
        action="merge",
        target_entity_id=next(iter(unique_eids)),
        resolved_via="exact",
    )


# ── Layer 2: MinHash LSH + Jaccard ───────────────────────────────


def layer2_minhash(new_ref: str, indexes: ResolverIndexes) -> ResolverDecision | None:
    fuzzy = _normalize_fuzzy(new_ref)
    if not _has_high_entropy(fuzzy):
        return None

    new_shingles = _shingles(fuzzy)
    sig = _minhash_signature(new_shingles)

    candidate_eids: set[str] = set()
    for band_idx, band in enumerate(_lsh_bands(sig)):
        candidate_eids.update(indexes.lsh_buckets.get((band_idx, band), []))

    best_eid: str | None = None
    best_score = 0.0
    for entry in indexes.entries:
        if entry.entity_id not in candidate_eids:
            continue
        score = _jaccard(new_shingles, entry.shingles)
        if score > best_score:
            best_score = score
            best_eid = entry.entity_id

    if best_eid is not None and best_score >= RESOLVER_JACCARD_THRESHOLD:
        return ResolverDecision(
            new_entity_ref=new_ref,
            action="merge",
            target_entity_id=best_eid,
            resolved_via="minhash",
        )
    return None


# ── Layer 3: LLM fallback ────────────────────────────────────────


async def layer3_llm(
    unresolved_refs: list[str],
    unresolved_descs: dict[str, str],
    indexes: ResolverIndexes,
    llm: Runnable,
) -> list[ResolverDecision]:
    """Send unresolved new entities + deduplicated candidates to LLM."""
    if not unresolved_refs:
        return []

    # Dedup candidates by entity_id for LLM context
    seen_eids: set[str] = set()
    deduped_candidates: list[dict] = []
    eid_by_candidate_id: dict[int, str] = {}

    for entry in indexes.entries:
        if entry.entity_id in seen_eids:
            continue
        seen_eids.add(entry.entity_id)
        cid = len(deduped_candidates)
        ent_model = indexes.by_entity_id.get(entry.entity_id)
        deduped_candidates.append(
            {
                "id": cid,
                "canonical_ref": ent_model.get("canonical_ref", entry.name) if ent_model else entry.name,
                "aliases": indexes.aliases_by_entity.get(entry.entity_id, []),
                "desc": ent_model.get("desc", "") if ent_model else "",
            }
        )
        eid_by_candidate_id[cid] = entry.entity_id

    new_entities_ctx = [
        {"id": i, "ref": ref, "desc": unresolved_descs.get(ref, "")}
        for i, ref in enumerate(unresolved_refs)
    ]

    new_json = json.dumps(new_entities_ctx, ensure_ascii=False)
    existing_json = json.dumps(deduped_candidates, ensure_ascii=False)

    prompt = [
        SystemMessage(
            content="You are an entity resolver. Decide if each new entity matches any existing entity."
            " Output ONLY valid JSON: {\"resolutions\": [{\"new_entity_id\": 0, \"matched_entity_id\": -1}, ...]}"
            " Use matched_entity_id=-1 to keep as new entity."
        ),
        HumanMessage(
            content=f"New entities:\n{new_json}\n\nExisting entities:\n{existing_json}"
        ),
    ]

    try:
        resp = await llm.ainvoke(prompt)
        data = _extract_json(str(getattr(resp, "content", "")))
    except Exception:
        logger.warning("entity-resolver LLM failed, keeping all as create")
        return [ResolverDecision(new_entity_ref=r, action="keep") for r in unresolved_refs]

    new_id_to_ref = {i: ref for i, ref in enumerate(unresolved_refs)}
    decisions: list[ResolverDecision] = []
    resolved_new_ids: set[int] = set()

    resolutions = data.get("resolutions", [])
    if not isinstance(resolutions, list):
        resolutions = []

    for res in resolutions:
        nid = res.get("new_entity_id")
        mid = res.get("matched_entity_id")
        if nid not in new_id_to_ref:
            continue
        resolved_new_ids.add(nid)
        if mid == -1:
            decisions.append(ResolverDecision(new_entity_ref=new_id_to_ref[nid], action="keep"))
            continue
        if mid not in eid_by_candidate_id:
            decisions.append(ResolverDecision(new_entity_ref=new_id_to_ref[nid], action="keep"))
            continue
        target_eid = eid_by_candidate_id[mid]
        decisions.append(
            ResolverDecision(
                new_entity_ref=new_id_to_ref[nid],
                action="merge",
                target_entity_id=target_eid,
                resolved_via="llm",
            )
        )

    # Fill missing (LLM didn't return resolution)
    for nid, ref in new_id_to_ref.items():
        if nid not in resolved_new_ids:
            decisions.append(ResolverDecision(new_entity_ref=ref, action="keep"))

    return decisions


# ── Top-level entry point ────────────────────────────────────────


async def resolve_entities_membrain(
    new_entities: list[str],
    existing_entities: dict[str, dict],  # entity_id -> {canonical_ref, desc}
    existing_aliases: dict[str, list[str]],
    llm: Runnable | None = None,
) -> list[ResolverDecision]:
    """Run three-layer resolution on new entities.

    Returns list of ResolverDecision (keep or merge with target).
    """
    if not new_entities:
        return []

    # Build candidate entries
    entries: list[CandidateEntry] = []
    for eid, ent in existing_entities.items():
        ref = ent.get("canonical_ref", eid)
        fuzzy = _normalize_fuzzy(ref)
        entries.append(CandidateEntry(
            entity_id=eid,
            name=ref,
            shingles=_shingles(fuzzy),
        ))
        # Also index aliases
        for alias in existing_aliases.get(eid, []):
            fuzzy_alias = _normalize_fuzzy(alias)
            entries.append(CandidateEntry(
                entity_id=eid,
                name=alias,
                shingles=_shingles(fuzzy_alias),
            ))

    if not entries:
        return [ResolverDecision(new_entity_ref=r, action="keep") for r in new_entities]

    indexes = build_resolver_indexes(entries, existing_entities, existing_aliases)

    # Layer 1 + 2 pass
    unresolved_refs: list[str] = []
    resolver_map: dict[str, ResolverDecision] = {}

    for ref in new_entities:
        dec = layer1_exact(ref, indexes)
        if dec is LAYER1_AMBIGUOUS:
            unresolved_refs.append(ref)
        elif dec is not None:
            resolver_map[ref] = dec
        else:
            dec = layer2_minhash(ref, indexes)
            if dec is not None:
                resolver_map[ref] = dec
            else:
                unresolved_refs.append(ref)

    # Layer 3
    if unresolved_refs and llm:
        desc_map = {ref: "" for ref in unresolved_refs}
        llm_decisions = await layer3_llm(unresolved_refs, desc_map, indexes, llm)
        for dec in llm_decisions:
            resolver_map[dec.new_entity_ref] = dec
    elif unresolved_refs:
        for ref in unresolved_refs:
            resolver_map[ref] = ResolverDecision(new_entity_ref=ref, action="keep")

    return [resolver_map[r] for r in new_entities if r in resolver_map]


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
