"""Asyncpg pool + helpers for the `orya` business schema.

Note: LangGraph's PostgresSaver creates its own tables in the `public` schema
and is unrelated to this module.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import asyncpg

from ..settings import get_settings

logger = logging.getLogger(__name__)

_pool: Optional[asyncpg.Pool] = None


async def init_pool() -> asyncpg.Pool:
    """Create the global asyncpg pool. Idempotent."""

    global _pool
    if _pool is not None:
        return _pool

    s = get_settings()
    _pool = await asyncpg.create_pool(
        dsn=s.postgres_dsn,
        min_size=1,
        max_size=10,
        command_timeout=15,
    )
    # Verify schema exists (init.sql should have run already in container).
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT 1 FROM information_schema.schemata WHERE schema_name = 'orya'"
        )
        if row is None:
            logger.warning(
                "Schema 'orya' not found — make sure init.sql was applied."
            )
    logger.info("PG pool ready (db=%s, host=%s).", s.POSTGRES_DB, s.POSTGRES_HOST)
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("Postgres pool not initialized — call init_pool() first.")
    return _pool


# ============================================================
# Users
# ============================================================


async def upsert_user(user_id: str, alias: Optional[str] = None) -> None:
    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO orya.users (user_id, alias, last_seen_at)
            VALUES ($1, $2, now())
            ON CONFLICT (user_id) DO UPDATE
                SET alias = COALESCE(EXCLUDED.alias, orya.users.alias),
                    last_seen_at = now()
            """,
            user_id,
            alias,
        )


async def set_user_tutoyer(user_id: str, tutoyer: bool) -> None:
    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE orya.users SET tutoyer = $2 WHERE user_id = $1",
            user_id,
            tutoyer,
        )


async def get_user(user_id: str) -> Optional[dict[str, Any]]:
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT user_id, alias, tutoyer, created_at, last_seen_at "
            "FROM orya.users WHERE user_id = $1",
            user_id,
        )
        return dict(row) if row else None


# ============================================================
# Feedback
# ============================================================


async def record_feedback(
    *,
    user_id: str,
    user_text: str,
    assistant_reply: str,
    rating: int,
) -> None:
    if rating not in (-1, 1):
        raise ValueError("rating must be -1 or 1")
    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO orya.feedback (user_id, user_text, assistant_reply, rating)
            VALUES ($1, $2, $3, $4)
            """,
            user_id,
            user_text,
            assistant_reply,
            rating,
        )


async def get_good_examples(
    *, exclude_user_id: Optional[str] = None, limit: int = 5
) -> list[dict[str, Any]]:
    """Return up to `limit` recent positive feedback rows, optionally excluding
    one user (so we don't show their own examples back to themselves)."""

    pool = get_pool()
    async with pool.acquire() as conn:
        if exclude_user_id is not None:
            rows = await conn.fetch(
                """
                SELECT user_text, assistant_reply
                FROM orya.feedback
                WHERE rating = 1 AND user_id <> $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                exclude_user_id,
                limit,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT user_text, assistant_reply
                FROM orya.feedback
                WHERE rating = 1
                ORDER BY created_at DESC
                LIMIT $1
                """,
                limit,
            )
        return [dict(r) for r in rows]


# ============================================================
# Opt-ins (double opt-in state machine)
# ============================================================


async def create_opt_in(
    *,
    seeker_id: str,
    provider_id: str,
    need_summary: str,
    candidate_uuid: str,
) -> Optional[dict[str, Any]]:
    """Insert a fresh opt_in in pending_seeker. Returns the row, or None if a
    same-triplet row already exists (UNIQUE constraint hit)."""

    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO orya.opt_ins (
                seeker_id, provider_id, need_summary, candidate_uuid
            )
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (seeker_id, provider_id, candidate_uuid)
            DO NOTHING
            RETURNING opt_in_id, seeker_id, provider_id, need_summary,
                      candidate_uuid, status, created_at, expires_at
            """,
            seeker_id,
            provider_id,
            need_summary,
            candidate_uuid,
        )
        return dict(row) if row else None


async def get_opt_in(opt_in_id: str) -> Optional[dict[str, Any]]:
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT opt_in_id, seeker_id, provider_id, need_summary,
                   candidate_uuid, status, seeker_decision_at,
                   provider_decision_at, matched_at, expires_at, created_at
            FROM orya.opt_ins
            WHERE opt_in_id = $1
            """,
            opt_in_id,
        )
        return dict(row) if row else None


async def respond_seeker(
    opt_in_id: str, decision: str
) -> Optional[dict[str, Any]]:
    """Apply the seeker's decision. Returns the updated row."""

    if decision not in ("accept", "reject"):
        raise ValueError("decision must be 'accept' or 'reject'")
    new_status = "pending_provider" if decision == "accept" else "rejected_seeker"

    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            UPDATE orya.opt_ins
            SET status = $2,
                seeker_decision_at = now()
            WHERE opt_in_id = $1
              AND status = 'pending_seeker'
            RETURNING opt_in_id, seeker_id, provider_id, need_summary,
                      candidate_uuid, status
            """,
            opt_in_id,
            new_status,
        )
        return dict(row) if row else None


async def respond_provider(
    opt_in_id: str, decision: str
) -> Optional[dict[str, Any]]:
    if decision not in ("accept", "reject"):
        raise ValueError("decision must be 'accept' or 'reject'")
    new_status = "matched" if decision == "accept" else "rejected_provider"

    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            UPDATE orya.opt_ins
            SET status = $2,
                provider_decision_at = now(),
                matched_at = CASE WHEN $2 = 'matched' THEN now() ELSE matched_at END
            WHERE opt_in_id = $1
              AND status = 'pending_provider'
            RETURNING opt_in_id, seeker_id, provider_id, need_summary,
                      candidate_uuid, status
            """,
            opt_in_id,
            new_status,
        )
        return dict(row) if row else None


async def list_pending_opt_ins(user_id: str) -> list[dict[str, Any]]:
    """Return active opt_ins where the user is either seeker or provider and
    their decision is required."""

    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT opt_in_id, seeker_id, provider_id, need_summary,
                   candidate_uuid, status, expires_at
            FROM orya.opt_ins
            WHERE expires_at > now()
              AND (
                  (seeker_id = $1 AND status = 'pending_seeker')
                  OR (provider_id = $1 AND status = 'pending_provider')
              )
            ORDER BY created_at DESC
            """,
            user_id,
        )
        return [dict(r) for r in rows]
