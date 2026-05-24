"""Async PostgreSQL access layer using asyncpg.

Consolidates all SQL operations. Fixes v2 issues:
- Explicit transactions where needed
- Reflections can be cleared (removed COALESCE bug)
- Consistent error handling
"""
import logging
from typing import Any, Optional

import asyncpg

from ..core.config import get_settings

logger = logging.getLogger(__name__)

_pool: Optional[asyncpg.Pool] = None


async def init_pool() -> asyncpg.Pool:
    """Create and return the global asyncpg pool."""
    global _pool
    settings = get_settings()
    _pool = await asyncpg.create_pool(
        dsn=settings.postgres_dsn,
        min_size=2,
        max_size=10,
    )
    logger.info("PostgreSQL pool initialized")
    return _pool


async def close_pool() -> None:
    """Gracefully close the global pool."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        logger.info("PostgreSQL pool closed")


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("PostgreSQL pool not initialized. Call init_pool() first.")
    return _pool


# ── Users ─────────────────────────────────────────────────────────
async def upsert_user(user_id: str, alias: str) -> dict[str, Any]:
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO orya.users (id, alias)
            VALUES ($1, $2)
            ON CONFLICT (id) DO UPDATE
                SET alias = COALESCE(EXCLUDED.alias, orya.users.alias)
            RETURNING id, alias, tutoyer
            """,
            user_id,
            alias,
        )
    return dict(row) if row else {}


async def get_user(user_id: str) -> Optional[dict[str, Any]]:
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, alias, tutoyer FROM orya.users WHERE id = $1",
            user_id,
        )
    return dict(row) if row else None


async def set_user_tutoyer(user_id: str, tutoyer: bool) -> None:
    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE orya.users SET tutoyer = $1 WHERE id = $2",
            tutoyer,
            user_id,
        )


# ── Feedback ──────────────────────────────────────────────────────
async def record_feedback(
    user_id: str, user_input: str, orya_response: str, rating: str
) -> None:
    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO orya.feedback (user_id, user_input, orya_response, rating)
            VALUES ($1, $2, $3, $4)
            """,
            user_id,
            user_input,
            orya_response,
            rating,
        )


async def get_good_examples(user_id: str, limit: int = 3) -> list[dict[str, str]]:
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT user_input, orya_response
            FROM orya.feedback
            WHERE user_id = $1 AND rating = 'good'
            ORDER BY created_at DESC
            LIMIT $2
            """,
            user_id,
            limit,
        )
    return [{"user_input": r["user_input"], "orya_response": r["orya_response"]} for r in rows]


# ── Opt-Ins ───────────────────────────────────────────────────────
async def create_opt_in(
    seeker_id: str,
    provider_id: str,
    reason: str,
    candidate_uuid: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    pool = get_pool()
    async with pool.acquire() as conn:
        try:
            row = await conn.fetchrow(
                """
                INSERT INTO orya.opt_ins (seeker_id, provider_id, reason, candidate_uuid, status)
                VALUES ($1, $2, $3, $4, 'pending_seeker')
                ON CONFLICT (seeker_id, provider_id, candidate_uuid) DO NOTHING
                RETURNING *
                """,
                seeker_id,
                provider_id,
                reason,
                candidate_uuid,
            )
            return dict(row) if row else None
        except asyncpg.ForeignKeyViolationError:
            logger.warning("Opt-in FK violation: seeker=%s provider=%s", seeker_id, provider_id)
            return None


async def get_opt_in(opt_in_id: str) -> Optional[dict[str, Any]]:
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM orya.opt_ins WHERE id = $1",
            opt_in_id,
        )
    return dict(row) if row else None


async def respond_seeker(opt_in_id: str, accepted: bool) -> Optional[dict[str, Any]]:
    pool = get_pool()
    status = "pending_provider" if accepted else "declined_seeker"
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            UPDATE orya.opt_ins
            SET status = $1, seeker_accepted = $2
            WHERE id = $3 AND status = 'pending_seeker'
            RETURNING *
            """,
            status,
            accepted,
            opt_in_id,
        )
    return dict(row) if row else None


async def respond_provider(opt_in_id: str, accepted: bool) -> Optional[dict[str, Any]]:
    pool = get_pool()
    status = "both_accepted" if accepted else "declined_provider"
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            UPDATE orya.opt_ins
            SET status = $1, provider_accepted = $2, resolved_at = NOW()
            WHERE id = $3 AND status = 'pending_provider'
            RETURNING *
            """,
            status,
            accepted,
            opt_in_id,
        )
    return dict(row) if row else None


async def list_pending_opt_ins(user_id: str) -> list[dict[str, Any]]:
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT * FROM orya.opt_ins
            WHERE seeker_id = $1 AND status IN ('pending_seeker', 'pending_provider')
            OR provider_id = $1 AND status = 'pending_provider'
            ORDER BY created_at DESC
            """,
            user_id,
        )
    return [dict(r) for r in rows]


# ── Reflections ───────────────────────────────────────────────────
async def get_reflections(user_id: str) -> dict[str, Optional[str]]:
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT user_reflection, orya_reflection FROM orya.reflections WHERE user_id = $1",
            user_id,
        )
    if row:
        return {
            "user_reflection": row["user_reflection"],
            "orya_reflection": row["orya_reflection"],
        }
    return {"user_reflection": None, "orya_reflection": None}


async def save_reflections(
    user_id: str, user_reflection: Optional[str], orya_reflection: Optional[str]
) -> None:
    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO orya.reflections (user_id, user_reflection, orya_reflection)
            VALUES ($1, $2, $3)
            ON CONFLICT (user_id) DO UPDATE
                SET user_reflection = EXCLUDED.user_reflection,
                    orya_reflection = EXCLUDED.orya_reflection,
                    updated_at = NOW()
            """,
            user_id,
            user_reflection,
            orya_reflection,
        )
