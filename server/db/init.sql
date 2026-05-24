-- Orya v2 — PostgreSQL initialization
-- Schema: orya (business data: users, sessions, feedback, opt_ins)
-- LangGraph PostgresSaver creates its own schema via setup() at first run.

CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE SCHEMA IF NOT EXISTS orya;

-- ============================================================
-- Users
-- ============================================================
CREATE TABLE IF NOT EXISTS orya.users (
    user_id      TEXT PRIMARY KEY,
    alias        TEXT,
    tutoyer      BOOLEAN NOT NULL DEFAULT TRUE,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS users_alias_idx ON orya.users (alias);

-- ============================================================
-- Sessions (one per WS connection)
-- ============================================================
CREATE TABLE IF NOT EXISTS orya.sessions (
    session_id  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     TEXT NOT NULL REFERENCES orya.users(user_id) ON DELETE CASCADE,
    started_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at    TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS sessions_user_id_idx ON orya.sessions (user_id);

-- ============================================================
-- Feedback (good/bad ratings → dynamic few-shot)
-- ============================================================
CREATE TABLE IF NOT EXISTS orya.feedback (
    feedback_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         TEXT NOT NULL REFERENCES orya.users(user_id) ON DELETE CASCADE,
    user_text       TEXT NOT NULL,
    assistant_reply TEXT NOT NULL,
    rating          SMALLINT NOT NULL CHECK (rating IN (-1, 1)),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS feedback_user_id_idx ON orya.feedback (user_id);
CREATE INDEX IF NOT EXISTS feedback_rating_idx ON orya.feedback (rating);
CREATE INDEX IF NOT EXISTS feedback_created_at_idx ON orya.feedback (created_at DESC);

-- ============================================================
-- Opt-ins (double opt-in state machine)
-- ============================================================
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'opt_in_status') THEN
        CREATE TYPE orya.opt_in_status AS ENUM (
            'pending_seeker',
            'rejected_seeker',
            'pending_provider',
            'rejected_provider',
            'matched',
            'expired'
        );
    END IF;
END$$;

CREATE TABLE IF NOT EXISTS orya.opt_ins (
    opt_in_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    seeker_id            TEXT NOT NULL REFERENCES orya.users(user_id) ON DELETE CASCADE,
    provider_id          TEXT NOT NULL REFERENCES orya.users(user_id) ON DELETE CASCADE,
    need_summary         TEXT NOT NULL,
    candidate_uuid       TEXT NOT NULL,
    status               orya.opt_in_status NOT NULL DEFAULT 'pending_seeker',
    seeker_decision_at   TIMESTAMPTZ,
    provider_decision_at TIMESTAMPTZ,
    matched_at           TIMESTAMPTZ,
    expires_at           TIMESTAMPTZ NOT NULL DEFAULT (now() + interval '72 hours'),
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (seeker_id, provider_id, candidate_uuid)
);

CREATE INDEX IF NOT EXISTS opt_ins_seeker_status_idx
    ON orya.opt_ins (seeker_id, status);
CREATE INDEX IF NOT EXISTS opt_ins_provider_status_idx
    ON orya.opt_ins (provider_id, status);
CREATE INDEX IF NOT EXISTS opt_ins_expires_at_idx
    ON orya.opt_ins (expires_at);

-- ============================================================
-- Default test user (idempotent)
-- ============================================================
INSERT INTO orya.users (user_id, alias)
VALUES ('orya_default', 'Test User')
ON CONFLICT (user_id) DO NOTHING;

-- ============================================================
-- Versioning row (for future migrations)
-- ============================================================
CREATE TABLE IF NOT EXISTS orya.schema_version (
    version    INT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============================================================
-- Reflections (v3 — user portrait + orya self-cognition)
-- ============================================================
CREATE TABLE IF NOT EXISTS orya.reflections (
    reflection_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         TEXT NOT NULL REFERENCES orya.users(user_id) ON DELETE CASCADE,
    user_reflection TEXT,
    orya_reflection TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id)
);

CREATE INDEX IF NOT EXISTS reflections_user_id_idx ON orya.reflections (user_id);

-- ============================================================
-- Versioning row (for future migrations)
-- ============================================================
CREATE TABLE IF NOT EXISTS orya.schema_version (
    version    INT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO orya.schema_version (version)
VALUES (2)
ON CONFLICT (version) DO NOTHING;
