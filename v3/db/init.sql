-- Schéma Orya v3 — PostgreSQL
-- Idempotent : peut être réexécuté sans erreur

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE SCHEMA IF NOT EXISTS orya;

-- ── Users ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS orya.users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    alias VARCHAR(100) NOT NULL,
    graphiti_node_uuid VARCHAR(100),
    tutoyer BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ── Sessions ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS orya.sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES orya.users(id) ON DELETE CASCADE,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    last_message_at TIMESTAMPTZ,
    message_count INT DEFAULT 0
);

-- ── Feedback (few-shot dynamique) ─────────────────────────────────
CREATE TABLE IF NOT EXISTS orya.feedback (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES orya.users(id) ON DELETE CASCADE,
    user_input TEXT NOT NULL,
    orya_response TEXT NOT NULL,
    rating VARCHAR(10) NOT NULL CHECK (rating IN ('good', 'bad')),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_feedback_user ON orya.feedback(user_id);
CREATE INDEX IF NOT EXISTS idx_feedback_rating ON orya.feedback(rating);

-- ── Double Opt-In ─────────────────────────────────────────────────
DO $$ BEGIN
    CREATE TYPE orya.opt_in_status AS ENUM (
        'pending_seeker',
        'pending_provider',
        'both_accepted',
        'declined_seeker',
        'declined_provider',
        'expired'
    );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

CREATE TABLE IF NOT EXISTS orya.opt_ins (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    seeker_id UUID REFERENCES orya.users(id) ON DELETE CASCADE,
    provider_id UUID REFERENCES orya.users(id) ON DELETE CASCADE,
    candidate_uuid VARCHAR(100),
    reason TEXT,
    status orya.opt_in_status DEFAULT 'pending_seeker',
    seeker_accepted BOOLEAN,
    provider_accepted BOOLEAN,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    resolved_at TIMESTAMPTZ,
    UNIQUE(seeker_id, provider_id, candidate_uuid)
);

CREATE INDEX IF NOT EXISTS idx_opt_ins_seeker ON orya.opt_ins(seeker_id);
CREATE INDEX IF NOT EXISTS idx_opt_ins_provider ON orya.opt_ins(provider_id);
CREATE INDEX IF NOT EXISTS idx_opt_ins_status ON orya.opt_ins(status);

-- ── Reflections (mémoire long terme textuelle) ────────────────────
CREATE TABLE IF NOT EXISTS orya.reflections (
    user_id UUID PRIMARY KEY REFERENCES orya.users(id) ON DELETE CASCADE,
    user_reflection TEXT,
    orya_reflection TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ── MemBrain: Entities ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS orya.mb_entities (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(128) NOT NULL,
    entity_id VARCHAR(64) NOT NULL,
    canonical_ref VARCHAR(512) NOT NULL,
    description TEXT DEFAULT '',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, entity_id)
);
CREATE INDEX IF NOT EXISTS idx_mb_entities_user ON orya.mb_entities(user_id);
CREATE INDEX IF NOT EXISTS idx_mb_entities_canonical ON orya.mb_entities(user_id, canonical_ref);

-- ── MemBrain: Facts ─────────────────────────────────────────────
-- NOTE: Embeddings are stored in Qdrant (vector DB), not here.
CREATE TABLE IF NOT EXISTS orya.mb_facts (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(128) NOT NULL,
    text TEXT NOT NULL,
    search_text TEXT,
    session_number INT,
    fact_ts VARCHAR(64),
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'invalidated')),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_mb_facts_user ON orya.mb_facts(user_id);
CREATE INDEX IF NOT EXISTS idx_mb_facts_status ON orya.mb_facts(user_id, status);

-- ── MemBrain: Fact Refs (many-to-many) ──────────────────────────
CREATE TABLE IF NOT EXISTS orya.mb_fact_refs (
    id SERIAL PRIMARY KEY,
    fact_id INT REFERENCES orya.mb_facts(id) ON DELETE CASCADE,
    entity_id VARCHAR(64) NOT NULL,
    alias_text VARCHAR(512) NOT NULL,
    user_id VARCHAR(128) NOT NULL,
    UNIQUE(fact_id, entity_id, alias_text)
);
CREATE INDEX IF NOT EXISTS idx_mb_fact_refs_entity ON orya.mb_fact_refs(user_id, entity_id);
CREATE INDEX IF NOT EXISTS idx_mb_fact_refs_fact ON orya.mb_fact_refs(fact_id);

-- ── MemBrain: Entity Tree Nodes ─────────────────────────────────
CREATE TABLE IF NOT EXISTS orya.mb_entity_trees (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(128) NOT NULL,
    entity_id VARCHAR(64) NOT NULL,
    parent_id INT REFERENCES orya.mb_entity_trees(id) ON DELETE CASCADE,
    node_type VARCHAR(20) NOT NULL CHECK (node_type IN ('root', 'aspect', 'leaf')),
    fact_id INT REFERENCES orya.mb_facts(id) ON DELETE CASCADE,
    description TEXT,
    support INT DEFAULT 0,
    fresh_count INT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_mb_trees_unique_leaf ON orya.mb_entity_trees(user_id, entity_id, fact_id) WHERE (fact_id IS NOT NULL);
CREATE INDEX IF NOT EXISTS idx_mb_trees_user_entity ON orya.mb_entity_trees(user_id, entity_id);
CREATE INDEX IF NOT EXISTS idx_mb_trees_parent ON orya.mb_entity_trees(parent_id);

-- ── MemBrain: Session Summaries ─────────────────────────────────
CREATE TABLE IF NOT EXISTS orya.mb_session_summaries (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(128) NOT NULL,
    session_number INT NOT NULL,
    subject TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, session_number)
);
CREATE INDEX IF NOT EXISTS idx_mb_summaries_user ON orya.mb_session_summaries(user_id);

-- ── MemBrain: Time Annotations ──────────────────────────────────
CREATE TABLE IF NOT EXISTS orya.mb_time_annotations (
    id SERIAL PRIMARY KEY,
    fact_id INT REFERENCES orya.mb_facts(id) ON DELETE CASCADE,
    time_raw TEXT NOT NULL,
    time_resolved VARCHAR(64) NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mb_time_fact ON orya.mb_time_annotations(fact_id);

-- ── MemBrain: Match Index (cross-user) ──────────────────────────
CREATE TABLE IF NOT EXISTS orya.mb_match_index (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(128) NOT NULL,
    entity_id VARCHAR(64) NOT NULL,
    canonical_ref VARCHAR(512) NOT NULL,
    fact_summary TEXT NOT NULL,
    category VARCHAR(64),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_mb_match_user ON orya.mb_match_index(user_id);
CREATE INDEX IF NOT EXISTS idx_mb_match_entity ON orya.mb_match_index(entity_id);
CREATE INDEX IF NOT EXISTS idx_mb_match_category ON orya.mb_match_index(category);

-- ── Schema versioning ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS orya.schema_version (
    version INT PRIMARY KEY,
    applied_at TIMESTAMPTZ DEFAULT NOW()
);

INSERT INTO orya.schema_version (version) VALUES (4)
ON CONFLICT (version) DO NOTHING;
