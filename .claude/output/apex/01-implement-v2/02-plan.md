# Step 02: Plan (refined post-doc-analysis)

**Task:** Implémenter Orya v2 sur branche v2 selon SESSION_CONTEXT.md
**Started:** 2026-05-19T11:28:38Z

---

## Implementation order (avoids broken-state windows)

### Phase A — Cleanup (delete obsolete code)

```
DELETE  server/services/agent-orya/    (5 files + dir)
DELETE  server/services/orchestrator/  (5 files + dir)
DELETE  server/services/memory/        (3 files + dir)
DELETE  server/services/               (empty parent)
```

### Phase B — PostgreSQL schema

| File | Purpose |
|------|---------|
| `server/db/init.sql` | Schema `orya` complet (users, sessions, feedback, opt_ins) — voir analyze §3. Idempotent: `CREATE … IF NOT EXISTS`. |

### Phase C — Agent (LangGraph) `server/agent/`

```
server/agent/
├── requirements.txt
├── main.py                        # FastAPI lifespan: init Graphiti + PG pool + compile graph
├── graph.py                       # LangGraph StateGraph builder
├── settings.py                    # env vars + constants (single source of truth)
├── models/
│   ├── __init__.py
│   ├── state.py                   # OryaState TypedDict
│   ├── entities.py                # Pydantic Person/Skill/Need/City/Company + edges
│   └── schemas.py                 # ChatRequest/Response, FeedbackRequest, OptInResponse
├── providers/
│   ├── __init__.py
│   ├── llm_router.py              # ChatGroq + ChatOpenAI(NVIDIA/CEREBRAS/OR) + with_fallbacks
│   ├── embedder.py                # build HF OpenAIEmbedder
│   └── graphiti_client.py         # Singleton Graphiti instance + builder
├── persona/
│   ├── __init__.py
│   ├── system_prompt.py           # Migration verbatim from agent-orya/persona.py
│   ├── negatives.py               # Negative few-shot examples
│   └── few_shot.py                # build_messages() with positive examples from PG
├── nodes/
│   ├── __init__.py
│   ├── retrieve_context.py
│   ├── persona_respond.py
│   ├── persist_episode.py
│   ├── extract_quick.py
│   ├── detect_intent.py
│   ├── search_match.py
│   ├── opt_in_propose.py
│   └── notify_user.py
└── db/
    ├── __init__.py
    └── postgres.py                # asyncpg pool + helpers
```

### Phase D — Graphiti REST `server/graphiti-server/`

```
server/graphiti-server/
├── requirements.txt
├── main.py                # FastAPI :8000
├── settings.py            # shared env loader
└── client.py              # build same Graphiti instance as agent
```

### Phase E — Gateway updates

- `server/gateway/src/router.ts` — single call to `${AGENT_URL}/chat`.
- `server/gateway/src/index.ts` — env var `AGENT_URL`.

### Phase F — Docker / supervisor / env

- `server/Dockerfile` — rewrite (Postgres + agent/ + graphiti-server/ + bun gateway).
- `server/supervisord.conf` — 5 programs (priority order).
- `server/.env.example` — full vars.
- `server/docker-compose.yml` — mirror new layout.
- `server/scripts/init-postgres.sh` — initdb + run init.sql + exec postgres.

### Phase G — Validation

- python `import` smoke tests.
- `tsc --noEmit` for gateway + root.
- `bun --bun server/gateway/src/index.ts` syntax check.
- Document in PR what's NOT live-tested (LLM E2E needs keys).

### Phase H — Commit / Push / PR

- Single semantic commit per phase.
- Push to `v2`.
- Create PR `v2 → main`.
