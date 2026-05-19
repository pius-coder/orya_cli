# Step 01: Analyze (deep — documentation-driven)

**Task:** Implémenter Orya v2 sur branche v2 selon SESSION_CONTEXT.md
**Started:** 2026-05-19T11:28:38Z
**Methodology:** llms.txt → fetch every relevant page → cross-reference with existing code → consolidate

---

## 1. Codebase Context (existing repo)

### Files & Roles

| File | Role | v2 status |
|------|------|-----------|
| `orya.ts`, `types.ts`, `package.json`, `tsconfig.json` | CLI client (Bun + readline) | ✅ **Inchangé** — protocole WS déjà bon |
| `ARCHITECTURE.md` | Plan technique complet | Source de vérité pour v2 |
| `SESSION_CONTEXT.md` | Mémoire de session précédente | Pièges connus à respecter |
| `server/Dockerfile` | All-in-one Ubuntu 24.04 + FalkorDB stage | **À étendre** : PostgreSQL + graphiti-server + agent dir |
| `server/supervisord.conf` | falkordb + memory + agent-orya + orchestrator + gateway | **À refactorer** : 5 process (falkordb, postgres, graphiti-server, agent, gateway) |
| `server/.env.example` | Variables LLM | **À étendre** : POSTGRES_*, LANGCHAIN_*, GRAPHITI_SERVER_URL, semaphore |
| `server/docker-compose.yml` | Alternative dev | **À aligner** sur la nouvelle archi |
| `server/gateway/src/{index,router,sessions}.ts` | Hono + Bun WS, sessions par `ws.raw` | ✅ Quasi inchangé — `router.ts` simplifié pour 1 seul appel agent |
| `server/services/agent-orya/*` | Agent custom (httpx + persona JSON) | **À supprimer** — remplacé par `server/agent/` (LangGraph) |
| `server/services/orchestrator/*` | Pipeline async séparé | **À supprimer** — fusionné dans `server/agent/` (nodes LangGraph) |
| `server/services/memory/*` | Graphiti standalone (Nvidia embedder bug) | **À supprimer** — Graphiti instancié dans `server/agent/` + REST officiel |

### Protocole WebSocket à conserver (contrat client/serveur)

Vu dans `types.ts` :

- **Client → server** : `register {userId, alias?}`, `message {text}`, `tutoyer {value}`, `ping`
- **Server → client** : `registered`, `typing {value}`, `reply {text}`, `candidates {items[]}`, `system {text}`, `trace {step,detail?}`, `fact_recorded {label,value,confidence}`, `address_form {countries,intent}`, `pong`

→ Le gateway garde ce contrat exact. L'agent communique avec le gateway via `POST /internal/push/:userId` (déjà implémenté) pour les événements asynchrones (candidates, fact_recorded, trace).

### Bugs connus (de SESSION_CONTEXT.md, confirmés par la doc Graphiti)

1. **Embedder Nvidia** (`memory/main.py`) : `nvidia/nv-embedqa-e5-v5` exige `input_type` → **HuggingFace `all-MiniLM-L6-v2` (dim=384)** via OpenAI-compat ; OU `OpenAIEmbedder` direct avec base_url HF.
2. **cross_encoder** : `OpenAIRerankerClient()` default crash sans `OPENAI_API_KEY`. Les options sont :
   - **Option A** : passer `cross_encoder=None` (perdre le reranking ; `RRF` reste disponible).
   - **Option B** : `OpenAIRerankerClient(client=llm_client, config=LLMConfig(model="...", base_url=..., api_key=GROQ_API_KEY))`.
   - **Option C** : `BGERerankerClient` local (sentence-transformers, CPU).
   → **Choix** : Option A (cross_encoder=None) — le RRF est déjà excellent et on évite tout call externe pour le rerank.
3. **Package** : `graphiti-core[falkordb]` → module `graphiti_core` (≠ `graphiti-core-falkordb` qui installe `graphiti_core_falkordb`). On utilise le premier.
4. **Python 3.12** Ubuntu 24.04 → `uv pip install --system --break-system-packages`.
5. **SEMAPHORE_LIMIT** : variable d'env Graphiti, défaut 10. Si rate-limit Groq → baisser à 3-5.

---

## 2. Documentation digest (from llms.txt + fetched pages)

### 2.1 Graphiti `graphiti-core[falkordb]` — Source : help.getzep.com/graphiti

#### Construction (FalkorDB driver)

```python
from graphiti_core import Graphiti
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.llm_client.groq_client import GroqClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

driver = FalkorDriver(host="127.0.0.1", port="6379", username=None, password=None)
llm = GroqClient(config=LLMConfig(
    api_key=os.getenv("GROQ_API_KEY"),
    model="meta-llama/llama-4-scout-17b-16e-instruct",     # main
    small_model="llama-3.1-8b-instant",                     # small (quick ops)
))
embedder = OpenAIEmbedder(config=OpenAIEmbedderConfig(
    api_key=os.getenv("HUGGINGFACE_API_KEY"),
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    embedding_dim=384,
    base_url="https://api-inference.huggingface.co/pipeline/feature-extraction",
))
graphiti = Graphiti(
    graph_driver=driver,
    llm_client=llm,
    embedder=embedder,
    cross_encoder=None,  # avoid OpenAIRerankerClient default
)
await graphiti.build_indices_and_constraints()  # idempotent — first run only
```

> **Important** : la doc officielle dit que pour Groq, il faut OpenAI keys pour embeddings/reranking. On contourne via HuggingFace (gratuit) pour les embeddings et `cross_encoder=None`.

#### API principale

| Méthode | Signature | Usage |
|---------|-----------|-------|
| `add_episode` | `name, episode_body, source: EpisodeType, source_description, reference_time, group_id?, entity_types?, edge_types?, edge_type_map?, excluded_entity_types?` | Ingère un événement (texte/message/json). LLM extrait entités+relations en arrière-plan. |
| `add_episode_bulk` | `[RawEpisode(...)]` | Bulk. **Pas d'edge invalidation** — pour graph vide ou bootstrap seulement. |
| `search` | `query, center_node_uuid?, group_ids?, num_results?` | Hybrid RRF (BM25 + semantic). Center node = rerank par distance. |
| `_search` | `query, config: SearchConfig, group_ids?` | Avancé — recipes : `NODE_HYBRID_SEARCH_RRF`, `EDGE_HYBRID_SEARCH_RRF`, `COMBINED_HYBRID_SEARCH_RRF`, etc. |
| `add_triplet` | `source_node, edge, target_node` | Manual fact triple. Auto-déduplique. |
| `get_nodes_by_query` | `query` | Find by natural language. |

#### Types d'épisodes

- `EpisodeType.text` — texte non structuré (article, doc).
- `EpisodeType.message` — conversationnel `"speaker: message\n..."`.
- `EpisodeType.json` — structuré (dict → `json.dumps` avant).

#### Graph namespacing — `group_id`

- **Multi-tenant** : chaque user a son `group_id = user_id`. Tous les episodes/entités/edges héritent.
- `search(group_ids=["user_a"])` → isole un user.
- `search(query, num_results=10)` SANS `group_ids` → **cross-group search** (cf. matching cross-user dans Orya !).

#### Custom Entity & Edge Types — clé pour Orya

```python
from pydantic import BaseModel, Field

# Entity types
class Person(BaseModel):
    """Une personne réelle."""
    occupation: str | None = Field(None, description="Métier ou rôle pro")
    location: str | None = Field(None, description="Ville/pays courant")
    age_range: str | None = Field(None, description="Tranche d'âge: jeune/adulte/senior")

class Skill(BaseModel):
    """Compétence professionnelle."""
    domain: str | None = Field(None, description="ex: dev front, plomberie, droit fiscal")
    seniority: str | None = Field(None, description="junior/intermediate/senior/expert")

class Need(BaseModel):
    """Besoin exprimé par un user (recherche d'aide ou de service)."""
    domain: str | None = Field(None, description="Domaine du besoin")
    urgency: str | None = Field(None, description="immediate/days/weeks/exploratory")
    location: str | None = Field(None, description="Ville où le besoin doit être résolu")

class City(BaseModel):
    """Ville."""
    country: str | None = Field(None, description="Pays")

class Company(BaseModel):
    """Entreprise."""
    industry: str | None = Field(None, description="Secteur")

# Edge types
class HasSkill(BaseModel):
    """Person possède Skill."""
    seniority: str | None = Field(None)
    years: int | None = Field(None)

class Wants(BaseModel):
    """Person exprime Need."""
    expressed_at: str | None = Field(None, description="ISO timestamp d'expression")

class LocatedIn(BaseModel):
    """Person ou Need ou Company → City."""

class WorksAt(BaseModel):
    """Person → Company."""
    role: str | None = Field(None)

edge_type_map = {
    ("Person", "Skill"): ["HasSkill"],
    ("Person", "Need"): ["Wants"],
    ("Person", "City"): ["LocatedIn"],
    ("Person", "Company"): ["WorksAt"],
    ("Need", "City"): ["LocatedIn"],
    ("Company", "City"): ["LocatedIn"],
    ("Entity", "Entity"): ["RELATES_TO"],  # fallback
}
```

→ Définit le domaine Orya précisément. Chaque message → `add_episode(entity_types=…, edge_types=…, edge_type_map=…)`.

#### Search recipes utiles pour Orya

- `EDGE_HYBRID_SEARCH_RRF` — chercher des **facts** (e.g. "qui sait coder Python à Lyon").
- `NODE_HYBRID_SEARCH_NODE_DISTANCE` — recommander des **personnes proches** d'un user.
- `EDGE_HYBRID_SEARCH_NODE_DISTANCE` — facts proches d'un user (pour `retrieve_context`).

#### Concurrency / telemetry

- `SEMAPHORE_LIMIT=10` (default). Baisser à `5` pour Groq free tier.
- `GRAPHITI_TELEMETRY_ENABLED=false` à set en prod.

### 2.2 LangGraph — Source : docs.langchain.com/oss/python/langgraph

#### Concepts utilisés

- **`StateGraph(State, context_schema=Context)`** : graph builder. State est `TypedDict` (recommandé) ou Pydantic.
- **`MessagesState`** prebuilt avec `messages: Annotated[list[AnyMessage], add_messages]`.
- **`add_messages` reducer** : append + handle message ID updates (HITL friendly).
- **Node** : `def node(state, runtime: Runtime[Context])` ; retourne dict d'updates.
- **Edges** : `add_edge(a, b)` (static) ; `add_conditional_edges(node, fn, dict?)`.
- **`Command(update=, goto=)`** : combine state + routing dans un node. Type-annoté `Command[Literal["next_node"]]`.
- **`interrupt(payload)`** : pause graph pour HITL → utile pour double opt-in (le user reçoit la candidate, son `oui/non` reprend le graph).
- **`Runtime`** : injection de contexte runtime (`runtime.context`, `runtime.store`, `runtime.execution_info`).

#### Persistence (checkpointer)

- `InMemorySaver` (dev/tests).
- **`AsyncPostgresSaver.from_conn_string(DB_URI)`** (prod). Package `langgraph-checkpoint-postgres`. Premier run : `await checkpointer.setup()`.
- `thread_id` config-key obligatoire pour persister.
- Stockage par checkpoint à chaque super-step → re-démarrage robuste.

#### Streaming v2

```python
async for chunk in graph.astream(
    {"messages": [...]},
    config={"configurable": {"thread_id": "..."}},
    stream_mode=["messages", "updates", "custom"],
    version="v2",
    subgraphs=True,
):
    if chunk["type"] == "messages":
        msg, metadata = chunk["data"]            # token + langgraph_node
        if metadata["langgraph_node"] == "persona":
            ws_send_token(msg.content)
    elif chunk["type"] == "updates":
        for node, update in chunk["data"].items():
            ...
    elif chunk["type"] == "custom":
        # emis depuis un node via get_stream_writer()
        writer({"trace": "step", "detail": ...})
```

→ Map direct sur le protocole CLI : `messages` mode → `reply` token-by-token (option future), `custom` mode → `trace` events.

#### Async + Python 3.12 (notre cas)

✅ Pas de souci : Python 3.11+ supporte `asyncio.create_task(context=...)`. `get_stream_writer()` fonctionne dans nodes async.

#### Idempotence des nodes (avec `interrupt`)

- Le node redémarre depuis son début à la reprise. Side-effects pré-`interrupt` doivent être idempotents.
- → Pour le double opt-in : créer la ligne `opt_ins(status='pending')` AVANT `interrupt` est OK si on utilise `INSERT ... ON CONFLICT DO NOTHING` ou `MERGE` Cypher.

### 2.3 LangChain providers — Source : docs.langchain.com/oss/python/integrations/providers

| Provider | Class | Package | Notes |
|----------|-------|---------|-------|
| Groq | `ChatGroq` | `langchain-groq` | `init_chat_model("groq:llama-3.3-70b-versatile")` |
| Nvidia | `ChatNVIDIA` | `langchain-nvidia-ai-endpoints` | Heavy deps (gRPC). **Préférer `ChatOpenAI(base_url=NVIDIA_URL)`** |
| Cerebras | (via OpenAI compat) | `langchain-openai` | `ChatOpenAI(base_url="https://api.cerebras.ai/v1", model="llama3.1-8b")` |
| OpenRouter | (via OpenAI compat) | `langchain-openai` | `ChatOpenAI(base_url="https://openrouter.ai/api/v1", model="meta-llama/llama-3.3-70b-instruct:free")` |

→ **Stratégie** : LangChain `ChatOpenAI` pour les 3 providers OpenAI-compat (Nvidia/Cerebras/OR), `ChatGroq` pour Groq. Combiner via `RunnableWithFallbacks` (`.with_fallbacks([...])`).

```python
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

primary = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.7)
fallback_1 = ChatOpenAI(
    model="meta/llama-4-maverick-17b-128e-instruct",
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0.7,
)
fallback_2 = ChatOpenAI(
    model="llama3.1-8b",
    base_url="https://api.cerebras.ai/v1",
    api_key=os.getenv("CEREBRAS_API_KEY"),
    temperature=0.7,
)
fallback_3 = ChatOpenAI(
    model="meta-llama/llama-3.3-70b-instruct:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.7,
)

llm = primary.with_fallbacks([fallback_1, fallback_2, fallback_3])
```

### 2.4 LangSmith — Source : docs.langchain.com/langsmith

Activation par env vars seulement (auto-tracing pour LangChain + LangGraph) :

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_pt_...
LANGSMITH_PROJECT=orya-v2
# legacy aliases:
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_...
LANGCHAIN_PROJECT=orya-v2
```

→ Aucun code à écrire. Les invocations `model.ainvoke`, `graph.astream`, etc. sont auto-tracées. Pour des fonctions custom : `@traceable`.

### 2.5 Bun WebSocket + Hono — Source : bun.com/docs + existing code

- `Bun.serve({ fetch, websocket: { open, message, close, drain }, idleTimeout, maxPayloadLength })`.
- Données contextuelles via `server.upgrade(req, { data: { userId, alias } })` puis `ws.data`.
- Le code existant utilise `createBunWebSocket` from `hono/bun` (adapter), c'est le bon pattern et ça marche.
- Server-side broadcast : `server.publish(topic, msg)` ; `ws.subscribe(topic)`.

### 2.6 PostgreSQL pour données métier (en plus de checkpointer LangGraph)

- LangGraph utilise `AsyncPostgresSaver` pour stocker `checkpoints` (state du graph par thread).
- Mais on a aussi besoin de tables métier : `users`, `sessions`, `feedback`, `opt_ins`, `match_queues`, `notifications`. → tables custom gérées via `asyncpg` ou `psycopg`.
- Architecture : 1 seule DB Postgres, schéma `public` pour LangGraph (auto), schéma `orya` pour le métier.

### 2.7 FalkorDB — Source : docs.falkordb.com (llms.txt)

- Image Docker officielle `falkordb/falkordb:latest`. Expose `6379` (Redis protocol) et `3000` (Browser).
- Stocke `falkordb.so` module Redis + `redis-server` + `redis-cli`. On les copie dans notre image dans un build stage (déjà fait dans le Dockerfile actuel).
- Persistence : RDB + AOF activables via config Redis. Volume `/data` pour persister le graph.
- Cypher avec extensions FalkorDB (vector, full-text, range indexes).

---

## 3. Domain model Orya — confirmé

### Tables PostgreSQL (schéma `orya`)

```sql
CREATE SCHEMA IF NOT EXISTS orya;

CREATE TABLE orya.users (
    user_id     TEXT PRIMARY KEY,
    alias       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at TIMESTAMPTZ,
    tutoyer     BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE orya.sessions (
    session_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      TEXT REFERENCES orya.users(user_id) ON DELETE CASCADE,
    started_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at     TIMESTAMPTZ
);

CREATE TABLE orya.feedback (
    feedback_id  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      TEXT REFERENCES orya.users(user_id),
    user_text    TEXT NOT NULL,
    assistant_reply TEXT NOT NULL,
    rating       SMALLINT NOT NULL CHECK (rating IN (-1, 1)),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX feedback_user_id_idx ON orya.feedback(user_id);
CREATE INDEX feedback_rating_idx ON orya.feedback(rating);

CREATE TYPE orya.opt_in_status AS ENUM (
    'pending_seeker', 'rejected_seeker',
    'pending_provider', 'rejected_provider',
    'matched', 'expired'
);

CREATE TABLE orya.opt_ins (
    opt_in_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    seeker_id    TEXT NOT NULL REFERENCES orya.users(user_id),
    provider_id  TEXT NOT NULL REFERENCES orya.users(user_id),
    need_summary TEXT NOT NULL,
    candidate_uuid TEXT NOT NULL,           -- node uuid Graphiti
    status       orya.opt_in_status NOT NULL DEFAULT 'pending_seeker',
    seeker_decision_at  TIMESTAMPTZ,
    provider_decision_at TIMESTAMPTZ,
    matched_at   TIMESTAMPTZ,
    expires_at   TIMESTAMPTZ NOT NULL DEFAULT (now() + interval '72 hours'),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (seeker_id, provider_id, candidate_uuid)
);
CREATE INDEX opt_ins_seeker_status_idx ON orya.opt_ins(seeker_id, status);
CREATE INDEX opt_ins_provider_status_idx ON orya.opt_ins(provider_id, status);
```

### Custom Entity/Edge types Graphiti (cf. §2.1)

`Person`, `Skill`, `Need`, `City`, `Company` + `HasSkill`, `Wants`, `LocatedIn`, `WorksAt` + fallback `RELATES_TO`.

### LangGraph State — schéma

```python
class OryaState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]   # short-term thread
    user_id: str
    user_alias: str | None
    facts_context: list[str]                              # facts from graph search
    extracted_facts: list[dict]                           # quick-extract results for UI feedback
    intent: dict | None                                   # detect_intent output
    candidates: list[dict]                                # search results for matching
    pending_opt_in: dict | None                           # active opt-in waiting decision
    last_user_text: str
    last_assistant_reply: str
```

### Node DAG (LangGraph)

```
START
  ↓
retrieve_context  ← search graphiti(user_id, last_user_text)
  ↓
persona_respond   ← LLM router with facts_context as system context
  ↓
persist_episode   ← graphiti.add_episode(message format, group_id=user_id)
  ↓
extract_quick     ← rule-based extraction → fact_recorded events
  ↓
detect_intent     ← heuristic + LLM fallback → intent.action
  ↓
[conditional]
  ├─ search_match → candidates → opt_in_propose → notify_user → END
  └─ END  (no match needed)
```

`notify_user` POST → `${GATEWAY_URL}/internal/push/{user_id}` avec types `reply`, `candidates`, `fact_recorded`, `trace`.

---

## 4. Inferred Acceptance Criteria

- [ ] AC1: Branche `v2` (créée) — code v2 commité/pushé, PR ouverte vers main.
- [ ] AC2: `server/agent/` (LangGraph) avec `graph.py` + nodes (`retrieve_context`, `persona_respond`, `persist_episode`, `extract_quick`, `detect_intent`, `search_match`, `opt_in_propose`, `notify_user`) + `providers/` (`llm_router.py`, `embedder.py`, `graphiti_client.py`) + `persona/` (`system_prompt.py`, `negatives.py`, `few_shot.py`) + `models/` (`state.py`, `entities.py`, `schemas.py`) + `db/postgres.py` + `main.py` (FastAPI) + `requirements.txt`.
- [ ] AC3: `server/db/init.sql` avec schéma `orya` complet (users, sessions, feedback, opt_ins).
- [ ] AC4: `server/graphiti-server/` (FastAPI) port 8000 — endpoints `/health`, `/ingest/messages`, `/retrieve/search`, `/retrieve/episodes/{group}`.
- [ ] AC5: Anciens services (`agent-orya`, `orchestrator`, `memory`) supprimés.
- [ ] AC6: `server/Dockerfile` mis à jour : ajout PostgreSQL apt, init script `init.sql`, agent + graphiti-server workdirs, retrait des anciens.
- [ ] AC7: `server/supervisord.conf` gère 5 process : falkordb (priority=1), postgres (priority=1), graphiti-server (priority=2), agent (priority=3), gateway (priority=4).
- [ ] AC8: `server/gateway/src/router.ts` simplifié : 1 seul appel à `${AGENT_URL}/chat`. Les events async (candidates, fact_recorded, trace) viennent du gateway via `/internal/push`.
- [ ] AC9: HuggingFace embedder utilisé (dim=384), `cross_encoder=None`, `GroqClient` natif, custom Pydantic entity/edge types passés à `add_episode`.
- [ ] AC10: `server/.env.example` complet (LLM keys, Postgres, Graphiti, LangSmith, semaphore).
- [ ] AC11: AsyncPostgresSaver utilisé pour LangGraph thread persistence.
- [ ] AC12: LangChain `with_fallbacks` chaining 4 providers (Groq → Nvidia → Cerebras → OpenRouter).
- [ ] AC13: Build Docker réussi (sandbox).
- [ ] AC14: Sanity tests : import-only smoke test des modules Python ; `tsc --noEmit` sur gateway ; `bun --bun` sur orya.ts.

---

## 5. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Graphiti `GroqClient` exige `OPENAI_API_KEY` pour reranker default | `cross_encoder=None` (RRF suffit). |
| HuggingFace serverless embeddings cold/slow | Acceptable MVP ; documenter fallback `sentence-transformers` local. |
| LangGraph PostgresSaver requiert `setup()` au premier run | `await checkpointer.setup()` dans `lifespan` FastAPI. |
| `langchain-nvidia-ai-endpoints` heavy (gRPC) | Utiliser `ChatOpenAI(base_url=NVIDIA_URL)` à la place. |
| Postgres + FalkorDB + Bun + Python dans un seul container = ~2 GB | Acceptable pour MVP Coolify. Documenter split possible plus tard. |
| Build sandbox réseau-restreint | On vérifie le syntax/structure ; pas de full build E2E. |
| Concurrence LLM Graphiti vs LangChain | `SEMAPHORE_LIMIT=5` env ; les 2 utilisent providers différents (Graphiti=Groq direct, LangChain=fallback chain). |
| Validation `interrupt()` pour double opt-in | On utilise tables PG + nodes LangGraph séparés (pas `interrupt`) — plus simple et l'opt-in est asynchrone (sur jours). |

→ Note importante : on **ne pas utiliser `interrupt()`** pour le double opt-in. Trop fragile. À la place : `opt_in_propose` insère une row `pending_seeker` dans Postgres, le gateway envoie `candidates` au seeker, et quand le seeker répond `oui/non`, le CLI envoie un message `{type: "opt_in_response", opt_in_id, decision}` qui re-invoque le graph avec un input différent. C'est cleaner.

---

## 6. References (verified via web fetch)

- Graphiti llms.txt : https://help.getzep.com/graphiti/llms.txt
- Graphiti Quick Start : https://help.getzep.com/graphiti/getting-started/quick-start.mdx
- Graphiti FalkorDB Configuration : https://help.getzep.com/graphiti/configuration/falkor-db-configuration.mdx
- Graphiti LLM Configuration : https://help.getzep.com/graphiti/configuration/llm-configuration.mdx
- Graphiti Adding Episodes : https://help.getzep.com/graphiti/core-concepts/adding-episodes.mdx
- Graphiti Searching : https://help.getzep.com/graphiti/working-with-data/searching.mdx
- Graphiti Graph Namespacing : https://help.getzep.com/graphiti/core-concepts/graph-namespacing.mdx
- Graphiti Custom Entity/Edge Types : https://help.getzep.com/graphiti/core-concepts/custom-entity-and-edge-types.mdx
- Graphiti LangGraph integration : https://help.getzep.com/graphiti/integrations/lang-graph-agent.mdx
- LangChain llms.txt : https://docs.langchain.com/llms.txt
- LangGraph Graph API : https://docs.langchain.com/oss/python/langgraph/graph-api.md
- LangGraph Use Graph API : https://docs.langchain.com/oss/python/langgraph/use-graph-api.md
- LangGraph Persistence : https://docs.langchain.com/oss/python/langgraph/persistence.md
- LangGraph Quickstart : https://docs.langchain.com/oss/python/langgraph/quickstart.md
- LangGraph Streaming : https://docs.langchain.com/oss/python/langgraph/streaming.md
- LangGraph Add Memory : https://docs.langchain.com/oss/python/langgraph/add-memory.md
- LangGraph Interrupts : https://docs.langchain.com/oss/python/langgraph/interrupts.md
- LangGraph llms.txt : https://langchain-ai.github.io/langgraph/llms.txt
- LangChain providers Groq : https://docs.langchain.com/oss/python/integrations/providers/groq.md
- LangChain providers Nvidia : https://docs.langchain.com/oss/python/integrations/providers/nvidia.md
- LangSmith Trace LangGraph : https://docs.langchain.com/langsmith/trace-with-langgraph.md
- Bun llms.txt : https://bun.com/docs/llms.txt
- Bun WebSockets : https://bun.com/docs/runtime/http/websockets.md
- Hono llms.txt : https://hono.dev/llms.txt
- FalkorDB llms.txt : https://docs.falkordb.com/llms.txt

> Some FalkorDB pages (e.g. `/getting-started`, `/browser`, `/agentic-memory/graphiti`) returned HTTP errors during this analysis. Compensated by reading the index llms.txt + the Graphiti FalkorDB Configuration page from getzep, which together give the full picture for our usage.
