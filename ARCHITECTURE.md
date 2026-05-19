# Orya — Architecture Complète & Plan d'Implémentation

> Document de planification technique. Aucun élément n'est optionnel.
> Tout ce qui est décrit ici DOIT être implémenté.

---

## 1. Vision Produit

Orya est un système conversationnel qui :
- Se comporte comme un **humain** (style texto, pas chatbot)
- **Extrait passivement** des informations des conversations (sans que l'utilisateur s'en rende compte)
- Construit un **knowledge graph temporel** de chaque utilisateur
- **Matche anonymement** les utilisateurs entre eux (seeker ↔ provider)
- Gère un **double opt-in** avant toute mise en relation
- Est **manageable** via des dashboards visuels (graphe, traces, API)

---

## 2. Stack Technologique Définitive

| Couche | Technologie | Rôle |
|--------|-------------|------|
| **Graph DB** | FalkorDB (Redis 8 + module) | Stockage du knowledge graph + vector index intégré |
| **Knowledge Graph Framework** | Graphiti (`graphiti-core[falkordb]`) | Extraction d'entités, relations, search hybrid, temporal tracking |
| **LLM — Conversation** | Groq `GroqClient` natif (Llama 4 Scout) | Réponses Orya — rapide, high RPM |
| **LLM — Graphiti (extraction)** | Nvidia `OpenAIGenericClient` (Llama 4 Maverick) | Entity extraction, summarization pour Graphiti |
| **LLM — Fallback** | Cerebras (Llama 3.1 8B) + OpenRouter (Llama 3.3 70B) | Quand Groq/Nvidia sont rate-limited |
| **Embeddings** | HuggingFace Inference API (`sentence-transformers/all-MiniLM-L6-v2`) | Embeddings pour Graphiti hybrid search — gratuit, compatible |
| **NER locale** | GLiNER2 (`fastino/gliner2-multi-v1`) | Extraction d'entités sur CPU, pas de rate limit, multilingue |
| **Agent Orchestration** | LangGraph (StateGraph + tools + MemorySaver) | Workflow agent : conversation → extraction → search → opt-in |
| **Observabilité** | LangSmith | Tracing de tous les appels LLM, latence, tokens, erreurs |
| **Gateway** | Hono + Bun WebSocket | Interface CLI ↔ serveur |
| **DB relationnelle** | PostgreSQL | Users, sessions, feedback, opt-in state, metadata persistante |
| **Persona** | System prompt + few-shot dynamique + feedback loop | Humanisation d'Orya |
| **CLI** | Bun/TypeScript | Client terminal pour les utilisateurs |

---

## 3. Architecture des Services

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CONTENEUR UNIQUE (Docker)                         │
│                                                                           │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  FalkorDB   │  │ PostgreSQL  │  │   Graphiti   │  │  FalkorDB    │  │
│  │  (Graph+Vec)│  │  (Users,    │  │  REST Server │  │  Browser UI  │  │
│  │  :6379      │  │   State)    │  │  :8000       │  │  :3000       │  │
│  │             │  │  :5432      │  │              │  │              │  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬───────┘  └──────────────┘  │
│         │                 │                 │                             │
│  ┌──────┴─────────────────┴─────────────────┴──────────────────────────┐ │
│  │                    PYTHON SERVICES (LangGraph)                        │ │
│  │                                                                       │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │ │
│  │  │              Agent Orya (LangGraph StateGraph)                   │ │ │
│  │  │                                                                  │ │ │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │ │ │
│  │  │  │Persona   │  │GLiNER2   │  │Search    │  │Double Opt-In  │  │ │ │
│  │  │  │Node      │  │Extractor │  │Tool      │  │Manager        │  │ │ │
│  │  │  │(respond) │  │(extract) │  │(match)   │  │(connect)      │  │ │ │
│  │  │  └──────────┘  └──────────┘  └──────────┘  └───────────────┘  │ │ │
│  │  │                                                                  │ │ │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                     │ │ │
│  │  │  │Memory    │  │Feedback  │  │Notify    │                     │ │ │
│  │  │  │Persist   │  │Loop      │  │User      │                     │ │ │
│  │  │  │(graphiti)│  │(rate)    │  │(push)    │                     │ │ │
│  │  │  └──────────┘  └──────────┘  └──────────┘                     │ │ │
│  │  └─────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                       │ │
│  │  Port :5001 (FastAPI — agent endpoint)                                │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                    GATEWAY (Hono + Bun WebSocket)                      │ │
│  │  Port :4001                                                            │ │
│  │  - /ws          → WebSocket pour les CLI                               │ │
│  │  - /health      → Status de tous les services                          │ │
│  │  - /internal/*  → Push events vers users connectés                     │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
│  supervisord gère : falkordb, postgres, graphiti-server, agent, gateway   │
└─────────────────────────────────────────────────────────────────────────┘

     Ports exposés :
     - 4001 → Gateway WebSocket (orya.globalimex.online)
     - 3000 → FalkorDB Browser (graph.globalimex.online)
     - 8000 → Graphiti REST API (api.globalimex.online)
```

---

## 4. Flux de Données Détaillé

### 4.1 — User envoie un message

```
CLI (bun) ──WebSocket──→ Gateway ──HTTP──→ Agent (LangGraph)
                                                │
                         ┌──────────────────────┼──────────────────────┐
                         ▼                      ▼                      ▼
                   [Persona Node]        [GLiNER2 Node]         [Memory Node]
                   Génère réponse        Extrait entités        Persiste dans
                   style humain          localement (CPU)       Graphiti
                         │                      │                      │
                         ▼                      ▼                      ▼
                   Reply → Gateway       Facts → Graphiti       Episode → FalkorDB
                         │               (add_episode)
                         ▼
                   WebSocket → CLI
```

### 4.2 — Détection d'intent "search"

```
Agent détecte intent "search" (via LLM ou heuristique)
         │
         ▼
   [Search Tool]
         │
         ├── graphiti.search(query, center_node_uuid=user_node)
         │   → Hybrid: semantic + BM25 + graph distance reranking
         │
         ├── Résultats filtrés (exclude self, exclude already matched)
         │
         ▼
   Si candidats trouvés :
         │
         ├── Notify user : "j'ai peut-être quelqu'un..."
         │
         └── [Double Opt-In Tool]
              ├── Demande au seeker : "ça te dit ?"
              ├── Si oui → Demande au provider : "quelqu'un a besoin de toi"
              └── Si les deux OK → Tunnel (échange contacts)
```

### 4.3 — Feedback Loop (humanisation sans fine-tuning)

```
User envoie /good ou /bad
         │
         ▼
   [Feedback Node]
         │
         ├── "good" → Stocke (input, response) dans PostgreSQL
         │             Devient few-shot dynamique pour le futur
         │
         └── "bad"  → Stocke comme negative example
                      Régénère avec contraintes renforcées
                      Si le retry est "good" → sauvegarde
```

---

## 5. LangGraph — Workflow Agent Détaillé

```python
# Structure du StateGraph Orya

class OryaState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    user_node_uuid: str  # Node Graphiti de l'user
    facts_context: str   # Facts connus injectés dans le prompt
    pending_search: Optional[Intent]
    pending_opt_in: Optional[str]

# Nodes du graph :
# 1. retrieve_context  → Search Graphiti pour facts de l'user
# 2. persona_respond   → LLM avec persona + facts + few-shot
# 3. persist_episode   → add_episode() dans Graphiti (fire-and-forget)
# 4. extract_entities  → GLiNER2 local + stocke dans graph
# 5. detect_intent     → Heuristique + LLM si nécessaire
# 6. search_graph      → graphiti.search() cross-group
# 7. double_opt_in     → State machine seeker/provider
# 8. notify_user       → Push via Gateway /internal/push

# Edges conditionnels :
# START → retrieve_context → persona_respond → persist_episode
#       → extract_entities → detect_intent
#       → SI intent=search : search_graph → double_opt_in → notify_user
#       → END
```

---

## 6. Dashboards & Interfaces Exposés

| Interface | Port | Domaine | Rôle |
|-----------|------|---------|------|
| **FalkorDB Browser** | 3000 | `graph.globalimex.online` | Visualisation Cypher du knowledge graph, exécuter des queries, voir nodes/edges/relations en temps réel |
| **Graphiti REST API** | 8000 | `api.globalimex.online` | `POST /ingest/messages`, `POST /retrieve/search`, `GET /retrieve/episodes/{group}`, `DELETE /entity/group/{id}` — gestion complète du graphe sans code |
| **Gateway Health** | 4001 | `orya.globalimex.online/health` | Status JSON de tous les services + métriques |
| **LangSmith** | Cloud | `smith.langchain.com` | Traces complètes de chaque appel LLM : prompt, response, latence, tokens, coût, erreurs, chain visualization |
| **PostgreSQL** | 5432 (interne) | — | Accessible via `psql` ou pgAdmin si besoin de debug |

---

## 7. Structure des Dossiers (Nouvelle)

```
orya_cli/
├── orya.ts                      # CLI client (existant, inchangé)
├── types.ts                     # Types partagés CLI ↔ Server
├── package.json
├── tsconfig.json
│
├── server/
│   ├── Dockerfile               # All-in-one (FalkorDB + PG + services)
│   ├── supervisord.conf         # Gère tous les process
│   ├── .env.example             # Template complet des variables
│   │
│   ├── gateway/                 # Hono + Bun WebSocket
│   │   ├── package.json
│   │   ├── src/
│   │   │   ├── index.ts         # Server Hono, /ws, /health, /internal
│   │   │   ├── router.ts        # Dispatch messages → Agent
│   │   │   └── sessions.ts      # Session manager (ws.raw identity)
│   │   └── Dockerfile
│   │
│   ├── agent/                   # Agent Orya (LangGraph + FastAPI)
│   │   ├── requirements.txt
│   │   ├── main.py              # FastAPI endpoint /chat, /feedback
│   │   ├── graph.py             # LangGraph StateGraph definition
│   │   ├── nodes/
│   │   │   ├── persona.py       # Node: génération réponse humaine
│   │   │   ├── extract.py       # Node: GLiNER2 + extraction
│   │   │   ├── search.py        # Node: Graphiti search tool
│   │   │   ├── memory.py        # Node: persist episode
│   │   │   ├── intent.py        # Node: detect intent
│   │   │   ├── opt_in.py        # Node: double opt-in state machine
│   │   │   └── notify.py        # Node: push to user via gateway
│   │   ├── providers/
│   │   │   ├── llm_router.py    # Multi-provider LLM (Groq/Nvidia/Cerebras/OR)
│   │   │   └── embedder.py      # HuggingFace sentence-transformers
│   │   ├── persona/
│   │   │   ├── system_prompt.py # Prompt Orya (style humain)
│   │   │   ├── few_shot.py      # Dynamic few-shot from feedback
│   │   │   └── negatives.py     # Negative examples (anti-chatbot)
│   │   ├── models/
│   │   │   ├── state.py         # OryaState TypedDict
│   │   │   └── schemas.py       # Pydantic models (request/response)
│   │   └── Dockerfile
│   │
│   ├── graphiti-server/         # Graphiti REST API (fork/config du server officiel)
│   │   ├── main.py              # FastAPI exposant Graphiti endpoints
│   │   ├── config.py            # Connection FalkorDB + LLM providers
│   │   └── requirements.txt
│   │
│   ├── db/                      # PostgreSQL init
│   │   ├── init.sql             # Schema: users, sessions, feedback, opt_ins
│   │   └── migrations/
│   │
│   └── docker-compose.yml       # Pour dev local (alternative au all-in-one)
```

---

## 8. Base de Données PostgreSQL — Schema

```sql
-- Users
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    alias VARCHAR(100) NOT NULL,
    graphiti_node_uuid VARCHAR(100),  -- Lien vers le node Graphiti
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Sessions (conversation state)
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    started_at TIMESTAMPTZ DEFAULT NOW(),
    last_message_at TIMESTAMPTZ,
    message_count INT DEFAULT 0
);

-- Feedback (pour few-shot dynamique)
CREATE TABLE feedback (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    user_input TEXT NOT NULL,
    orya_response TEXT NOT NULL,
    rating VARCHAR(10) NOT NULL CHECK (rating IN ('good', 'bad')),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Double Opt-In
CREATE TABLE opt_ins (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    seeker_id UUID REFERENCES users(id),
    provider_id UUID REFERENCES users(id),
    reason TEXT,
    status VARCHAR(20) DEFAULT 'pending_seeker'
        CHECK (status IN ('pending_seeker', 'pending_provider', 'both_accepted', 'declined', 'expired')),
    seeker_accepted BOOLEAN,
    provider_accepted BOOLEAN,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    resolved_at TIMESTAMPTZ
);
```

---

## 9. Variables d'Environnement Complètes

```env
# ── LLM Providers ─────────────────────────────────────────────────
GROQ_API_KEY=gsk_xxx
NVIDIA_API_KEY=nvapi-xxx
CEREBRAS_API_KEY=csk-xxx
OPENROUTER_API_KEY=sk-or-v1-xxx
HUGGINGFACE_API_KEY=hf_xxx

# ── Graphiti / FalkorDB ───────────────────────────────────────────
OPENAI_API_KEY=nvapi-xxx
FALKORDB_HOST=127.0.0.1
FALKORDB_PORT=6379

# ── PostgreSQL ────────────────────────────────────────────────────
POSTGRES_HOST=127.0.0.1
POSTGRES_PORT=5432
POSTGRES_DB=orya
POSTGRES_USER=orya
POSTGRES_PASSWORD=orya_secret_2026

# ── LangSmith Observabilité ───────────────────────────────────────
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<ta_clé_langsmith>
LANGCHAIN_PROJECT=orya-production

# ── Services internes ─────────────────────────────────────────────
AGENT_URL=http://127.0.0.1:5001
GRAPHITI_SERVER_URL=http://127.0.0.1:8000
GATEWAY_PORT=4001
```

---

## 10. Providers LLM — Détail des Limites

| Provider | Model | RPM | TPM | Usage |
|----------|-------|-----|-----|-------|
| **Groq** | `meta-llama/llama-4-scout-17b-16e-instruct` | 30 | 130K | Conversation Orya (primary) |
| **Nvidia** | `meta/llama-4-maverick-17b-128e-instruct` | 60 | 200K | Graphiti extraction + Conversation (secondary) |
| **Cerebras** | `llama3.1-8b` | 30 | 60K | Fallback rapide (fastest inference) |
| **OpenRouter** | `meta-llama/llama-3.3-70b-instruct:free` | 10 | 30K | Dernier fallback (qualité) |
| **HuggingFace** | `sentence-transformers/all-MiniLM-L6-v2` | ∞ (serverless) | — | Embeddings pour Graphiti |

**Stratégie de rotation :** Groq → Nvidia → Cerebras → OpenRouter. Rotation à chaque appel pour répartir la charge. Si rate-limited → next provider.

---

## 11. Graphiti — Configuration Providers

```python
from graphiti_core import Graphiti
from graphiti_core.llm_client.groq_client import GroqClient  # Client natif !
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.driver.falkordb_driver import FalkorDriver

# FalkorDB
driver = FalkorDriver(host="127.0.0.1", port=6379)

# LLM pour Graphiti (entity extraction, summarization)
# Utilise GroqClient natif — gère retries, structured output
llm_client = GroqClient()  # Lit GROQ_API_KEY automatiquement

# Embeddings via HuggingFace (gratuit, pas de rate limit)
embedder = OpenAIEmbedder(config=OpenAIEmbedderConfig(
    api_key=os.getenv("HUGGINGFACE_API_KEY"),
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    embedding_dim=384,
    base_url="https://api-inference.huggingface.co/pipeline/feature-extraction",
))

# Init Graphiti
graphiti = Graphiti(
    graph_driver=driver,
    llm_client=llm_client,
    embedder=embedder,
    cross_encoder=None,  # Pas de reranker OpenAI — on utilise center_node_uuid
)
await graphiti.build_indices_and_constraints()
```

---

## 12. GLiNER2 — Extraction Locale

```python
# Pas de rate limit, pas d'appel API, fonctionne sur CPU
# Modèle multilingue : français + anglais
from graphiti_core.llm_client.gliner2_client import GLiNER2Client

gliner_client = GLiNER2Client(
    llm_client=llm_client,  # Fallback LLM pour edges/facts complexes
    threshold=0.5,
    model_name="fastino/gliner2-multi-v1",  # Multilingue
)

# Utiliser GLiNER2 comme LLM client de Graphiti
graphiti = Graphiti(
    graph_driver=driver,
    llm_client=gliner_client,  # GLiNER2 pour NER + Groq pour le reste
    embedder=embedder,
    cross_encoder=None,
)
```

---

## 13. LangSmith — Tracing

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "<clé>"
os.environ["LANGCHAIN_PROJECT"] = "orya-production"

# Chaque appel LLM (conversation, extraction, search) est automatiquement
# tracé dans LangSmith quand on utilise LangGraph + LangChain.
# Dashboard : https://smith.langchain.com/
#
# Ce qu'on voit :
# - Chaque message user → response Orya (latence, tokens)
# - Les appels Graphiti (add_episode, search)
# - Les extractions GLiNER2
# - Les erreurs de rate limiting
# - Le coût estimé par conversation
```

---

## 14. Ports Exposés (Dockerfile final)

```dockerfile
EXPOSE 4001  # Gateway WebSocket (CLI se connecte ici)
EXPOSE 3000  # FalkorDB Browser (visualisation graphe)
EXPOSE 8000  # Graphiti REST API (management du graphe)
# 6379 interne (FalkorDB Redis)
# 5432 interne (PostgreSQL)
# 5001 interne (Agent LangGraph)
```

---

## 15. Déploiement Coolify

| Paramètre | Valeur |
|-----------|--------|
| Dockerfile | `server/Dockerfile` |
| Build context | `server/` |
| Branch | `main` |
| Ports mappés | `4001`, `3000`, `8000` |
| Domaines | `orya.globalimex.online:4001`, `graph.globalimex.online:3000`, `api.globalimex.online:8000` |
| Volume | `/data` (FalkorDB), `/var/lib/postgresql/data` (PG) |
| Env vars | Voir section 9 |

---

## 16. Résumé Exécutif

```
Ce système combine :
- La mémoire conversationnelle de Graphiti (knowledge graph temporel)
- L'extraction d'entités locale de GLiNER2 (zéro rate limit)
- L'orchestration élégante de LangGraph (state machine)
- L'observabilité de LangSmith (debug en prod)
- La vitesse de Groq/Cerebras (inference <100ms)
- La persistance PostgreSQL (users, feedback, opt-in)
- L'interface de debug FalkorDB Browser (visualisation graphe)
- L'API REST Graphiti (management sans code)
- Le style Character.AI via persona engineering (sans fine-tuning)
- Le double opt-in anonyme (privacy-first matching)

Le tout dans UN SEUL conteneur Docker, deployable sur Coolify.
```
