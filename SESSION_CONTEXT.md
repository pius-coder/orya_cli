# Orya — Session Context & Memory (pour la prochaine session)

> Ce document contient TOUT le contexte nécessaire pour reprendre l'implémentation v2.
> Colle ce fichier en début de prochaine session.

---

## État actuel du repo

- **Repo** : `pius-coder/orya_cli`
- **Branche active** : `docs/architecture-plan` (PR #6 ouverte vers main)
- **PRs ouvertes** : #4 (graphiti integration), #6 (architecture plan)
- **Ce qui fonctionne** : CLI + Gateway Hono + Agent Orya + FalkorDB + Conversation humaine + Extraction de facts
- **Ce qui est cassé** : L'embedder Nvidia (besoin de `input_type` param) — à remplacer par HuggingFace

---

## Décisions de design prises

### Architecture (1.6% LLM / 98.4% infra)

1. **Orchestrateur = code déterministe** (PAS un LLM). Boucle async toutes les 30s qui scanne le graphe.
2. **Instances Orya = stateless**. Chaque instance reconstruit son contexte via `graphiti.search(center_node_uuid=user_node)` à chaque message.
3. **Pas de détection d'intent LLM** — le graphe détecte les matchs naturellement via search cross-group.
4. **Mémoire = 100% Graphiti/FalkorDB**. Chaque message = `add_episode()`. Recall = `search()`.
5. **Double opt-in séquentiel** avec reveal progressif (1 candidat à la fois, jamais de liste).
6. **Lien web pour les "shares"** (pas de cartes dans le terminal). Format : `orya.globalimex.online/s/{id}`
7. **Gates déterministes** avant chaque match : fraîcheur, disponibilité, historique, saturation.

### Stack technique définitive

| Composant | Techno |
|-----------|--------|
| Graph DB | FalkorDB (Redis 8 + module, dans le conteneur) |
| Knowledge Graph | Graphiti `graphiti-core[falkordb]` — module Python = `graphiti_core` |
| LLM Conversation | LangChain (`ChatGroq`, `ChatNVIDIA`, `ChatOpenAI` pour Cerebras/OR) |
| LLM Graphiti | `GroqClient` natif de graphiti_core |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (dim=384, gratuit) |
| NER locale | GLiNER2 `fastino/gliner2-multi-v1` (CPU, multilingue) |
| Orchestration | LangGraph StateGraph + LangChain |
| Observabilité | LangSmith (tracing auto avec LangGraph) |
| Gateway | Hono + Bun WebSocket |
| DB relationnelle | PostgreSQL (users, sessions, feedback, opt_ins, match_queues) |
| Persona | System prompt + few-shot dynamique + feedback loop |

### Providers LLM (tous gratuits)

| Provider | Model | API Key env var | Usage |
|----------|-------|-----------------|-------|
| Groq | `meta-llama/llama-4-scout-17b-16e-instruct` | `GROQ_API_KEY` | Conversation primary + Graphiti |
| Nvidia | `meta/llama-4-maverick-17b-128e-instruct` | `NVIDIA_API_KEY` | Graphiti extraction secondary |
| Cerebras | `llama3.1-8b` | `CEREBRAS_API_KEY` | Fallback rapide |
| OpenRouter | `meta-llama/llama-3.3-70b-instruct:free` | `OPENROUTER_API_KEY` | Dernier fallback |
| HuggingFace | `sentence-transformers/all-MiniLM-L6-v2` | `HUGGINGFACE_API_KEY` | Embeddings |

### Ports exposés

- `4001` → Gateway WebSocket (`orya.globalimex.online`)
- `3000` → FalkorDB Browser (`graph.globalimex.online`)
- `8000` → Graphiti REST API (`api.globalimex.online`)

---

## Problèmes connus à résoudre

1. **Embedder** : `nvidia/nv-embedqa-e5-v5` exige `input_type` param que l'API standard ne passe pas → **remplacer par HuggingFace**
2. **cross_encoder** : Graphiti default `OpenAIRerankerClient()` crash sans `OPENAI_API_KEY` → **workaround actuel** : `OPENAI_API_KEY=<nvidia_key>`. **Fix propre** : passer `cross_encoder=None` (ou utiliser un custom reranker)
3. **Package name** : `graphiti-core[falkordb]` installe le module `graphiti_core`. L'ancien `graphiti-core-falkordb` installe `graphiti_core_falkordb` (différent !)
4. **Python version** : Ubuntu 24.04 a Python 3.12, besoin de `--break-system-packages` pour uv

---

## Code qui fonctionne (prouvé en tests locaux)

### Graphiti init correcte (FalkorDB)
```python
from graphiti_core import Graphiti
from graphiti_core.llm_client.groq_client import GroqClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.driver.falkordb_driver import FalkorDriver

driver = FalkorDriver(host="127.0.0.1", port=6379, database="orya_memory")
llm_client = GroqClient()  # Lit GROQ_API_KEY auto

embedder = OpenAIEmbedder(config=OpenAIEmbedderConfig(
    api_key=os.getenv("HUGGINGFACE_API_KEY"),
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    embedding_dim=384,
    base_url="https://api-inference.huggingface.co/pipeline/feature-extraction",
))

graphiti = Graphiti(graph_driver=driver, llm_client=llm_client, embedder=embedder, cross_encoder=None)
await graphiti.build_indices_and_constraints()
```

### Persona Orya (style humain validé)
```
Réponses réelles obtenues en test :
- "ah cool, dev quoi ? front ou back ?"
- "bah du coup je peux te filer quelques contacts..."
- "ah merde, c'est chaud ! Tu as postulé où récemment ?"
```

### Matching cross-group (prouvé)
```
User A (plombier Lyon) parle → facts extraits
User B (cherche plombier Lyon) parle → search cross-group → MATCH trouvé → candidates envoyés
```

---

## Structure de dossiers cible (v2)

```
orya_cli/
├── orya.ts, types.ts, package.json (CLI — inchangé)
├── ARCHITECTURE.md (plan complet)
├── server/
│   ├── Dockerfile (all-in-one Ubuntu 24.04)
│   ├── supervisord.conf
│   ├── gateway/ (Hono + Bun WS)
│   ├── agent/ (LangGraph + FastAPI)
│   │   ├── graph.py (StateGraph)
│   │   ├── nodes/ (persona, extract, search, memory, intent, opt_in, notify)
│   │   ├── providers/ (llm_router, embedder)
│   │   └── persona/ (system_prompt, few_shot, negatives)
│   ├── graphiti-server/ (REST API)
│   └── db/ (PostgreSQL init.sql)
```

---

## Dockerfile qui build (testé localement)

- Base: `ubuntu:24.04`
- Stage 1: copie `falkordb.so` + `redis-server` + `redis-cli` depuis `falkordb/falkordb:latest`
- Installe: `libgomp1`, `python3`, `python3-venv`, `supervisor`, `curl`, `bun`, `uv`
- uv avec `--break-system-packages`
- supervisord gère: falkordb, postgres, graphiti-server, agent, gateway

---

## Docs lues (résumé des llms.txt)

### Graphiti (25K tokens analysés)
- `add_episode(name, episode_body, source, reference_time, group_id, saga)`
- `search(query, center_node_uuid, group_ids, num_results)`
- `search_(query, config)` avec recipes: `NODE_HYBRID_SEARCH_RRF`, `EDGE_HYBRID_SEARCH_RRF`
- `SearchFilters` : node_labels, edge_types, date ranges
- `add_episode_bulk()` pour batch
- Sagas : grouper les episodes d'une conversation
- Providers : `GroqClient`, `OpenAIGenericClient`, `GeminiClient`
- FalkorDB : `FalkorDriver(host, port, database)`
- GLiNER2 : `GLiNER2Client(llm_client, threshold, model_name)`

### LangGraph (22K tokens analysés)
- `StateGraph(State)` + `add_node` + `add_edge` + `add_conditional_edges`
- `add_messages` reducer pour chat
- `InMemorySaver` / `PostgresSaver` pour checkpoints
- `thread_id` pour multi-user
- `interrupt()` pour human-in-the-loop
- `Command(update, goto)` pour navigation
- `Send()` pour parallel map-reduce
- `create_react_agent` prebuilt
- `ToolNode` + `tools_condition`
- Streaming: `stream_mode="updates"|"values"|"custom"`
- `StreamWriter` pour custom progress

### LangChain/LangSmith (index 116K)
- `langchain_groq.ChatGroq`
- `langchain_nvidia_ai_endpoints.ChatNVIDIA`
- LangSmith : `LANGCHAIN_TRACING_V2=true` + `LANGCHAIN_API_KEY` + `LANGCHAIN_PROJECT`
- Tracing auto avec LangGraph

---

## Analyse concurrentielle

### Series.so
- iMessage, "shares" = 10 cartes swipables, anonymat par défaut
- Pas de double opt-in explicite — swipe = intérêt, conversation via Series

### Boardy
- Appel vocal AI → comprend tes besoins → email d'intro aux DEUX parties simultanément
- Opt-in implicite (répondre = accepter, ignorer = refuser)
- 20K+ contacts dans le réseau

### Orya (notre approche)
- CLI + lien web pour swiper
- Double opt-in séquentiel (1 candidat à la fois)
- Orchestrateur déterministe + gates
- Graphe comme mémoire (pas de profils statiques)

---

## Commande pour la prochaine session

```
/apex -asi Implémenter la v2 complète sur branche v2.
Lire ARCHITECTURE.md et SESSION_CONTEXT.md pour le plan.
Merger PR #6 d'abord. Créer branche v2 depuis main.
Implémenter dans l'ordre : PostgreSQL → Graphiti proper → LangGraph agent → Orchestrateur → Gateway update → Tests E2E.
```

---

## Clés API (à passer en env, JAMAIS dans le code)

```
GROQ_API_KEY=gsk_xxx
NVIDIA_API_KEY=nvapi-xxx
CEREBRAS_API_KEY=csk-xxx
OPENROUTER_API_KEY=sk-or-v1-xxx
HUGGINGFACE_API_KEY=hf_xxx
OPENAI_API_KEY=<same as NVIDIA_API_KEY>  # trick pour Graphiti cross_encoder
LANGCHAIN_API_KEY=<à créer sur smith.langchain.com>
POSTGRES_PASSWORD=orya_secret_2026
```
