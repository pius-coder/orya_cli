# Orya v3 — Architecture Hybride MemBrain + Graphiti

## Vision

Chaque utilisateur possède un **Personal Knowledge Graph (PKG)** structuré comme MemBrain :
- Faits en langage naturel avec entités entre crochets `[Caroline]`
- **Entity Tree** hiérarchique par entité (root → aspect → leaf)
- **Session summaries** épisodiques
- **Temporal annotations** `[raw::resolved]`
- **Résolution d'entités** robuste (3 couches)

Un **Match Graph Global** indexe les entités et faits de TOUS les utilisateurs pour permettre le matching cross-user.

Quand un utilisateur discute avec Orya :
1. **Track chaud** : Réponse conversationnelle immédiate (style SMS)
2. **Track froid** : Ingestion MemBrain dans son PKG + mise à jour de l'Entity Tree
3. **Track matching** : Si l'utilisateur exprime un besoin, query automatique de tous les PKGs

## Couches Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AGENT ORYA (FastAPI)                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Memory      │  │ Matching    │  │ Conversation        │  │
│  │ Router      │  │ Engine      │  │ Engine (ReAct)      │  │
│  └──────┬──────┘  └──────┬──────┘  └─────────────────────┘  │
└─────────┼────────────────┼──────────────────────────────────┘
          │                │
          ▼                ▼
┌─────────────────────────────────────────────────────────────┐
│              MEMORY LAYER (MemBrain Patterns)                │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Personal Knowledge Graph (PKG) — par utilisateur       │ │
│  │  ├── PostgreSQL (faits, entités, arbres, sessions)     │ │
│  │  ├── Entity Tree (root→aspect→leaf)                    │ │
│  │  ├── Fact natural language with [Entity] refs          │ │
│  │  └── Session summaries (épisodique)                    │ │
│  └─────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Match Graph Global (cross-user)                        │ │
│  │  ├── Index global des entités/faits de tous les users   │ │
│  │  ├── Cross-user retrieval (6 chemins)                   │ │
│  │  └── Rerank + sequential reveal (1 candidat)           │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│              GRAPHITI + NEO4J (Fallback/Bridge)              │
│  ├── Temporal knowledge graph (épisodes, entités, edges)     │
│  ├── Cross-group search natif                                │
│  └── Bridge vers PKG pour migration/rétrocompatibilité       │
└─────────────────────────────────────────────────────────────┘
```

## Modules

### `agent/memory/` — Personal Knowledge Graph
- `extractor.py` : Extraction d'entités (2 passes LLM)
- `fact_generator.py` : Génération de faits naturels avec validation
- `entity_resolver.py` : Résolution 3 couches (exact, LSH, LLM)
- `entity_tree.py` : Arbre hiérarchique + routing + audit + propagation
- `session_summarizer.py` : Résumés épisodiques
- `ingest_pipeline.py` : Orchestration ingestion complète
- `models.py` : Modèles SQLAlchemy/Pydantic

### `agent/matching/` — Cross-User Matching
- `cross_user_retrieval.py` : Query tous les PKGs
- `matcher.py` : Reranking + scoring + proposition séquentielle
- `candidate_ranker.py` : Algorithme de ranking multi-facteur

### `agent/nodes/` — Nodes LangGraph mis à jour
- `memory_router.py` : Décide fast_think / deep_think / match
- `ingest_memory.py` : Ingestion MemBrain-style (fire-and-forget)
- `retrieve_context.py` : Retrieval depuis PKG (pas Graphiti seul)
- `cross_user_match.py` : Déclenche le matching cross-user

## Flux de données

### Ingestion (passive)
```
Message user → Extract entities (2 passes) → Generate facts → 
Resolve entities (3 layers) → Persist to PKG → Update Entity Trees
```

### Matching (quand l'utilisateur cherche)
```
Message user → Memory Router (detect "search intent") →
Cross-User Retrieval (6 paths sur tous les PKGs) →
Rerank candidates → Sequential reveal (1 candidat) →
Double opt-in state machine
```

### Conversation normale
```
Message user → Memory Router (fast_think) →
Retrieve from own PKG (facts context) →
LLM response with persona
```

## Termes techniques

- **PKG** = Personal Knowledge Graph (graphe personnel d'un utilisateur)
- **Match Graph** = Graphe global agrégant les PKGs pour le matching
- **Entity Tree** = Arbre hiérarchique sémantique par entité (MemBrain)
- **Sequential Reveal** = Révélation d'1 candidat à la fois (pas de liste)
- **Federated Entity Graph** = Architecture de graphes fédérés (1 par user)
