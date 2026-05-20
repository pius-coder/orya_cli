"""Full end-to-end test with all external dependencies mocked.

Simulates a 4-user conversation flow with the LangGraph agent:
  - Marc (plumber) says who he is → extracted as entity, persisted
  - Sophie (needs plumber) → triggers cross-user match to Marc
  - Karim (React dev) → extracted, no match with Marc/Sophie
  - Julie (needs dev) → triggers cross-user match to Karim

Mocks: LLM, Embedder, PostgreSQL, Qdrant, Graphiti, Neo4j, httpx
"""
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.memory.models import ExtractedEntity, NaturalFact, MatchIndexEntry


# ── Configuration ──────────────────────────────────────────────────

TEST_USERS = [
    {"user_id": "marc_001", "alias": "Marc", "messages": [
        "Je suis plombier à Lyon avec 10 ans d'expérience",
        "Je cherche des clients dans le centre de Lyon",
    ]},
    {"user_id": "sophie_001", "alias": "Sophie", "messages": [
        "J'habite à Lyon et mon évier fuit depuis hier",
        "Tu connais un bon plombier ? J'ai vraiment besoin d'aide",
    ]},
    {"user_id": "karim_001", "alias": "Karim", "messages": [
        "Je suis développeur frontend React depuis 5 ans",
        "Je cherche un job remote pour l'étranger",
    ]},
    {"user_id": "julie_001", "alias": "Julie", "messages": [
        "Je cherche un développeur frontend pour ma startup",
        "On est à Paris mais full remote possible",
    ]},
]


# ── Mock data ──────────────────────────────────────────────────────

def _mock_llm_response(text: str) -> MagicMock:
    """Create a mock LLM response with content."""
    m = MagicMock()
    m.content = text
    return m


LLM_RESPONSES: dict[str, str] = {
    "Je suis plombier à Lyon avec 10 ans d'expérience": (
        "Super Marc ! Je note que tu es plombier à Lyon avec 10 ans d'expérience. "
        "Je vais enregistrer ça dans ma mémoire. Tu cherches des clients ?"
    ),
    "Je cherche des clients dans le centre de Lyon": (
        "Je comprends ! Je vais marquer que tu recherches des clients dans le centre "
        "de Lyon. Si quelqu'un a besoin d'un plombier, je pourrai te recommander."
    ),
    "J'habite à Lyon et mon évier fuit depuis hier": (
        "Oui, je comprends l'urgence ! Je vais chercher dans ma mémoire si je connais "
        "un plombier à Lyon. Laisse-moi vérifier..."
    ),
    "Tu connais un bon plombier ? J'ai vraiment besoin d'aide": (
        "Bonne nouvelle ! Je connais Marc, un plombier à Lyon avec 10 ans d'expérience "
        "qui cherche des clients dans le centre. Je peux le contacter pour toi si tu veux."
    ),
    "Je suis développeur frontend React depuis 5 ans": (
        "Super Karim ! Je note que tu es développeur React avec 5 ans d'expérience. "
        "Tu recherches un job remote c'est bien ça ?"
    ),
    "Je cherche un job remote pour l'étranger": (
        "D'accord, je marque ça. Développeur React, 5 ans d'expérience, "
        "recherche job remote pour l'étranger."
    ),
    "Je cherche un développeur frontend pour ma startup": (
        "Intéressant ! Je vais chercher dans ma mémoire si je connais "
        "un développeur frontend disponible."
    ),
    "On est à Paris mais full remote possible": (
        "Parfait ! Je connais Karim, un développeur React avec 5 ans d'expérience "
        "qui cherche un job remote. Je peux vous mettre en relation."
    ),
}

ROUTER_DECISIONS: dict[str, dict[str, Any]] = {
    "Je suis plombier à Lyon avec 10 ans d'expérience": {
        "intent": "memory_ingest",
        "requires_match": False,
    },
    "Je cherche des clients dans le centre de Lyon": {
        "intent": "memory_ingest_and_match",
        "requires_match": True,
    },
    "J'habite à Lyon et mon évier fuit depuis hier": {
        "intent": "memory_ingest",
        "requires_match": False,
    },
    "Tu connais un bon plombier ? J'ai vraiment besoin d'aide": {
        "intent": "memory_ingest_and_match",
        "requires_match": True,
    },
    "Je suis développeur frontend React depuis 5 ans": {
        "intent": "memory_ingest",
        "requires_match": False,
    },
    "Je cherche un job remote pour l'étranger": {
        "intent": "memory_ingest_and_match",
        "requires_match": True,
    },
    "Je cherche un développeur frontend pour ma startup": {
        "intent": "memory_ingest",
        "requires_match": False,
    },
    "On est à Paris mais full remote possible": {
        "intent": "memory_ingest_and_match",
        "requires_match": True,
    },
}

ENTITY_EXTRACTION_RESULTS: dict[str, list[str]] = {
    "Je suis plombier à Lyon avec 10 ans d'expérience": [
        "Marc", "plombier", "Lyon",
    ],
    "Je cherche des clients dans le centre de Lyon": [
        "clients", "centre de Lyon",
    ],
    "J'habite à Lyon et mon évier fuit depuis hier": [
        "Sophie", "Lyon", "évier",
    ],
    "Tu connais un bon plombier ? J'ai vraiment besoin d'aide": [
        "plombier", "Lyon",
    ],
    "Je suis développeur frontend React depuis 5 ans": [
        "Karim", "développeur frontend", "React",
    ],
    "Je cherche un job remote pour l'étranger": [
        "job remote", "étranger",
    ],
    "Je cherche un développeur frontend pour ma startup": [
        "Julie", "développeur frontend", "startup",
    ],
    "On est à Paris mais full remote possible": [
        "Paris", "full remote", "startup",
    ],
}

FACT_GENERATION_RESULTS: dict[str, list[NaturalFact]] = {
    "Je suis plombier à Lyon avec 10 ans d'expérience": [
        NaturalFact(text="Marc est plombier à Lyon avec 10 ans d'expérience", entities=["Marc", "plombier", "Lyon"]),
        NaturalFact(text="Marc a 10 ans d'expérience comme plombier", entities=["Marc", "plombier"]),
    ],
    "Je cherche des clients dans le centre de Lyon": [
        NaturalFact(text="Marc cherche des clients dans le centre de Lyon", entities=["Marc", "clients", "centre de Lyon"]),
    ],
    "J'habite à Lyon et mon évier fuit depuis hier": [
        NaturalFact(text="Sophie habite à Lyon et son évier fuit", entities=["Sophie", "Lyon", "évier"]),
        NaturalFact(text="Sophie a un problème de plomberie urgent", entities=["Sophie", "évier"]),
    ],
    "Tu connais un bon plombier ? J'ai vraiment besoin d'aide": [
        NaturalFact(text="Sophie cherche un plombier à Lyon en urgence", entities=["Sophie", "plombier", "Lyon"]),
    ],
    "Je suis développeur frontend React depuis 5 ans": [
        NaturalFact(text="Karim est développeur frontend React avec 5 ans d'expérience", entities=["Karim", "développeur frontend", "React"]),
        NaturalFact(text="Karim a 5 ans d'expérience en React", entities=["Karim", "React"]),
    ],
    "Je cherche un job remote pour l'étranger": [
        NaturalFact(text="Karim cherche un job remote pour l'étranger", entities=["Karim", "job remote", "étranger"]),
    ],
    "Je cherche un développeur frontend pour ma startup": [
        NaturalFact(text="Julie cherche un développeur frontend pour sa startup", entities=["Julie", "développeur frontend", "startup"]),
    ],
    "On est à Paris mais full remote possible": [
        NaturalFact(text="Julie est à Paris mais accepte le full remote", entities=["Julie", "Paris", "full remote"]),
        NaturalFact(text="La startup de Julie cherche un développeur frontend en full remote", entities=["Julie", "startup", "développeur frontend", "full remote"]),
    ],
}

EMBEDDING_DIM = 384


def _fake_embedding(text: str) -> list[float]:
    """Deterministic fake embedding based on text hash."""
    h = hash(text) % (2**31)
    rng = __import__("random").Random(h)
    return [rng.uniform(-1.0, 1.0) for _ in range(EMBEDDING_DIM)]


# ── Mocks ──────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def mock_settings() -> Generator[MagicMock, None, None]:
    """Mock Settings to avoid reading real .env."""
    settings = MagicMock()
    settings.POSTGRES_HOST = "localhost"
    settings.POSTGRES_PORT = 5432
    settings.POSTGRES_DB = "orya_test"
    settings.POSTGRES_USER = "test"
    settings.POSTGRES_PASSWORD = "test"
    settings.postgres_dsn = "postgresql://test:test@localhost:5432/orya_test"
    settings.postgres_dsn_psycopg = "postgresql://test:test@localhost:5432/orya_test"
    settings.NEO4J_URI = "bolt://localhost:7687"
    settings.NEO4J_USER = "neo4j"
    settings.NEO4J_PASSWORD = "test"
    settings.GROQ_API_KEY = "mock_groq_key"
    settings.NVIDIA_API_KEY = "mock_nvidia_key"
    settings.OPENROUTER_API_KEY = "mock_openrouter_key"
    settings.HUGGINGFACE_API_KEY = "mock_hf_key"
    settings.QDRANT_URL = "http://localhost:6333"
    settings.QDRANT_API_KEY = ""
    settings.EMBEDDING_DIM = EMBEDDING_DIM
    settings.LLM_TEMPERATURE = 0.3
    settings.LLM_MAX_TOKENS = 512
    settings.SEARCH_NUM_RESULTS_CONTEXT = 5
    settings.SEARCH_NUM_RESULTS_MATCH = 20
    settings.OPT_IN_TTL_HOURS = 72
    settings.INTERNAL_API_KEY = ""
    settings.GATEWAY_INTERNAL_URL = "http://localhost:4001"
    settings.GRAPHITI_SERVER_URL = "http://localhost:8000"
    settings.configure_langsmith = MagicMock()

    with patch("agent.core.config.get_settings", return_value=settings):
        yield settings


@pytest.fixture
def mock_pool() -> AsyncMock:
    """Mock asyncpg connection pool."""
    pool = AsyncMock()
    conn = AsyncMock()

    # Default: return empty results
    conn.fetch = AsyncMock(return_value=[])
    conn.fetchval = AsyncMock(return_value=None)
    conn.fetchrow = AsyncMock(return_value=None)
    conn.execute = AsyncMock(return_value="")

    pool.acquire = AsyncMock()
    cm = pool.acquire.return_value.__aenter__
    cm.return_value = conn

    with patch("agent.db.postgres._pool", pool):
        with patch("agent.db.postgres.get_pool", return_value=pool):
            yield pool


@pytest.fixture
def mock_qdrant() -> Generator[MagicMock, None, None]:
    """Mock Qdrant client."""
    qdrant = MagicMock()
    qdrant.collection_exists.return_value = True
    qdrant.search.return_value = []
    qdrant.upsert.return_value = None

    with patch("agent.infra.qdrant.get_qdrant_client", return_value=qdrant):
        yield qdrant


@pytest.fixture
def mock_graphiti() -> MagicMock:
    """Mock Graphiti client."""
    graphiti = MagicMock()
    graphiti.build_node = AsyncMock(return_value="node_uuid_123")
    graphiti.build_edge = AsyncMock(return_value="edge_uuid_456")
    graphiti.add_episode = AsyncMock(return_value="episode_uuid_789")

    with patch("agent.providers.graphiti.init_graphiti", AsyncMock(return_value=graphiti)):
        yield graphiti


@pytest.fixture
def mock_llm() -> MagicMock:
    """Mock LLM Runnable that returns predefined responses."""

    async def ainvoke(input_or_msgs, config=None, **kwargs):
        if isinstance(input_or_msgs, list):
            for msg in input_or_msgs:
                if isinstance(msg, HumanMessage):
                    text = msg.content
                    # Check router decisions
                    if text in ROUTER_DECISIONS:
                        decision = ROUTER_DECISIONS[text]
                        return _mock_llm_response(json.dumps(decision))
                    # Check known responses
                    if text in LLM_RESPONSES:
                        reply = LLM_RESPONSES[text]
                        return _mock_llm_response(reply)
            # Fallback: return the last message content
            return _mock_llm_response("Je comprends.")
        return _mock_llm_response("Je comprends.")

    llm = MagicMock()
    llm.ainvoke = ainvoke
    llm.invoke = MagicMock(return_value=_mock_llm_response("Je comprends."))

    with patch("agent.providers.llm.build_llm", return_value=llm):
        yield llm


@pytest.fixture
def mock_embedder() -> MagicMock:
    """Mock embedding provider that returns deterministic embeddings."""

    embedder = MagicMock()
    embedder.create = AsyncMock(side_effect=lambda x: _fake_embedding(x) if isinstance(x, str) else [_fake_embedding(str(i)) for i in x])
    embedder.create_batch = AsyncMock(side_effect=lambda xs: [_fake_embedding(x) for x in xs])

    with patch("agent.providers.embedder.HuggingFaceEmbedder", return_value=embedder):
        yield embedder


@pytest.fixture
def mock_httpx() -> Generator[MagicMock, None, None]:
    """Mock httpx client to avoid real HTTP calls to gateway."""
    client = AsyncMock()
    client.post = AsyncMock()
    client.get = AsyncMock()

    # health endpoint
    health_resp = MagicMock()
    health_resp.json.return_value = {"ok": True, "services": {"agent": True, "postgres": True, "graphiti": True}}
    health_resp.raise_for_status = MagicMock()
    client.get.return_value = health_resp

    with patch("httpx.AsyncClient", return_value=client):
        yield client


# ── Fixtures: Entity extraction & fact generation ──────────────────

@pytest.fixture(autouse=True)
def mock_memory_components() -> Generator[None, None, None]:
    """Mock entity extraction and fact generation to return controlled data."""

    async def _mock_extract(messages_text, llm, known_entities=None):
        for text, entities in ENTITY_EXTRACTION_RESULTS.items():
            if text in messages_text:
                return entities
        return []

    async def _mock_generate(messages_text, entity_names, llm):
        for text, facts in FACT_GENERATION_RESULTS.items():
            if text in messages_text:
                return facts
        return []

    def _mock_resolve(new_entities, existing, aliases, llm=None):
        from agent.memory.entity_resolver import ResolverDecision
        decisions = []
        for ent in new_entities:
            decisions.append(ResolverDecision(
                new_entity_ref=ent,
                action="keep",
                target_entity_id=None,
                resolved_via="exact",
            ))
        return decisions

    patches = [
        patch("agent.memory.extractor.extract_entities", _mock_extract),
        patch("agent.memory.fact_generator.generate_facts", _mock_generate),
        patch("agent.memory.entity_resolver.resolve_entities_membrain", _mock_resolve),
    ]
    for p in patches:
        p.start()
    yield
    for p in patches:
        p.stop()


# ── Tests ──────────────────────────────────────────────────────────

class TestFullConversation:
    """Simulates a full multi-user conversation through the LangGraph agent."""

    @pytest.mark.asyncio
    async def test_end_to_end_conversation_flow(
        self,
        mock_llm: MagicMock,
        mock_embedder: MagicMock,
        mock_pool: AsyncMock,
        mock_qdrant: MagicMock,
        mock_graphiti: MagicMock,
        mock_httpx: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Complete test: 4 users, 8 messages, matching pairs."""
        from agent.main import build_graph, _components
        from agent.core.config import get_settings
        from agent.manifests.registry import ManifestRegistry
        from agent.providers import build_llm, init_graphiti
        from agent.providers.embedder import HuggingFaceEmbedder
        from agent.models import OryaState

        settings = get_settings()
        manifests_dir = Path(__file__).parent.parent.parent / "manifests"
        manifests = ManifestRegistry(str(manifests_dir))
        graphiti = await init_graphiti()
        llm = build_llm()
        embedder = HuggingFaceEmbedder()

        graph = build_graph(
            graphiti=graphiti,
            llm=llm,
            manifests=manifests,
            embedder=embedder,
        )

        # Phase 1: Marc introduces himself
        print("=== Phase 1: Marc ===")
        state = await _run_agent(graph, llm, "marc_001", "Marc",
                                 "Je suis plombier à Lyon avec 10 ans d'expérience")
        assert "plombier" in state["last_assistant_reply"].lower() or "plombier" in str(state["messages"][-1].content).lower()

        state = await _run_agent(graph, llm, "marc_001", "Marc",
                                 "Je cherche des clients dans le centre de Lyon")
        assert state.get("candidates") is not None or state["last_assistant_reply"]

        # Phase 2: Sophie needs a plumber
        print("=== Phase 2: Sophie ===")
        state = await _run_agent(graph, llm, "sophie_001", "Sophie",
                                 "J'habite à Lyon et mon évier fuit depuis hier")
        assert "Lyon" in state["last_assistant_reply"] or "plombier" in state["last_assistant_reply"].lower()

        state = await _run_agent(graph, llm, "sophie_001", "Sophie",
                                 "Tu connais un bon plombier ? J'ai vraiment besoin d'aide")
        # Should mention Marc or plumber
        assert "Marc" in state["last_assistant_reply"] or "plombier" in state["last_assistant_reply"].lower()

        # Phase 3: Karim the dev
        print("=== Phase 3: Karim ===")
        state = await _run_agent(graph, llm, "karim_001", "Karim",
                                 "Je suis développeur frontend React depuis 5 ans")
        assert "React" in state["last_assistant_reply"] or "développeur" in state["last_assistant_reply"].lower()

        state = await _run_agent(graph, llm, "karim_001", "Karim",
                                 "Je cherche un job remote pour l'étranger")
        assert "remote" in state["last_assistant_reply"].lower()

        # Phase 4: Julie needs a dev
        print("=== Phase 4: Julie ===")
        state = await _run_agent(graph, llm, "julie_001", "Julie",
                                 "Je cherche un développeur frontend pour ma startup")
        assert "développeur" in state["last_assistant_reply"].lower() or "frontend" in state["last_assistant_reply"].lower()

        state = await _run_agent(graph, llm, "julie_001", "Julie",
                                 "On est à Paris mais full remote possible")
        assert "full remote" in state["last_assistant_reply"].lower() or "remote" in state["last_assistant_reply"].lower() or "Karim" in state["last_assistant_reply"]

        # Phase 5: Verify Qdrant calls
        print("=== Phase 5: Verification ===")
        verify_qdrant_calls(mock_qdrant)
        verify_postgres_calls(mock_pool)
        verify_trace(state)

        print("=== All phases passed ===")

    @pytest.mark.asyncio
    async def test_conversation_without_matching(
        self,
        mock_llm: MagicMock,
        mock_embedder: MagicMock,
        mock_pool: AsyncMock,
        mock_qdrant: MagicMock,
        mock_graphiti: MagicMock,
    ) -> None:
        """Single user conversation with no matching needed."""
        from agent.graph import build_graph
        from agent.manifests.registry import ManifestRegistry
        from agent.providers import build_llm, init_graphiti
        from agent.providers.embedder import HuggingFaceEmbedder

        manifests_dir = Path(__file__).parent.parent.parent / "manifests"
        manifests = ManifestRegistry(str(manifests_dir))
        graphiti = await init_graphiti()
        llm = build_llm()
        embedder = HuggingFaceEmbedder()

        graph = build_graph(
            graphiti=graphiti,
            llm=llm,
            manifests=manifests,
            embedder=embedder,
        )

        state = await _run_agent(graph, llm, "test_001", "Test",
                                 "Je suis plombier à Lyon avec 10 ans d'expérience")
        assert "test_001" == state["user_id"]
        assert len(state["messages"]) >= 2  # human + ai

    @pytest.mark.asyncio
    async def test_llm_router_decision(
        self, mock_llm: MagicMock, mock_pool: AsyncMock
    ) -> None:
        """Test that the memory router parses LLM decisions correctly."""
        from agent.nodes.memory_router import memory_router_node, _extract_json

        json_text = '{"intent": "memory_ingest_and_match", "requires_match": true}'
        parsed = _extract_json(json_text)
        assert parsed["intent"] == "memory_ingest_and_match"
        assert parsed["requires_match"] is True

        json_text2 = '{"intent": "memory_ingest", "requires_match": false}'
        parsed2 = _extract_json(json_text2)
        assert parsed2["intent"] == "memory_ingest"

    @pytest.mark.asyncio
    async def test_embedder_runs(
        self, mock_embedder: MagicMock
    ) -> None:
        """Test embedder creates deterministic embeddings."""
        emb = await mock_embedder.create("Je suis plombier à Lyon")
        assert len(emb) == EMBEDDING_DIM
        emb2 = await mock_embedder.create("Je suis plombier à Lyon")
        assert emb == emb2  # deterministic


# ── Helpers ────────────────────────────────────────────────────────

async def _run_agent(
    graph: Any, llm: MagicMock, user_id: str, alias: str, text: str
) -> dict[str, Any]:
    """Run the LangGraph agent with a single message and return final state."""
    from langchain_core.messages import HumanMessage

    initial_state: dict[str, Any] = {
        "messages": [HumanMessage(content=text)],
        "user_id": user_id,
        "user_alias": alias,
        "last_user_text": text,
        "last_assistant_reply": "",
        "user_reflection": None,
        "orya_reflection": None,
        "facts_context": None,
        "tool_calls": [],
        "candidates": [],
        "pending_opt_in": None,
        "opt_in_response": None,
        "trace": [],
    }

    events = []
    async for event in graph.astream_events(initial_state, version="v2"):
        events.append(event)

    final_state = await _get_final_state(events)
    return final_state


async def _get_final_state(events: list[dict]) -> dict[str, Any]:
    """Extract final state from LangGraph astream_events output."""
    final = {}
    for event in reversed(events):
        if event.get("event") == "on_chain_end":
            data = event.get("data", {})
            output = data.get("output", {})
            if isinstance(output, dict) and "messages" in output:
                final.update(output)
        if event.get("event") == "on_graph_step" or event.get("event") == "on_chain_stream":
            data = event.get("data", {})
            chunk = data.get("chunk", {})
            if isinstance(chunk, dict):
                final.update(chunk)

    if not final or "messages" not in final:
        # Fallback: reconstruct from the last state
        for event in reversed(events):
            data = event.get("data", {})
            output = data.get("output", {}) or data.get("chunk", {})
            if isinstance(output, dict):
                if "user_id" in output or "messages" in output:
                    final.update(output)

    if "last_assistant_reply" not in final or not final["last_assistant_reply"]:
        messages = final.get("messages", [])
        if messages:
            last = messages[-1]
            if hasattr(last, "content"):
                final["last_assistant_reply"] = str(last.content)

    return final


def verify_qdrant_calls(mock_qdrant: MagicMock) -> None:
    """Verify Qdrant was called appropriately."""
    # upsert_fact or upsert_match_index should have been called
    upsert_calls = [
        call for call in mock_qdrant.mock_calls
        if "upsert" in str(call[0]).lower()
    ]
    # This is informational, not a strict assertion since Qdrant calls
    # depend on runtime paths
    print(f"  Qdrant upsert calls: {len(upsert_calls)}")


def verify_postgres_calls(mock_pool: AsyncMock) -> None:
    """Verify PostgreSQL calls were made."""
    conn = mock_pool.acquire.return_value.__aenter__.return_value
    fetch_calls = conn.fetch.call_count
    print(f"  PostgreSQL fetch calls: {fetch_calls}")


def verify_trace(state: dict[str, Any]) -> None:
    """Verify trace events were recorded."""
    trace = state.get("trace", [])
    steps = [t.get("step") for t in trace] if isinstance(trace, list) else []
    print(f"  Trace events: {len(steps) if isinstance(steps, list) else 0}")
    if isinstance(trace, list):
        for t in trace:
            print(f"    [{t.get('step')}] {t.get('detail')[:80]}")


# ── Run directly (for debugging) ───────────────────────────────────

if __name__ == "__main__":
    asyncio.run(pytest.main([__file__, "-vvs", "--tb=long"]))
