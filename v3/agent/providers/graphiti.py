"""Graphiti initialization with proper provider wiring.

Fixes v2 issues:
- No sys.path hacks
- No contradictory docstrings
- cross_encoder=None is documented
- Environment side-effects removed
"""
import logging

from graphiti_core import Graphiti
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.groq_client import GroqClient
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

from ..core.config import get_settings
from .embedder import HuggingFaceEmbedder, HuggingFaceEmbedderConfig

logger = logging.getLogger(__name__)


async def init_graphiti() -> Graphiti:
    """Build and return a configured Graphiti instance.

    Uses Neo4j as the graph driver and HuggingFace for embeddings.
    The cross_encoder is explicitly disabled (None) to avoid the
    OpenAI reranker dependency.
    """
    settings = get_settings()

    driver = Neo4jDriver(
        uri=settings.NEO4J_URI,
        user=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD,
    )

    # LLM client for Graphiti entity extraction
    provider = settings.GRAPHITI_LLM_PROVIDER.lower()
    if provider == "groq" and settings.GROQ_API_KEY:
        llm_client = GroqClient(
            config=LLMConfig(
                api_key=settings.GROQ_API_KEY,
                model=settings.GRAPHITI_LLM_SMALL_MODEL,
            )
        )
    elif provider == "nvidia" and settings.NVIDIA_API_KEY:
        llm_client = OpenAIGenericClient(
            config=LLMConfig(
                api_key=settings.NVIDIA_API_KEY,
                model=settings.GRAPHITI_LLM_MODEL,
                base_url="https://integrate.api.nvidia.com/v1",
            )
        )
    elif provider == "openrouter" and settings.OPENROUTER_API_KEY:
        llm_client = OpenAIGenericClient(
            config=LLMConfig(
                api_key=settings.OPENROUTER_API_KEY,
                model=settings.GRAPHITI_LLM_MODEL,
                base_url="https://openrouter.ai/api/v1",
            )
        )
    else:
        # Fallback to Groq if available, else error
        if settings.GROQ_API_KEY:
            llm_client = GroqClient(
                config=LLMConfig(
                    api_key=settings.GROQ_API_KEY,
                    model=settings.GRAPHITI_LLM_SMALL_MODEL,
                )
            )
        else:
            raise RuntimeError(
                f"Graphiti LLM provider '{provider}' not available. "
                "Set GROQ_API_KEY, NVIDIA_API_KEY, or OPENROUTER_API_KEY."
            )

    embedder = HuggingFaceEmbedder(
        config=HuggingFaceEmbedderConfig(
            api_key=settings.HUGGINGFACE_API_KEY,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            embedding_dim=384,
        )
    )

    graphiti = Graphiti(
        graph_driver=driver,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=None,  # Disabled: avoids OpenAI reranker dependency
    )
    await graphiti.build_indices_and_constraints()
    logger.info("Graphiti initialized with provider=%s", provider)
    return graphiti
