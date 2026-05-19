"""Graphiti client builder.

Single source of truth for how the Graphiti instance is constructed so that
both the agent and the graphiti-server use exactly the same config.
"""

from __future__ import annotations

import logging
import os

from graphiti_core import Graphiti
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.groq_client import GroqClient
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

from ..settings import get_settings
from .embedder import build_embedder

logger = logging.getLogger(__name__)


async def init_graphiti() -> Graphiti:
    """Build and initialize a Graphiti instance bound to Neo4j.

    - Uses Groq via the native `GroqClient` for entity/edge extraction.
    - Uses HuggingFace via the OpenAI-compatible `OpenAIEmbedder` for
      embeddings (avoids OpenAI key requirement).
    - Disables `cross_encoder` to avoid the default `OpenAIRerankerClient`,
      which would crash without an OpenAI key. Reciprocal Rank Fusion (RRF)
      ranking still works without it.
    """

    s = get_settings()

    # Apply Graphiti env tuning (must be set before Graphiti is built so the
    # internal semaphore and telemetry honour them).
    os.environ.setdefault("SEMAPHORE_LIMIT", str(s.SEMAPHORE_LIMIT))
    os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", s.GRAPHITI_TELEMETRY_ENABLED)

    driver = Neo4jDriver(
        uri=s.NEO4J_URI,
        user=s.NEO4J_USER,
        password=s.NEO4J_PASSWORD,
    )

    provider = os.getenv("GRAPHITI_LLM_PROVIDER", "openrouter").lower()

    if provider == "openrouter":
        logger.info("Initializing Graphiti with OpenRouter OpenAIGenericClient...")
        llm_client = OpenAIGenericClient(
            config=LLMConfig(
                api_key=s.OPENROUTER_API_KEY or "missing",
                base_url="https://openrouter.ai/api/v1",
                model=s.GRAPHITI_LLM_MODEL,
            )
        )
    elif provider == "nvidia":
        logger.info("Initializing Graphiti with NVIDIA OpenAIGenericClient...")
        llm_client = OpenAIGenericClient(
            config=LLMConfig(
                api_key=s.NVIDIA_API_KEY or "missing",
                base_url="https://integrate.api.nvidia.com/v1",
                model=s.GRAPHITI_LLM_MODEL,
            )
        )
    else:
        logger.info("Initializing Graphiti with native GroqClient...")
        llm_client = GroqClient(
            config=LLMConfig(
                api_key=s.GROQ_API_KEY or "missing",
                model=s.GRAPHITI_LLM_MODEL,
                small_model=s.GRAPHITI_LLM_SMALL_MODEL,
            )
        )

    embedder = build_embedder()

    graphiti = Graphiti(
        graph_driver=driver,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=None,  # See module docstring above
    )

    # Idempotent — only creates indices if missing.
    try:
        await graphiti.build_indices_and_constraints()
        logger.info("Graphiti indices ready (Neo4j %s).",
                    s.NEO4J_URI)
    except Exception:  # pragma: no cover — boot-time best-effort
        logger.exception("Failed to build Graphiti indices.")
        raise

    return graphiti
