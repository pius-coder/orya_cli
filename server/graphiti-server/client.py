"""Build the Graphiti instance for the REST server."""

from __future__ import annotations

import logging
import os
import sys

from graphiti_core import Graphiti
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.groq_client import GroqClient

from settings import get_settings

# Import the HF embedder directly (avoid triggering agent package __init__)
sys.path.insert(0, "/app/agent/providers")
from hf_embedder import HuggingFaceEmbedder, HuggingFaceEmbedderConfig

logger = logging.getLogger(__name__)


async def init_graphiti() -> Graphiti:
    s = get_settings()
    os.environ.setdefault("SEMAPHORE_LIMIT", str(s.SEMAPHORE_LIMIT))
    os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", s.GRAPHITI_TELEMETRY_ENABLED)

    driver = Neo4jDriver(
        uri=s.NEO4J_URI,
        user=s.NEO4J_USER,
        password=s.NEO4J_PASSWORD,
    )
    llm = GroqClient(config=LLMConfig(
        api_key=s.GROQ_API_KEY or "missing",
        model=s.GRAPHITI_LLM_MODEL,
        small_model=s.GRAPHITI_LLM_SMALL_MODEL,
        max_tokens=8192,  # Groq limit for Llama 4 Scout
    ))

    embedder = HuggingFaceEmbedder(config=HuggingFaceEmbedderConfig(
        model_name="ibm-granite/granite-embedding-97m-multilingual-r2",
        embedding_dim=384,
        api_key=s.HUGGINGFACE_API_KEY or "missing",
    ))

    g = Graphiti(
        graph_driver=driver,
        llm_client=llm,
        embedder=embedder,
        cross_encoder=None,
    )
    await g.build_indices_and_constraints()
    logger.info("graphiti-server: Graphiti initialized.")
    return g
