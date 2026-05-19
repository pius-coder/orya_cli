"""Build the Graphiti instance for the REST server."""

from __future__ import annotations

import logging
import os

from graphiti_core import Graphiti
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.groq_client import GroqClient

from settings import get_settings

logger = logging.getLogger(__name__)

_HF_BASE_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction"
_HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_HF_DIM = 384


async def init_graphiti() -> Graphiti:
    s = get_settings()
    os.environ.setdefault("SEMAPHORE_LIMIT", str(s.SEMAPHORE_LIMIT))
    os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", s.GRAPHITI_TELEMETRY_ENABLED)

    driver = FalkorDriver(host=s.FALKORDB_HOST, port=s.FALKORDB_PORT)
    llm = GroqClient(config=LLMConfig(
        api_key=s.GROQ_API_KEY or "missing",
        model=s.GRAPHITI_LLM_MODEL,
        small_model=s.GRAPHITI_LLM_SMALL_MODEL,
    ))
    embedder = OpenAIEmbedder(config=OpenAIEmbedderConfig(
        api_key=s.HUGGINGFACE_API_KEY or "missing",
        embedding_model=_HF_MODEL,
        embedding_dim=_HF_DIM,
        base_url=_HF_BASE_URL,
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
