"""HuggingFace serverless embedder for Graphiti.

Uses `OpenAIEmbedder` from graphiti-core configured against HuggingFace's
OpenAI-compatible feature-extraction endpoint. The model
`sentence-transformers/all-MiniLM-L6-v2` returns 384-dim vectors and is free
to use with a HuggingFace token.
"""

from __future__ import annotations

import logging

from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

from ..settings import get_settings

logger = logging.getLogger(__name__)

_HF_BASE_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction"
_HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_HF_DIM = 384


def build_embedder() -> OpenAIEmbedder:
    s = get_settings()
    if not s.HUGGINGFACE_API_KEY:
        # We still construct the embedder so the agent can boot in dev, but
        # log a strong warning. Calls will fail until the key is provided.
        logger.warning(
            "HUGGINGFACE_API_KEY not set — embedder will fail at first call."
        )
    return OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key=s.HUGGINGFACE_API_KEY or "missing",
            embedding_model=_HF_MODEL,
            embedding_dim=_HF_DIM,
            base_url=_HF_BASE_URL,
        )
    )
