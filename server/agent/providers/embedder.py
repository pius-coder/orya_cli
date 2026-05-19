"""Embedder factory for Graphiti.

Uses a LOCAL sentence-transformers model (no API calls, no rate limits).
Runs on CPU — model is lazy-loaded on first embedding request.

Model: sentence-transformers/all-MiniLM-L6-v2 (384 dim)
"""

from __future__ import annotations

import logging

from .hf_embedder import HuggingFaceEmbedder, HuggingFaceEmbedderConfig

logger = logging.getLogger(__name__)


def build_embedder() -> HuggingFaceEmbedder:
    """Build the local HuggingFace embedder.

    No API key needed — runs entirely locally with sentence-transformers.
    The model is downloaded once (~90MB) and cached.
    """
    config = HuggingFaceEmbedderConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim=384,
    )
    logger.info("Embedder configured: local sentence-transformers/all-MiniLM-L6-v2 (384d)")
    return HuggingFaceEmbedder(config=config)
