"""Embedder factory for Graphiti.

Uses HuggingFace Inference API (serverless) — no local model, no PyTorch.
"""

from __future__ import annotations

import logging

from .hf_embedder import HuggingFaceEmbedder, HuggingFaceEmbedderConfig
from ..settings import get_settings

logger = logging.getLogger(__name__)


def build_embedder() -> HuggingFaceEmbedder:
    """Build the HuggingFace API embedder (serverless, no local deps)."""
    s = get_settings()
    if not s.HUGGINGFACE_API_KEY:
        logger.warning("HUGGINGFACE_API_KEY not set — embedder will fail.")
    return HuggingFaceEmbedder(config=HuggingFaceEmbedderConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim=384,
        api_key=s.HUGGINGFACE_API_KEY or "missing",
    ))
