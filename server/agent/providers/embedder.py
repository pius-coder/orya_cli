"""Embedder factory for Graphiti.

Uses HuggingFace serverless API with ibm-granite/granite-embedding-97m-multilingual-r2.
- Multilingual (French, English, etc.)
- 384 dimensions
- Free, no local model, no PyTorch
"""

from __future__ import annotations

import logging

from .hf_embedder import HuggingFaceEmbedder, HuggingFaceEmbedderConfig
from ..settings import get_settings

logger = logging.getLogger(__name__)


def build_embedder() -> HuggingFaceEmbedder:
    """Build the HuggingFace API embedder (serverless)."""
    s = get_settings()
    if not s.HUGGINGFACE_API_KEY:
        logger.warning("HUGGINGFACE_API_KEY not set — embedder will fail.")
    return HuggingFaceEmbedder(config=HuggingFaceEmbedderConfig(
        model_name="ibm-granite/granite-embedding-97m-multilingual-r2",
        embedding_dim=384,
        api_key=s.HUGGINGFACE_API_KEY or "missing",
    ))
