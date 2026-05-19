"""Local HuggingFace embedder using sentence-transformers.

Based on the community implementation from getzep/graphiti#437.
Runs locally on CPU — no API calls, no rate limits, no format issues.

Model: sentence-transformers/all-MiniLM-L6-v2 (384 dim)
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from graphiti_core.embedder.client import EmbedderClient, EmbedderConfig

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_DIM = 384


class HuggingFaceEmbedderConfig(EmbedderConfig):
    """Configuration for local HuggingFace embedder."""

    model_name: str = _DEFAULT_MODEL
    embedding_dim: int = _DEFAULT_DIM


class HuggingFaceEmbedder(EmbedderClient):
    """Local sentence-transformers embedder — CPU, no API, no rate limits."""

    def __init__(self, config: HuggingFaceEmbedderConfig | None = None):
        if config is None:
            config = HuggingFaceEmbedderConfig()
        self.config = config
        self._model = None

    def _get_model(self):
        """Lazy-load the model on first call."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for HuggingFaceEmbedder. "
                    "Install with: pip install sentence-transformers"
                ) from None
            logger.info("Loading embedding model: %s", self.config.model_name)
            self._model = SentenceTransformer(self.config.model_name)
            logger.info("Embedding model loaded (dim=%d).", self.config.embedding_dim)
        return self._model

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        """Create embedding for a single input."""
        if isinstance(input_data, str):
            text = input_data.replace("\n", " ")
        elif isinstance(input_data, list) and input_data and isinstance(input_data[0], str):
            text = input_data[0].replace("\n", " ")
        else:
            text = str(input_data).replace("\n", " ")

        model = self._get_model()
        embedding = model.encode(text, show_progress_bar=False)

        if hasattr(embedding, "tolist"):
            return embedding.tolist()
        return list(embedding)

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        """Create embeddings for a batch of inputs."""
        if not input_data_list:
            return []

        texts = [t.replace("\n", " ") for t in input_data_list]
        model = self._get_model()
        embeddings = model.encode(texts, show_progress_bar=False)

        if hasattr(embeddings, "tolist"):
            return embeddings.tolist()
        return [e.tolist() if hasattr(e, "tolist") else list(e) for e in embeddings]
