"""HuggingFace Inference API embedder (serverless, no local model).

Calls the HuggingFace feature-extraction endpoint directly.
No PyTorch, no sentence-transformers — just httpx HTTP calls.

Endpoint: POST https://api-inference.huggingface.co/models/{model}
Response format: [[float, float, ...]] for single input
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import httpx

from graphiti_core.embedder.client import EmbedderClient, EmbedderConfig

logger = logging.getLogger(__name__)

_HF_API_URL = "https://api-inference.huggingface.co/models"
_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_DIM = 384


class HuggingFaceEmbedderConfig(EmbedderConfig):
    model_name: str = _DEFAULT_MODEL
    embedding_dim: int = _DEFAULT_DIM
    api_key: str = ""


class HuggingFaceEmbedder(EmbedderClient):
    """HuggingFace serverless embedder — API calls only, no local model."""

    def __init__(self, config: HuggingFaceEmbedderConfig | None = None):
        if config is None:
            config = HuggingFaceEmbedderConfig()
        self.config = config
        self._url = f"{_HF_API_URL}/{config.model_name}"

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        """Create embedding for a single input via HF API."""
        if isinstance(input_data, str):
            text = input_data.replace("\n", " ")
        elif isinstance(input_data, list) and input_data and isinstance(input_data[0], str):
            text = input_data[0].replace("\n", " ")
        else:
            text = str(input_data).replace("\n", " ")

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                self._url,
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                json={"inputs": text, "options": {"wait_for_model": True}},
            )
            resp.raise_for_status()
            data = resp.json()

        # HF returns [[float,...]] for single input or [float,...] sometimes
        if isinstance(data, list):
            if isinstance(data[0], list):
                return data[0]
            return data
        raise TypeError(f"Unexpected HF embedding response: {type(data)}")

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        """Create embeddings for a batch via HF API."""
        if not input_data_list:
            return []

        texts = [t.replace("\n", " ") for t in input_data_list]

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                self._url,
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                json={"inputs": texts, "options": {"wait_for_model": True}},
            )
            resp.raise_for_status()
            data = resp.json()

        # HF returns [[float,...], [float,...], ...] for batch
        if isinstance(data, list) and all(isinstance(d, list) for d in data):
            return data
        raise TypeError(f"Unexpected HF batch embedding response: {type(data)}")
