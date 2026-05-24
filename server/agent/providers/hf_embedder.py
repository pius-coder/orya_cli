"""HuggingFace Inference API embedder (serverless, no local model).

Uses the NEW HuggingFace router endpoint (2025+):
  POST https://router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction

Model: ibm-granite/granite-embedding-97m-multilingual-r2
- Multilingual (French included)
- 384 dimensions
- Free on HF serverless
- Updated regularly by IBM
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import httpx

from graphiti_core.embedder.client import EmbedderClient, EmbedderConfig

logger = logging.getLogger(__name__)

_HF_ROUTER = "https://router.huggingface.co/hf-inference/models"
_DEFAULT_MODEL = "ibm-granite/granite-embedding-97m-multilingual-r2"
_DEFAULT_DIM = 384


class HuggingFaceEmbedderConfig(EmbedderConfig):
    model_name: str = _DEFAULT_MODEL
    embedding_dim: int = _DEFAULT_DIM
    api_key: str = ""


class HuggingFaceEmbedder(EmbedderClient):
    """HuggingFace serverless embedder via router API."""

    def __init__(self, config: HuggingFaceEmbedderConfig | None = None):
        if config is None:
            config = HuggingFaceEmbedderConfig()
        self.config = config
        self._url = f"{_HF_ROUTER}/{config.model_name}/pipeline/feature-extraction"

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

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                self._url,
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                json={"inputs": text, "options": {"wait_for_model": True}},
            )
            resp.raise_for_status()
            data = resp.json()

        # HF returns [float,...] for single input or [[float,...]] depending on model
        if isinstance(data, list):
            if isinstance(data[0], list):
                return data[0]
            return data
        raise TypeError(f"Unexpected HF response: {type(data)}")

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        """Create embeddings for a batch."""
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

        if isinstance(data, list) and data and isinstance(data[0], list):
            return data
        raise TypeError(f"Unexpected HF batch response: {type(data)}")
