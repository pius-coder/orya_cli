"""HuggingFace embedder for Graphiti.

Uses the serverless inference API. Free tier, no rate limit for lightweight models.
"""
import logging
from typing import Iterable

import httpx
from graphiti_core.embedder.client import EmbedderClient, EmbedderConfig

from ..core.config import get_settings

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_DIM = 384


class HuggingFaceEmbedderConfig(EmbedderConfig):
    api_key: str = ""
    embedding_model: str = _DEFAULT_MODEL
    embedding_dim: int = _DEFAULT_DIM
    base_url: str = "https://api-inference.huggingface.co/pipeline/feature-extraction"


class HuggingFaceEmbedder(EmbedderClient):
    def __init__(self, config: HuggingFaceEmbedderConfig | None = None) -> None:
        if config is None:
            settings = get_settings()
            config = HuggingFaceEmbedderConfig(
                api_key=settings.HUGGINGFACE_API_KEY,
                embedding_model=_DEFAULT_MODEL,
                embedding_dim=_DEFAULT_DIM,
            )
        self.config = config
        self._client = httpx.AsyncClient(timeout=60.0)

    async def create(self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]) -> list[float]:
        if isinstance(input_data, str):
            texts = [input_data]
        elif isinstance(input_data, list) and input_data and isinstance(input_data[0], str):
            texts = input_data  # type: ignore[assignment]
        else:
            texts = [str(input_data)]
        results = await self.create_batch(texts)
        return results[0] if results else []

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        if not input_data_list:
            return []
        url = f"{self.config.base_url}/{self.config.embedding_model}"
        headers = {"Authorization": f"Bearer {self.config.api_key}"} if self.config.api_key else {}
        payload = {"inputs": [t.replace("\n", " ") for t in input_data_list]}

        try:
            resp = await self._client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            logger.error("HF embedder HTTP error: %s — %s", e.response.status_code, e.response.text[:200])
            raise
        except Exception as e:
            logger.error("HF embedder error: %s", e)
            raise

        # Response can be [float] for single or [[float]] for batch
        if isinstance(data, list) and data:
            if isinstance(data[0], list):
                return data  # type: ignore[return-value]
            return [data]  # type: ignore[return-value]
        return []
