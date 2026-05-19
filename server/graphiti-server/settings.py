"""Graphiti-server settings — mirrors agent/settings.py for the subset it
needs."""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@lru_cache(maxsize=1)
def get_settings() -> "Settings":
    return Settings()


class Settings:
    GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
    HUGGINGFACE_API_KEY: str | None = os.getenv("HUGGINGFACE_API_KEY")
    GRAPHITI_LLM_MODEL: str = os.getenv(
        "GRAPHITI_LLM_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"
    )
    GRAPHITI_LLM_SMALL_MODEL: str = os.getenv(
        "GRAPHITI_LLM_SMALL_MODEL", "llama-3.1-8b-instant"
    )
    SEMAPHORE_LIMIT: int = int(os.getenv("SEMAPHORE_LIMIT", "5"))
    GRAPHITI_TELEMETRY_ENABLED: str = os.getenv(
        "GRAPHITI_TELEMETRY_ENABLED", "false"
    )
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "")
