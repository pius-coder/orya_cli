"""Agent settings — single source of truth for env vars."""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@lru_cache(maxsize=1)
def get_settings() -> "Settings":
    return Settings()


class Settings:
    # === LLM providers ===
    GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
    NVIDIA_API_KEY: str | None = os.getenv("NVIDIA_API_KEY")
    CEREBRAS_API_KEY: str | None = os.getenv("CEREBRAS_API_KEY")
    OPENROUTER_API_KEY: str | None = os.getenv("OPENROUTER_API_KEY")

    # === Embeddings (HuggingFace serverless) ===
    HUGGINGFACE_API_KEY: str | None = os.getenv("HUGGINGFACE_API_KEY")

    # === Graphiti tuning ===
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

    # === FalkorDB ===
    FALKORDB_HOST: str = os.getenv("FALKORDB_HOST", "127.0.0.1")
    FALKORDB_PORT: str = os.getenv("FALKORDB_PORT", "6379")

    # === PostgreSQL ===
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "127.0.0.1")
    POSTGRES_PORT: str = os.getenv("POSTGRES_PORT", "5432")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "orya")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "orya")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "orya_secret_2026")

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def postgres_dsn_psycopg(self) -> str:
        # langgraph-checkpoint-postgres uses psycopg-style URL
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
            f"?sslmode=disable"
        )

    # === Internal URLs ===
    AGENT_URL: str = os.getenv("AGENT_URL", "http://127.0.0.1:5001")
    GRAPHITI_SERVER_URL: str = os.getenv(
        "GRAPHITI_SERVER_URL", "http://127.0.0.1:8000"
    )
    GATEWAY_INTERNAL_URL: str = os.getenv(
        "GATEWAY_INTERNAL_URL", "http://127.0.0.1:4001"
    )

    # === Timeouts / limits ===
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "512"))
    SEARCH_NUM_RESULTS_CONTEXT: int = int(
        os.getenv("SEARCH_NUM_RESULTS_CONTEXT", "5")
    )
    SEARCH_NUM_RESULTS_MATCH: int = int(
        os.getenv("SEARCH_NUM_RESULTS_MATCH", "10")
    )
    OPT_IN_TTL_HOURS: int = int(os.getenv("OPT_IN_TTL_HOURS", "72"))

    # === LangSmith (auto if set) ===
    LANGSMITH_TRACING: str = os.getenv("LANGSMITH_TRACING", "false")
    LANGSMITH_API_KEY: str | None = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "orya-v2")

    def configure_langsmith(self) -> None:
        """Set legacy aliases so both old/new code paths work."""
        if self.LANGSMITH_TRACING.lower() in ("1", "true", "yes"):
            os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
            if self.LANGSMITH_API_KEY:
                os.environ.setdefault(
                    "LANGCHAIN_API_KEY", self.LANGSMITH_API_KEY
                )
            os.environ.setdefault("LANGCHAIN_PROJECT", self.LANGSMITH_PROJECT)
