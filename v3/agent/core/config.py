"""Single source of truth for Orya v3 configuration.

Merges agent/settings.py and graphiti-server/settings.py from v2
into one validated configuration module.
"""
import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Runtime configuration read from environment variables.

    Raises RuntimeError early on boot if required values are missing.
    """

    # ── LLM Providers ──────────────────────────────────────────────
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY", "")
    CEREBRAS_API_KEY: str = os.getenv("CEREBRAS_API_KEY", "")
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    HUGGINGFACE_API_KEY: str = os.getenv("HUGGINGFACE_API_KEY", "")

    # ── Neo4j / Graphiti ───────────────────────────────────────────
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "")
    GRAPHITI_LLM_PROVIDER: str = os.getenv("GRAPHITI_LLM_PROVIDER", "openrouter")
    GRAPHITI_LLM_MODEL: str = os.getenv(
        "GRAPHITI_LLM_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct"
    )
    GRAPHITI_LLM_SMALL_MODEL: str = os.getenv(
        "GRAPHITI_LLM_SMALL_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"
    )
    SEMAPHORE_LIMIT: int = int(os.getenv("SEMAPHORE_LIMIT", "5"))
    GRAPHITI_TELEMETRY_ENABLED: bool = os.getenv("GRAPHITI_TELEMETRY_ENABLED", "false").lower() == "true"

    # ── PostgreSQL ─────────────────────────────────────────────────
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "127.0.0.1")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "orya")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "orya")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "")

    # ── LangSmith ──────────────────────────────────────────────────
    LANGCHAIN_TRACING_V2: bool = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
    LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "orya-production")

    # ── Qdrant (Vector DB) ─────────────────────────────────────────
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "384"))

    # ── Service URLs ───────────────────────────────────────────────
    AGENT_URL: str = os.getenv("AGENT_URL", "http://127.0.0.1:5001")
    GRAPHITI_SERVER_URL: str = os.getenv("GRAPHITI_SERVER_URL", "http://127.0.0.1:8000")
    GATEWAY_PORT: int = int(os.getenv("GATEWAY_PORT", "4001"))
    GATEWAY_INTERNAL_URL: str = os.getenv("GATEWAY_INTERNAL_URL", "http://127.0.0.1:4001")
    INTERNAL_API_KEY: str = os.getenv("INTERNAL_API_KEY", "")

    # ── Tuning ─────────────────────────────────────────────────────
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "512"))
    SEARCH_NUM_RESULTS_CONTEXT: int = int(os.getenv("SEARCH_NUM_RESULTS_CONTEXT", "5"))
    SEARCH_NUM_RESULTS_MATCH: int = int(os.getenv("SEARCH_NUM_RESULTS_MATCH", "20"))
    OPT_IN_TTL_HOURS: int = int(os.getenv("OPT_IN_TTL_HOURS", "72"))

    @property
    def qdrant_url(self) -> str:
        return self.QDRANT_URL

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def postgres_dsn_psycopg(self) -> str:
        # psycopg-compatible DSN without sslmode=disable hardcoding
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    def __post_init__(self) -> None:
        # Validate required DB password
        if not self.POSTGRES_PASSWORD:
            raise RuntimeError("POSTGRES_PASSWORD is required")
        if not self.NEO4J_PASSWORD:
            raise RuntimeError("NEO4J_PASSWORD is required")

    def configure_langsmith(self) -> None:
        """Enable LangSmith tracing if configured."""
        if self.LANGCHAIN_TRACING_V2 and self.LANGCHAIN_API_KEY:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = self.LANGCHAIN_API_KEY
            os.environ["LANGCHAIN_PROJECT"] = self.LANGCHAIN_PROJECT


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.__post_init__()
    return s
