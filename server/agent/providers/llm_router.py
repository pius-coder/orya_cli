"""LangChain LLM router with provider fallbacks.

Composes a `RunnableWithFallbacks` chaining:
    Groq (primary) → Nvidia → Cerebras → OpenRouter

Provider availability is determined by env-var presence — any missing key
makes the corresponding fallback skipped instead of raising at construction.
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from ..settings import get_settings

logger = logging.getLogger(__name__)

# Endpoints documented by each provider — they are OpenAI-compatible chat
# completions.
_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
_CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Model defaults — Llama 4 family for primary, conservative fallbacks.
_GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
_NVIDIA_MODEL = "meta/llama-4-maverick-17b-128e-instruct"
_CEREBRAS_MODEL = "llama3.1-8b"
_OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct:free"


def _build_groq(temperature: float, max_tokens: int) -> Optional[BaseChatModel]:
    s = get_settings()
    if not s.GROQ_API_KEY:
        return None
    return ChatGroq(
        model=_GROQ_MODEL,
        api_key=s.GROQ_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=30,
    )


def _build_nvidia(temperature: float, max_tokens: int) -> Optional[BaseChatModel]:
    s = get_settings()
    if not s.NVIDIA_API_KEY:
        return None
    return ChatOpenAI(
        model=_NVIDIA_MODEL,
        base_url=_NVIDIA_BASE_URL,
        api_key=s.NVIDIA_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=30,
    )


def _build_cerebras(
    temperature: float, max_tokens: int
) -> Optional[BaseChatModel]:
    s = get_settings()
    if not s.CEREBRAS_API_KEY:
        return None
    return ChatOpenAI(
        model=_CEREBRAS_MODEL,
        base_url=_CEREBRAS_BASE_URL,
        api_key=s.CEREBRAS_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=30,
    )


def _build_openrouter(
    temperature: float, max_tokens: int
) -> Optional[BaseChatModel]:
    s = get_settings()
    if not s.OPENROUTER_API_KEY:
        return None
    return ChatOpenAI(
        model=_OPENROUTER_MODEL,
        base_url=_OPENROUTER_BASE_URL,
        api_key=s.OPENROUTER_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=30,
    )


def build_llm(
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Runnable:
    """Return a Runnable LLM with automatic provider fallbacks.

    Raises RuntimeError if no provider is configured at all.
    """

    s = get_settings()
    t = s.LLM_TEMPERATURE if temperature is None else temperature
    m = s.LLM_MAX_TOKENS if max_tokens is None else max_tokens

    candidates: list[BaseChatModel] = []
    for build in (_build_groq, _build_nvidia, _build_cerebras, _build_openrouter):
        llm = build(t, m)
        if llm is not None:
            candidates.append(llm)

    if not candidates:
        raise RuntimeError(
            "No LLM provider configured. Set at least one of "
            "GROQ_API_KEY / NVIDIA_API_KEY / CEREBRAS_API_KEY / "
            "OPENROUTER_API_KEY."
        )

    primary = candidates[0]
    fallbacks = candidates[1:]

    logger.info(
        "LLM router built — primary=%s, fallbacks=%d",
        type(primary).__name__,
        len(fallbacks),
    )

    if not fallbacks:
        return primary
    return primary.with_fallbacks(fallbacks)
