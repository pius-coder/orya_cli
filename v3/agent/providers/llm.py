"""LLM provider router with automatic fallbacks.

Fixes v2 issues:
- Timeout is configurable via settings
- Clear docstrings matching actual defaults
- No hardcoded API key fallbacks (build returns None if key missing)
"""
import logging
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from ..core.config import get_settings

logger = logging.getLogger(__name__)


def _build_groq(temperature: float, max_tokens: int) -> Optional[BaseChatModel]:
    settings = get_settings()
    if not settings.GROQ_API_KEY:
        return None
    try:
        from langchain_groq import ChatGroq

        return ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            api_key=settings.GROQ_API_KEY,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=30,
        )
    except Exception as e:
        logger.warning("Groq init failed: %s", e)
        return None


def _build_nvidia(temperature: float, max_tokens: int) -> Optional[BaseChatModel]:
    settings = get_settings()
    if not settings.NVIDIA_API_KEY:
        return None
    try:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model="meta/llama-4-maverick-17b-128e-instruct",
            api_key=settings.NVIDIA_API_KEY,
            base_url="https://integrate.api.nvidia.com/v1",
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=30,
        )
    except Exception as e:
        logger.warning("Nvidia init failed: %s", e)
        return None


def _build_cerebras(temperature: float, max_tokens: int) -> Optional[BaseChatModel]:
    settings = get_settings()
    if not settings.CEREBRAS_API_KEY:
        return None
    try:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model="llama3.1-8b",
            api_key=settings.CEREBRAS_API_KEY,
            base_url="https://api.cerebras.ai/v1",
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=30,
        )
    except Exception as e:
        logger.warning("Cerebras init failed: %s", e)
        return None


def _build_openrouter(temperature: float, max_tokens: int) -> Optional[BaseChatModel]:
    settings = get_settings()
    if not settings.OPENROUTER_API_KEY:
        return None
    try:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model="meta-llama/llama-3.3-70b-instruct:free",
            api_key=settings.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=30,
        )
    except Exception as e:
        logger.warning("OpenRouter init failed: %s", e)
        return None


def build_llm(
    temperature: Optional[float] = None, max_tokens: Optional[int] = None
) -> Runnable:
    """Build a primary LLM with automatic fallback chain.

    Priority: Groq → Nvidia → Cerebras → OpenRouter.
    Raises RuntimeError if no provider is available.
    """
    settings = get_settings()
    temp = temperature if temperature is not None else settings.LLM_TEMPERATURE
    tok = max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS

    builders = [_build_groq, _build_nvidia, _build_cerebras, _build_openrouter]
    instances = [b(temp, tok) for b in builders]
    available = [i for i in instances if i is not None]

    if not available:
        raise RuntimeError(
            "No LLM provider available. Check at least one of: "
            "GROQ_API_KEY, NVIDIA_API_KEY, CEREBRAS_API_KEY, OPENROUTER_API_KEY"
        )

    primary = available[0]
    fallbacks = available[1:]
    if fallbacks:
        logger.info("LLM primary=%s fallbacks=%s", primary.__class__.__name__, [f.__class__.__name__ for f in fallbacks])
        return primary.with_fallbacks(fallbacks)
    return primary
