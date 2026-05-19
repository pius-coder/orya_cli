"""
LLM Router — Cycles between free providers (Groq, Nvidia, OpenRouter)
with automatic fallback on rate-limit or error.

Strategy: Try providers in order. If one fails (429, 5xx, timeout),
move to the next. Rotate the priority each call to spread load.
"""

import os
import httpx
from typing import Any

PROVIDERS = [
    {
        "name": "groq",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "model": "llama-3.3-70b-versatile",
    },
    {
        "name": "nvidia",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key_env": "NVIDIA_API_KEY",
        "model": "meta/llama-3.3-70b-instruct",
    },
    {
        "name": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "model": "meta-llama/llama-3.3-70b-instruct:free",
    },
]

# Rotate index to spread load
_rotation_idx = 0


async def call_llm(
    messages: list[dict[str, str]],
    temperature: float = 0.85,
    max_tokens: int = 200,
) -> dict[str, Any]:
    """
    Try each provider in rotation. Returns {"text": ..., "provider": ...}
    or raises if all fail.
    """
    global _rotation_idx
    errors = []

    for i in range(len(PROVIDERS)):
        provider = PROVIDERS[(_rotation_idx + i) % len(PROVIDERS)]
        api_key = os.getenv(provider["api_key_env"], "")
        if not api_key:
            continue

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{provider['base_url']}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": provider["model"],
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                )

            if resp.status_code == 200:
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                _rotation_idx = (_rotation_idx + i + 1) % len(PROVIDERS)
                return {"text": text.strip(), "provider": provider["name"]}

            errors.append(f"{provider['name']}: HTTP {resp.status_code}")
        except Exception as e:
            errors.append(f"{provider['name']}: {e}")

    raise RuntimeError(f"All LLM providers failed: {errors}")
