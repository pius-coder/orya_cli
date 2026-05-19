"""
Passive Fact Extractor — Extrait des infos d'une conversation sans que l'user s'en rende compte.

Utilise le LLM pour analyser chaque message et extraire :
- skills (métier, compétences)
- city (ville, quartier)
- need (ce que l'user cherche)
- frustration (problèmes rencontrés)
- preference (ce qu'il aime/veut)
- personal (hobbies, famille, contexte perso)

Le prompt est conçu pour NE PAS halluciner : si rien de pertinent → retourne [].
"""

import os
import json
from typing import Optional
from pydantic import BaseModel

import httpx

# Reuse same LLM providers as agent-orya
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

EXTRACTION_PROMPT = """Tu es un extracteur d'informations. Analyse le message suivant et retourne UNIQUEMENT un JSON array avec les facts trouvées.

Catégories possibles :
- "skill" : métier, compétence, ce que la personne fait
- "city" : ville, quartier, zone géographique
- "need" : ce que la personne cherche (un service, un pro...)
- "frustration" : un problème, une galère mentionnée
- "preference" : un goût, une préférence exprimée
- "personal" : info personnelle (famille, hobby, contexte de vie)

RÈGLES :
- Si RIEN de pertinent → retourne []
- Ne JAMAIS inventer. Extraire UNIQUEMENT ce qui est explicitement dit.
- Confidence: 0.9 si c'est explicite, 0.6 si c'est implicite/déduit
- Retourne UNIQUEMENT le JSON, rien d'autre.

Format: [{{"kind": "...", "value": "...", "confidence": 0.X}}]

Message: "{text}"
"""


class ExtractedFact(BaseModel):
    kind: str
    value: str
    confidence: float
    source: str = "inline"
    ts: float = 0


async def extract_facts(user_id: str, text: str) -> list[ExtractedFact]:
    """
    Extract facts from a user message using LLM.
    Returns empty list if nothing relevant found.
    """
    import time

    # Skip very short messages (greetings, yes/no)
    if len(text.split()) < 3:
        return []

    prompt = EXTRACTION_PROMPT.format(text=text)
    messages = [
        {"role": "system", "content": "Tu es un extracteur JSON strict. Retourne UNIQUEMENT du JSON valide."},
        {"role": "user", "content": prompt},
    ]

    # Try each provider
    for provider in PROVIDERS:
        api_key = os.getenv(provider["api_key_env"], "")
        if not api_key:
            continue

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    f"{provider['base_url']}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": provider["model"],
                        "messages": messages,
                        "temperature": 0.1,  # Low temp for extraction
                        "max_tokens": 300,
                    },
                )

            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip()

                # Parse JSON from response
                facts_raw = _parse_json(content)
                if facts_raw:
                    now = time.time()
                    return [
                        ExtractedFact(
                            kind=f["kind"],
                            value=f["value"],
                            confidence=f.get("confidence", 0.7),
                            source="inline",
                            ts=now,
                        )
                        for f in facts_raw
                        if f.get("kind") and f.get("value")
                    ]
                return []
        except Exception as e:
            print(f"[extractor] {provider['name']} failed: {e}")
            continue

    return []


def _parse_json(content: str) -> Optional[list[dict]]:
    """Try to parse JSON from LLM output, handling markdown code blocks."""
    # Remove markdown code blocks if present
    if "```" in content:
        parts = content.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            try:
                result = json.loads(part)
                if isinstance(result, list):
                    return result
            except:
                continue

    # Direct parse
    try:
        result = json.loads(content)
        if isinstance(result, list):
            return result
    except:
        pass

    # Try to find array in text
    import re
    match = re.search(r'\[.*\]', content, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return result
        except:
            pass

    return None
