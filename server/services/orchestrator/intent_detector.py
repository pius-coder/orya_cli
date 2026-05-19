"""
Intent Detector — Détecte si l'utilisateur cherche quelqu'un/quelque chose.

Uses the same LLM router as agent-orya (Groq/Nvidia/Cerebras/OpenRouter).
"""

import os
import json
from typing import Optional
from pydantic import BaseModel
import httpx


class Intent(BaseModel):
    type: str  # "search" | "opt_in_reply" | "info"
    query: str = ""
    skills: list[str] = []
    city: Optional[str] = None


PROVIDERS = [
    {
        "name": "groq",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
    },
    {
        "name": "nvidia",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key_env": "NVIDIA_API_KEY",
        "model": "meta/llama-4-maverick-17b-128e-instruct",
    },
    {
        "name": "cerebras",
        "base_url": "https://api.cerebras.ai/v1",
        "api_key_env": "CEREBRAS_API_KEY",
        "model": "llama3.1-8b",
    },
    {
        "name": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "model": "meta-llama/llama-3.3-70b-instruct:free",
    },
]

INTENT_PROMPT = """Analyse ce message et détermine l'intention de l'utilisateur.

Types possibles :
- "search" : cherche un prestataire, un service, un professionnel, quelqu'un qui sait faire X
- "opt_in_reply" : répond oui/non à une proposition de mise en relation
- "info" : pose une question d'information
- "none" : conversation normale, bavardage, pas d'action nécessaire

Si type="search", extrais aussi :
- query: ce qu'il cherche (texte libre)
- skills: mots-clés de compétences ["plomberie", "react", "coiffure"...]
- city: ville si mentionnée (null sinon)

Retourne UNIQUEMENT un JSON :
{{"type": "...", "query": "...", "skills": [...], "city": "..." ou null}}

Message: "{text}"
"""


async def detect_intent(user_id: str, text: str) -> Optional[Intent]:
    """Classify user intent. Returns None if just conversation."""
    if len(text.split()) < 3:
        return None

    # Quick heuristic first (avoid LLM call for obvious cases)
    quick = _quick_intent(text)
    if quick:
        return quick

    prompt = INTENT_PROMPT.format(text=text)
    messages = [
        {"role": "system", "content": "Tu es un classifieur d'intention. JSON strict uniquement."},
        {"role": "user", "content": prompt},
    ]

    for provider in PROVIDERS:
        api_key = os.getenv(provider["api_key_env"], "")
        if not api_key:
            continue

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{provider['base_url']}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": provider["model"],
                        "messages": messages,
                        "temperature": 0.1,
                        "max_tokens": 200,
                        "stream": False,
                    },
                )

            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip()
                parsed = _parse_intent(content)
                if parsed and parsed.get("type") not in ("none", None):
                    return Intent(
                        type=parsed["type"],
                        query=parsed.get("query", ""),
                        skills=parsed.get("skills", []),
                        city=parsed.get("city"),
                    )
                return None
        except Exception as e:
            print(f"[intent_detector] {provider['name']} failed: {e}")
            continue

    return None


def _quick_intent(text: str) -> Optional[Intent]:
    """Fast rule-based intent detection to avoid LLM call."""
    text_lower = text.lower()
    
    search_triggers = [
        "tu connais", "cherche un", "cherche une", "besoin d'un", "besoin d'une",
        "quelqu'un qui", "un plombier", "un électricien", "un dev", "un coiffeur",
        "une coiffeuse", "un mécanicien", "trouver un", "trouver une",
    ]
    
    for trigger in search_triggers:
        if trigger in text_lower:
            # Extract skills from text
            skills = []
            skill_map = {
                "plombier": "plomberie", "électricien": "électricité",
                "dev": "développement", "coiffeur": "coiffure",
                "coiffeuse": "coiffure", "mécanicien": "mécanique",
            }
            for word, skill in skill_map.items():
                if word in text_lower:
                    skills.append(skill)
            
            # Extract city
            cities = ["paris", "lyon", "marseille", "toulouse", "nice", "nantes",
                      "bordeaux", "lille", "rennes", "montpellier"]
            city = None
            for c in cities:
                if c in text_lower:
                    city = c
                    break
            
            return Intent(type="search", query=text, skills=skills, city=city)
    
    return None


def _parse_intent(content: str) -> Optional[dict]:
    """Parse JSON from LLM output."""
    import re

    if "```" in content:
        parts = content.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            try:
                return json.loads(part)
            except:
                continue

    try:
        return json.loads(content)
    except:
        pass

    match = re.search(r'\{.*\}', content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass

    return None
