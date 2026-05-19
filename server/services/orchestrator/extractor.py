"""
Passive Fact Extractor — Now delegates heavy lifting to Graphiti.

Instead of manually extracting facts with LLM, we:
1. Send the raw conversation episode to Graphiti (which extracts entities automatically)
2. Only use LLM for quick intent/fact classification as a lightweight supplement

Graphiti handles: entity extraction, relationship building, temporal tracking.
We just feed it episodes.
"""

import os
import time
from typing import Optional
from pydantic import BaseModel
import httpx

MEMORY_URL = os.getenv("MEMORY_URL", "http://127.0.0.1:5003")


class ExtractedFact(BaseModel):
    kind: str
    value: str
    confidence: float
    source: str = "inline"
    ts: float = 0


async def extract_facts(user_id: str, text: str) -> list[ExtractedFact]:
    """
    Feed the conversation to Graphiti via Memory service.
    Graphiti automatically extracts entities and relationships.
    
    Returns empty list (facts are stored directly in the graph by Graphiti).
    We still return a lightweight local extraction for immediate UI feedback.
    """
    # Skip very short messages
    if len(text.split()) < 3:
        return []

    # Step 1: Send episode to Graphiti (this does the REAL extraction)
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(f"{MEMORY_URL}/episode", json={
                "userId": user_id,
                "text": text,
                "role": "user",
                "source": "conversation",
            })
    except Exception as e:
        print(f"[extractor] failed to send episode to Graphiti: {e}")

    # Step 2: Quick local heuristic extraction for immediate UI feedback
    # (Graphiti does the proper extraction async, this is just for speed)
    facts = _quick_extract(text)
    return facts


def _quick_extract(text: str) -> list[ExtractedFact]:
    """
    Fast rule-based extraction for immediate feedback.
    Graphiti does the real work — this is just for quick UI indicators.
    """
    facts = []
    text_lower = text.lower()
    now = time.time()

    # City detection (common French cities)
    cities = ["paris", "lyon", "marseille", "toulouse", "nice", "nantes", 
              "strasbourg", "montpellier", "bordeaux", "lille", "rennes",
              "villeurbanne", "saint-etienne", "grenoble"]
    for city in cities:
        if city in text_lower:
            facts.append(ExtractedFact(kind="city", value=city, confidence=0.9, ts=now))
            break

    # Skill detection (job keywords)
    job_keywords = {
        "plombier": "plomberie", "dev": "développement", "développeur": "développement",
        "développeuse": "développement", "coiffeur": "coiffure", "coiffeuse": "coiffure",
        "électricien": "électricité", "mécanicien": "mécanique", "boulanger": "boulangerie",
        "graphiste": "design graphique", "photographe": "photographie",
        "react": "React", "next.js": "Next.js", "typescript": "TypeScript",
        "python": "Python", "frontend": "front-end", "backend": "back-end",
    }
    for keyword, skill in job_keywords.items():
        if keyword in text_lower:
            facts.append(ExtractedFact(kind="skill", value=skill, confidence=0.9, ts=now))

    # Frustration/need detection
    frustration_words = ["galère", "problème", "cherche", "besoin", "urgent", "fuite", "panne"]
    for word in frustration_words:
        if word in text_lower:
            # Extract surrounding context
            idx = text_lower.index(word)
            start = max(0, idx - 20)
            end = min(len(text), idx + 40)
            context = text[start:end].strip()
            facts.append(ExtractedFact(kind="need" if "cherche" in word or "besoin" in word else "frustration", 
                                       value=context, confidence=0.7, ts=now))
            break

    return facts
