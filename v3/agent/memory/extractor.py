"""Entity extraction with 2-pass refinement.

Inspired by MemBrain's entity-extractor agent.
Pass 1: Extract entities from raw message.
Pass 2: Refine with known entities from user's PKG.
"""
import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

logger = logging.getLogger(__name__)

ENTITY_EXTRACTOR_PROMPT = """You are an entity extraction assistant.
Extract a deduplicated list of specific, identifiable entity references from the chat messages.

RULES:
- Extract people, places, companies, skills, needs, specific objects, pets, groups.
- DO NOT extract: generic activities, abstract concepts, emotions, category labels.
- ALWAYS extract: named pets, specific objects bought/owned, relational groups ("his coworkers"), key actions' objects.
- Entity refs should be 1-4 words, the shortest phrase identifying the referent.
- NEVER include temporal expressions in refs.

Output ONLY valid JSON: {"entities": ["Entity1", "Entity2", ...]}"""


async def extract_entities(
    messages_text: str,
    llm: Runnable,
    known_entities: list[str] | None = None,
) -> list[str]:
    """Two-pass entity extraction.

    Pass 1: Extract from raw text.
    Pass 2: If known_entities provided, refine to resolve aliases/pronouns.
    """
    # Pass 1
    prompt1 = [
        SystemMessage(content=ENTITY_EXTRACTOR_PROMPT),
        HumanMessage(content=f"Messages:\n{messages_text}"),
    ]
    try:
        resp1 = await llm.ainvoke(prompt1)
        data1 = _extract_json(str(getattr(resp1, "content", "")))
        entities = list(dict.fromkeys(data1.get("entities", [])))
    except Exception as e:
        logger.error("Entity extraction pass 1 failed: %s", e)
        return []

    if not known_entities or not entities:
        return entities

    # Pass 2: Refine with known entities
    known_text = "\n".join(f"- {e}" for e in known_entities[:30])
    prompt2 = [
        SystemMessage(
            content=ENTITY_EXTRACTOR_PROMPT
            + "\n\nWhen extracting, prefer refs that match known entities listed below."
            + "\nResolve pronouns and aliases to known entities when obvious."
        ),
        HumanMessage(
            content=f"Known Entities:\n{known_text}\n\nMessages:\n{messages_text}"
        ),
    ]
    try:
        resp2 = await llm.ainvoke(prompt2)
        data2 = _extract_json(str(getattr(resp2, "content", "")))
        entities = list(dict.fromkeys(data2.get("entities", [])))
    except Exception as e:
        logger.warning("Entity extraction pass 2 failed, keeping pass 1: %s", e)

    return [e.strip() for e in entities if e.strip()]


def _extract_json(text: str) -> dict[str, Any]:
    """Extract JSON from LLM response, handling markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```json"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)
