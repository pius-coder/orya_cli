"""Natural fact generation with entity coverage validation.

Inspired by MemBrain's fact-generator agent.
Generates self-contained natural language facts with [Entity] references.
"""
import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from .models import NaturalFact

logger = logging.getLogger(__name__)

FACT_GENERATOR_PROMPT = """You are a fact generation assistant.
Generate concise, self-contained facts from the conversation messages.

RULES:
- Each fact must be a standalone semantic unit (1-3 sentences).
- Reference ALL extracted entities using [EntityName] syntax (with brackets).
- Replace pronouns with their referents.
- Preserve concrete details: names, quantities, places, dates.
- Prioritize user messages. Only include assistant messages if they confirm a user fact.
- Include temporal annotations as [raw::YYYY-MM-DD] when a date is mentioned.
- Each fact must reference at least one entity.

Output ONLY valid JSON: {"facts": [{"text": "[Alice] works at [Google]", "entities": ["Alice", "Google"], "time_raw": null, "time_resolved": null}]}"""

_ENTITY_RE = re.compile(r"\[([^\]]+)\]")


async def generate_facts(
    messages_text: str,
    entity_names: list[str],
    llm: Runnable,
) -> list[NaturalFact]:
    """Generate natural facts with entity coverage validation."""
    entity_list = ", ".join(f"[{e}]" for e in entity_names)
    prompt = [
        SystemMessage(content=FACT_GENERATOR_PROMPT),
        HumanMessage(
            content=f"Extracted Entities: {entity_list}\n\nMessages:\n{messages_text}"
        ),
    ]

    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            resp = await llm.ainvoke(prompt)
            data = _extract_json(str(getattr(resp, "content", "")))
            raw_facts = data.get("facts", [])
            facts = []
            for f in raw_facts:
                text = f.get("text", "")
                refs = _ENTITY_RE.findall(text)
                # Validate: all bracket refs must be in allowed entities
                invalid = [r for r in refs if r not in entity_names]
                if invalid:
                    logger.warning("Fact has invalid entity refs: %s", invalid)
                    continue
                facts.append(
                    NaturalFact(
                        text=text,
                        entities=list(dict.fromkeys(refs)),
                        time_raw=f.get("time_raw"),
                        time_resolved=f.get("time_resolved"),
                    )
                )
            if facts:
                return facts
            if attempt == max_retries:
                return []
        except Exception as e:
            logger.error("Fact generation attempt %d failed: %s", attempt + 1, e)
            if attempt == max_retries:
                return []

    return []


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```json"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)
