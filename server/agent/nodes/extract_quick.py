"""Node: extract_quick.

Rule-based fast extraction of obvious facts from the user message, used to
give the CLI immediate UI feedback (`fact_recorded` events). The heavy
extraction is left to Graphiti in `persist_episode`.
"""

from __future__ import annotations

import re
from typing import Any

from ..models import OryaState

# A handful of quick patterns. Confidence values are deliberately conservative
# so the UI badges them as 'soft' findings.
_PATTERNS: list[tuple[str, str, float]] = [
    (r"\bje\s+m'appelle\s+([A-ZÀ-Ý][\w\-']{1,30})", "name", 0.85),
    (r"\bj['e]\s*habite\s+(?:à|en|au)\s+([A-ZÀ-Ý][\w\-']{2,40})", "city", 0.7),
    (r"\bje\s+vis\s+(?:à|en|au)\s+([A-ZÀ-Ý][\w\-']{2,40})", "city", 0.7),
    (r"\bje\s+suis\s+(?:dev|développeur|developer|ingénieur|plombier|avocat|coach|designer|product manager|pm|cto|ceo)\b", "occupation", 0.8),
    (r"\bje\s+cherche\s+([\w\sÀ-ÿ\-]{3,80}?)(?:[\.\!\?]|$)", "need", 0.6),
    (r"\b(\d{1,2})\s*ans\b", "age", 0.65),
    (r"\bje\s+suis\s+frustré", "emotion:frustration", 0.7),
]


def extract_quick_node(state: OryaState) -> dict[str, Any]:
    text = state.get("last_user_text") or ""
    facts: list[dict[str, Any]] = []
    if text:
        lc = text  # keep original casing for value capture
        for pattern, label, conf in _PATTERNS:
            m = re.search(pattern, lc, flags=re.IGNORECASE)
            if not m:
                continue
            value = m.group(1).strip() if m.groups() else "true"
            if not value:
                value = "true"
            facts.append(
                {"label": label, "value": value, "confidence": conf}
            )
    return {
        "extracted_facts": facts,
        "trace": _append_trace(state, "extract_quick", f"{len(facts)} facts"),
    }


def _append_trace(state: OryaState, step: str, detail: str) -> list[dict[str, Any]]:
    existing = list(state.get("trace") or [])
    existing.append({"step": step, "detail": detail})
    return existing
