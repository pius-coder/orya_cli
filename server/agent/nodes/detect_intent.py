"""Node: detect_intent.

Heuristic-first intent detection: classify whether the user is asking the
network for help (→ trigger search_match) or just chatting / sharing context.

We avoid an LLM call when a simple regex match suffices; otherwise fall back
to a small JSON-only LLM completion.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from ..models import OryaState

logger = logging.getLogger(__name__)


_HELP_TRIGGERS = re.compile(
    r"\b(?:je\s+cherche|je\s+recherche|j['e]\s*aurais\s+besoin|"
    r"qui\s+(?:peut|connaît|saurait)|t['u]?\s*connais|tu\s+aurais|"
    r"quelqu['e]?un\s+(?:qui|pour)|recommandes?|recommandez|"
    r"trouver\s+un[e]?|besoin\s+d['e]?un[e]?)",
    re.IGNORECASE,
)


_INTENT_SYSTEM = (
    "Tu es un classifieur. Tu reçois un message d'un utilisateur. "
    "Tu réponds STRICTEMENT par un JSON sur une seule ligne avec les clés: "
    "action ('search', 'chat', 'thanks', 'opt_in_response'), "
    "domain (string|null) et location (string|null). "
    "Pas d'explication, pas de markdown."
)


def make_detect_intent_node(small_llm: Runnable | None = None):
    """Build the node. `small_llm` should be a low-temperature LLM used for
    the JSON fallback only; if None, only the heuristic is used."""

    async def detect_intent_node(state: OryaState) -> dict[str, Any]:
        text = state.get("last_user_text") or ""
        if not text:
            return {"intent": None, "trace": _append_trace(state, "detect_intent", "no text")}

        # Fast path
        if _HELP_TRIGGERS.search(text):
            intent = {
                "action": "search",
                "domain": _guess_domain(text),
                "location": _guess_location(text),
            }
            return {
                "intent": intent,
                "trace": _append_trace(state, "detect_intent", "heuristic:search"),
            }

        if len(text.split()) < 5:
            return {
                "intent": {"action": "chat", "domain": None, "location": None},
                "trace": _append_trace(state, "detect_intent", "heuristic:short"),
            }

        if small_llm is None:
            return {
                "intent": {"action": "chat", "domain": None, "location": None},
                "trace": _append_trace(
                    state, "detect_intent", "no llm fallback"
                ),
            }

        try:
            response = await small_llm.ainvoke(
                [
                    SystemMessage(content=_INTENT_SYSTEM),
                    HumanMessage(content=text),
                ]
            )
            content = getattr(response, "content", str(response))
            content = _strip_to_json(str(content))
            intent = json.loads(content)
            if not isinstance(intent, dict) or "action" not in intent:
                raise ValueError("invalid json shape")
            return {
                "intent": intent,
                "trace": _append_trace(state, "detect_intent", "llm:ok"),
            }
        except Exception as e:
            logger.warning("intent llm fallback failed: %s", e)
            return {
                "intent": {"action": "chat", "domain": None, "location": None},
                "trace": _append_trace(state, "detect_intent", f"llm:err {e}"),
            }

    return detect_intent_node


def _guess_domain(text: str) -> str | None:
    text_l = text.lower()
    keywords = {
        "dev": "software_dev",
        "développeur": "software_dev",
        "developer": "software_dev",
        "front": "software_dev",
        "back": "software_dev",
        "fullstack": "software_dev",
        "designer": "design",
        "ui": "design",
        "ux": "design",
        "plombier": "plumbing",
        "avocat": "law",
        "fiscaliste": "fiscal_law",
        "coach": "coaching",
        "marketing": "marketing",
        "produit": "product",
        "pm": "product",
        "cto": "engineering_lead",
        "ceo": "founder",
    }
    for k, v in keywords.items():
        if k in text_l:
            return v
    return None


def _guess_location(text: str) -> str | None:
    m = re.search(
        r"\b(?:à|en|au|aux)\s+([A-ZÀ-Ý][\w\-']{2,40}(?:\s[A-ZÀ-Ý][\w\-']{2,40})?)",
        text,
    )
    if m:
        return m.group(1).strip()
    return None


def _strip_to_json(content: str) -> str:
    """Some models prepend ``` fences. Slice to the first '{' and last '}'."""

    if "{" in content and "}" in content:
        start = content.index("{")
        end = content.rindex("}") + 1
        return content[start:end]
    return content


def _append_trace(state: OryaState, step: str, detail: str) -> list[dict[str, Any]]:
    existing = list(state.get("trace") or [])
    existing.append({"step": step, "detail": detail})
    return existing
