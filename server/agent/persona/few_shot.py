"""Build the LLM prompt messages: system + few-shot + facts + history."""

from __future__ import annotations

from typing import Any, Iterable

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
)

from .negatives import render_negatives
from .system_prompt import get_system_prompt


def _format_facts(facts: Iterable[str]) -> str:
    items = [f"- {f}" for f in facts if f]
    if not items:
        return "Aucun fait pertinent encore connu sur cette personne."
    return "\n".join(items)


def build_messages(
    *,
    history: list[AnyMessage],
    facts_context: list[str],
    good_examples: list[dict[str, Any]] | None = None,
    tutoyer: bool = True,
    user_alias: str | None = None,
    user_prompt_content: str | None = None,
) -> list[AnyMessage]:
    """Assemble the full prompt context.

    Args:
        history: ongoing conversation messages from LangGraph state.
        facts_context: short factual blurbs returned by Graphiti search.
        good_examples: list of {user_text, assistant_reply} from PG feedback
            (rating=1) to use as positive few-shot.
        tutoyer: whether Orya tutoies (True) or vouvoies (False).
        user_alias: optional name to address.
        user_prompt_content: dynamic user prompt formatted as XML/Markdown.
    """

    system = get_system_prompt(tutoyer=tutoyer)

    if user_prompt_content:
        persona_addendum = "\n\n" + render_negatives(tutoyer=tutoyer)
    else:
        facts_block = _format_facts(facts_context)
        persona_addendum = (
            f"\n\nFaits déjà connus sur la personne :\n{facts_block}"
        )
        if user_alias:
            persona_addendum += f"\n\nLa personne s'appelle '{user_alias}'."
        persona_addendum += "\n\n" + render_negatives(tutoyer=tutoyer)

    messages: list[AnyMessage] = [SystemMessage(content=system + persona_addendum)]

    # Inline a small number of positive examples so the model learns what
    # 'good' looks like in production.
    if good_examples:
        for ex in good_examples[:3]:
            ut = ex.get("user_text") or ""
            ar = ex.get("assistant_reply") or ""
            if ut and ar:
                messages.append(HumanMessage(content=ut))
                messages.append(AIMessage(content=ar))

    # Then the live conversation history.
    if user_prompt_content and history and isinstance(history[-1], HumanMessage):
        messages.extend(history[:-1])
        messages.append(HumanMessage(content=user_prompt_content))
    else:
        messages.extend(history)
        
    return messages
