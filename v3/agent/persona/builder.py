"""Message builder assembling system prompt, context, few-shot, and history.

Fixes v2 issues:
- Removed fragile Neo4j direct access (episodes now come from Graphiti via facts_context)
- Removed duplicated prompt rules
- Simplified branching logic
"""
from typing import Any, Iterable

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage

from ..core.text import enforce_brevity
from .examples import render_negatives
from .system import get_system_prompt


def _format_facts(facts: Iterable[str]) -> str:
    lines = [f"- {f}" for f in facts if f.strip()]
    return "\n".join(lines) if lines else "Aucun fait connu."


def build_messages(
    *,
    history: list[AnyMessage],
    facts_context: list[str],
    good_examples: list[dict[str, str]],
    tutoyer: bool,
    user_alias: str | None,
) -> list[AnyMessage]:
    """Assemble the full message list for the LLM call.

    Args:
        history: Full conversation history (will be appended after system).
        facts_context: Known facts from Graphiti and other sources.
        good_examples: Positive feedback examples to inject as few-shot.
        tutoyer: Whether to use informal address.
        user_alias: User's alias for personalization.
    """
    system_text = get_system_prompt(tutoyer)

    # Negative examples (what NOT to do)
    negatives = render_negatives()
    if negatives:
        system_text += f"\n\nEXEMPLES DE CE QUE TU NE DOIS PAS DIRE :\n{negatives}\n"

    # Facts context
    system_text += f"\nFAITS CONNUS SUR CET UTILISATEUR :\n{_format_facts(facts_context)}\n"

    # Alias personalization
    if user_alias:
        system_text += f"\nTu parles à {user_alias}.\n"

    messages: list[AnyMessage] = [SystemMessage(content=system_text)]

    # Few-shot examples (max 3)
    for ex in good_examples[:3]:
        messages.append(HumanMessage(content=ex["user_input"]))
        messages.append(AIMessage(content=enforce_brevity(ex["orya_response"], max_words=70)))

    # Conversation history
    messages.extend(history)

    return messages
