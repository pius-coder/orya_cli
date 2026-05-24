"""Text extraction and formatting utilities.

Eliminates duplication of _extract_text and _enforce_brevity across nodes.
"""
from typing import Any


def extract_text(ai_message: Any) -> str:
    """Extract plain text from an LLM response (AIMessage or dict-like)."""
    if hasattr(ai_message, "content"):
        content = ai_message.content
    elif isinstance(ai_message, dict):
        content = ai_message.get("content", "")
    else:
        content = str(ai_message)

    if isinstance(content, str):
        return content.strip()
    return ""


def enforce_brevity(text: str, max_words: int = 70) -> str:
    """Truncate text to at most max_words, ending at a sentence boundary if possible."""
    words = text.split()
    if len(words) <= max_words:
        return text
    truncated = words[:max_words]
    result = " ".join(truncated)
    # Try to end cleanly at a sentence boundary
    for ending in (".", "!", "?", ","):
        idx = result.rfind(ending)
        if idx > len(result) * 0.6:
            return result[: idx + 1]
    return result + "..."
