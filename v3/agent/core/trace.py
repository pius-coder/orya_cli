"""Centralized trace helper for all agent nodes.

Eliminates the _append_trace duplication found in every node file of v2.
"""
from typing import Any


def append_trace(state: dict[str, Any], step: str, detail: str) -> list[dict[str, str]]:
    """Append a trace event to the state's trace list.

    Returns a new trace list (immutable-style) so callers can return it
    directly in node outputs.
    """
    trace = list(state.get("trace", []))
    trace.append({"step": step, "detail": detail})
    return trace
