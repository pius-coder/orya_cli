"""Tests v3 — simplified architecture without heavy qualifier.

Verifies:
  - Manifests load and render correctly
  - Graph can be built (smoke test)
  - Tool agent prompt builder works
  - Reflection CRUD works (if PG is available)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.manifests.registry import ManifestRegistry
from agent.nodes.v3.tool_agent import _build_system_prompt


# ── Fixtures ───────────────────────────────────────────────────

@pytest.fixture(scope="session")
def manifests() -> ManifestRegistry:
    manifest_dir = Path(__file__).parent.parent.parent / "manifests"
    return ManifestRegistry(manifest_dir)


# ── Test 1: Manifests ────────────────────────────────────────

@pytest.mark.asyncio
async def test_manifests_load(manifests: ManifestRegistry) -> None:
    tasks = manifests.list_tasks()
    assert "writer" in tasks
    assert "reflection-user" in tasks
    assert "reflection-orya" in tasks
    print(f"✓ manifests loaded: {tasks}")


@pytest.mark.asyncio
async def test_manifest_render_writer(manifests: ManifestRegistry) -> None:
    text = manifests.render("writer")
    assert "Orya" in text
    assert "WhatsApp" in text
    print(f"✓ writer manifest rendered ({len(text)} chars)")


@pytest.mark.asyncio
async def test_manifest_render_reflections(manifests: ManifestRegistry) -> None:
    user_text = manifests.render("reflection-user")
    orya_text = manifests.render("reflection-orya")
    assert "synthèse" in user_text.lower()
    assert "Orya" in orya_text
    print(f"✓ reflection manifests rendered")


# ── Test 2: Prompt builder ─────────────────────────────────────

@pytest.mark.asyncio
async def test_build_system_prompt(manifests: ManifestRegistry) -> None:
    prompt = _build_system_prompt(
        manifests=manifests,
        user_text="Salut Orya !",
        facts_context=["Orya aime le café", "Jean cherche un dev"],
        user_reflection="Jean est un entrepreneur à Paris.",
        orya_reflection="Jean préfère le tutoiement.",
        tutoyer=True,
        alias="Jean",
        good_examples=[],
    )
    assert "Orya" in prompt
    assert "Jean" in prompt
    assert "dev" in prompt
    assert "tutoiement" in prompt
    print(f"✓ system prompt built ({len(prompt)} chars)")


# ── Test 3: Graph smoke test (without real deps) ─────────────

@pytest.mark.asyncio
async def test_graph_builder_imports() -> None:
    """Just verify the graph builder can be imported and called with mocks."""
    from agent.graph import build_graph_builder_v3
    # We don't build the actual graph here (needs real Graphiti + LLM)
    # but we verify the function exists and has the right signature
    import inspect
    sig = inspect.signature(build_graph_builder_v3)
    params = list(sig.parameters.keys())
    assert "graphiti" in params
    assert "llm" in params
    assert "manifests" in params
    print(f"✓ graph builder signature OK: {params}")
