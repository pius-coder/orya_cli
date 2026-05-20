"""Smoke tests for Orya v3 agent architecture.

Validates:
- Manifest loading and rendering
- System prompt building
- Graph builder imports
"""
import inspect
from pathlib import Path

import pytest

from agent.manifests.registry import ManifestRegistry
from agent.nodes.tool_agent import make_tool_agent_node


@pytest.fixture(scope="session")
def manifests() -> ManifestRegistry:
    agent_dir = Path(__file__).parent.parent.parent
    return ManifestRegistry(str(agent_dir / "manifests"))


def test_manifests_load(manifests: ManifestRegistry) -> None:
    tasks = manifests.list_tasks()
    assert "writer" in tasks
    assert "reflection-user" in tasks
    assert "reflection-orya" in tasks


def test_manifest_render_writer(manifests: ManifestRegistry) -> None:
    rendered = manifests.render("writer", alias="Test", facts="none", tutoyer=True)
    assert "Orya" in rendered
    assert "WhatsApp" in rendered or "SMS" in rendered


def test_manifest_render_reflections(manifests: ManifestRegistry) -> None:
    user_rendered = manifests.render("reflection-user")
    assert "synthèse" in user_rendered.lower() or "portrait" in user_rendered.lower()
    orya_rendered = manifests.render("reflection-orya")
    assert "Orya" in orya_rendered


def test_graph_builder_signature() -> None:
    sig = inspect.signature(make_tool_agent_node)
    params = list(sig.parameters.keys())
    assert "llm" in params
    assert "manifests" in params
    assert "executor" in params
