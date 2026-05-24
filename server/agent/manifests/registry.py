from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined


class ManifestRegistry:
    """Simple Jinja2-based prompt registry.

    Loads manifests (JSON) and renders templates (Markdown) per task.
    Much lighter than the full MemBrain registry — no pydantic-ai dependency.
    """

    def __init__(self, manifests_dir: str | Path):
        self.manifests_dir = Path(manifests_dir)
        self.env = Environment(
            loader=FileSystemLoader(self.manifests_dir),
            undefined=StrictUndefined,
            autoescape=False,
        )
        self._manifests: dict[str, dict] = {}
        self._load_all()

    def _load_all(self) -> None:
        for task_dir in self.manifests_dir.iterdir():
            if not task_dir.is_dir():
                continue
            manifest_path = task_dir / "manifest.json"
            if manifest_path.exists():
                self._manifests[task_dir.name] = json.loads(manifest_path.read_text(encoding="utf-8"))

    def get_manifest(self, task_id: str) -> dict:
        if task_id not in self._manifests:
            raise ValueError(f"Manifest '{task_id}' not found in {self.manifests_dir}")
        return self._manifests[task_id]

    def render(self, task_id: str, **kwargs) -> str:
        """Render the system prompt template for a task with given variables."""
        manifest = self.get_manifest(task_id)
        template_name = manifest.get("template", "system.md")
        template = self.env.get_template(f"{task_id}/{template_name}")
        return template.render(**kwargs)

    def list_tasks(self) -> list[str]:
        return list(self._manifests.keys())
