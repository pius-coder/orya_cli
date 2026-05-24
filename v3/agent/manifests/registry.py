"""Lightweight Jinja2 manifest registry for prompt templates.

Unchanged from v2 (it was already clean), just moved to v3.
"""
import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined


class ManifestRegistry:
    def __init__(self, manifests_dir: str | Path) -> None:
        self.manifests_dir = Path(manifests_dir)
        self._manifests: dict[str, dict] = {}
        self._templates: dict[str, Environment] = {}
        self._load_all()

    def _load_all(self) -> None:
        for subdir in self.manifests_dir.iterdir():
            if not subdir.is_dir():
                continue
            manifest_file = subdir / "manifest.json"
            system_file = subdir / "system.md"
            if not manifest_file.exists() or not system_file.exists():
                continue
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            task_id = manifest.get("task_id", subdir.name)
            self._manifests[task_id] = manifest
            env = Environment(
                loader=FileSystemLoader(str(subdir)),
                undefined=StrictUndefined,
                autoescape=False,
            )
            self._templates[task_id] = env

    def get_manifest(self, task_id: str) -> dict:
        if task_id not in self._manifests:
            raise KeyError(f"Manifest '{task_id}' not found")
        return self._manifests[task_id]

    def render(self, task_id: str, **kwargs: object) -> str:
        if task_id not in self._templates:
            raise KeyError(f"Template '{task_id}' not found")
        template = self._templates[task_id].get_template("system.md")
        return template.render(**kwargs)

    def list_tasks(self) -> list[str]:
        return list(self._manifests.keys())
