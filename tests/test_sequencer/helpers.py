"""A fake project in a temp folder."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from qdesignoptimizer.sequencer.backends.fake import FakeBackend
from qdesignoptimizer.sequencer.engine.project import Project


def make_project(folder: Path) -> Project:
    """A fresh fake project rooted at ``folder``."""
    return Project(FakeBackend(folder))


def set_task(project: Project, job_id: str, **values) -> None:
    """Change keys of a queued study's ``task.yaml``."""
    path = project.queue.folder(job_id) / "task.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data.update(values)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def store(project: Project) -> dict:
    """The fake store."""
    return json.loads(project.backend.store_path.read_text("utf-8"))
