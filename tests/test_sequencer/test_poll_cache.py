"""The GUI's 2-second poll (2026-11-16): a study is read again only when
its own files change, so the poll stays cheap with a long queue."""

from __future__ import annotations

import os

import pytest

from qdesignoptimizer.sequencer.app import queue_state
from qdesignoptimizer.sequencer.app.queue_state import Bridge

from .helpers import make_project


@pytest.fixture
def project(tmp_path):
    return make_project(tmp_path / "p")


@pytest.fixture
def reads(monkeypatch):
    """Counts the job.yaml reads the poll makes."""
    seen = []
    real = queue_state.load_job

    def counted(folder, words=None):
        seen.append(folder)
        return real(folder, words) if words is not None else real(folder)

    monkeypatch.setattr(queue_state, "load_job", counted)
    return seen


def _bump(path):
    """Write a file so its modification time surely changes."""
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000))


def test_an_unchanged_study_is_not_read_again(project, reads):
    task = project.backend.catalog().tasks[0].key
    ids = [project.add(task) for _ in range(3)]
    bridge = Bridge(project=project)
    assert [r["id"] for r in bridge.snapshot()["rows"]] == ids
    assert len(reads) == 3
    reads.clear()
    bridge.snapshot()
    assert reads == []


def test_a_changed_study_shows_at_once(project, reads):
    task = project.backend.catalog().tasks[0].key
    first, second = project.add(task), project.add(task)
    bridge = Bridge(project=project)
    bridge.snapshot()
    reads.clear()
    path = project.queue.folder(first) / "job.yaml"
    path.write_text(
        path.read_text(encoding="utf-8") + "notes: edited by hand\n", "utf-8"
    )
    _bump(path)
    rows = {r["id"]: r for r in bridge.snapshot()["rows"]}
    assert rows[first]["notes"] == "edited by hand"
    assert [f.name for f in reads] == [first]  # not the other one
    assert second in rows


def test_a_study_s_problems_are_checked_again_after_a_while(
    project, reads, monkeypatch
):
    """What a study depends on outside its folder (templates, campaigns)
    is rechecked every ROW_RECHECK_S."""
    project.add(project.backend.catalog().tasks[0].key)
    bridge = Bridge(project=project)
    bridge.snapshot()
    reads.clear()
    monkeypatch.setattr(queue_state, "ROW_RECHECK_S", 0.0)
    bridge.snapshot()
    assert len(reads) == 1


def test_a_removed_or_held_study_leaves_the_poll(project):
    task = project.backend.catalog().tasks[0].key
    first, second = project.add(task), project.add(task)
    bridge = Bridge(project=project)
    bridge.snapshot()
    bridge.park(first)
    bridge.remove(second)
    snap = bridge.snapshot()
    assert snap["rows"] == []
    assert [h["id"] for h in snap["held"]] == [bridge.hold_key_of(first)]


def test_a_new_campaign_shows_in_the_next_poll(project):
    bridge = Bridge(project=project)
    assert bridge.snapshot()["campaigns"] == []
    project.new_campaign("tuneup", "Can it hit its targets?")
    assert [c["key"] for c in bridge.snapshot()["campaigns"]] == ["tuneup"]


def test_a_folder_without_job_yaml_is_not_queued(project):
    task = project.backend.catalog().tasks[0].key
    job_id = project.add(task)
    (project.queue.queue_dir / "half_made").mkdir()
    (project.queue.queue_dir / ".hidden").mkdir()
    assert project.queue.ids() == [job_id]
