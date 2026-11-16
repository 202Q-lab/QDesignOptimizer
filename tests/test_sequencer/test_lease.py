"""Who edits a study (2026-11-16): the GUI and an agent at the command
line never overwrite each other's edits of the same study."""

from __future__ import annotations

import io
import os
import time

import pytest

from qdesignoptimizer.sequencer.app import queue_state
from qdesignoptimizer.sequencer.app.queue_state import GUI_EDITOR, Bridge, GuiError
from qdesignoptimizer.sequencer.engine import cli
from qdesignoptimizer.sequencer.engine.lease import (
    LEASE_FILE,
    LeaseHeld,
    read_lease,
    release_lease,
    take_lease,
)

from .helpers import make_project

ADAPTER = (
    "from qdesignoptimizer.sequencer.backends.fake import FakeBackend\n"
    "from pathlib import Path\n"
    "def make_backend():\n"
    "    return FakeBackend(Path(__file__).parent)\n"
)


@pytest.fixture
def project(tmp_path):
    return make_project(tmp_path / "p")


@pytest.fixture
def study(project):
    return project.add(project.backend.catalog().tasks[0].key)


def _task_text(project, job_id):
    return (project.queue.folder(job_id) / "task.yaml").read_text(encoding="utf-8")


def test_a_lease_keeps_others_out_until_released_or_expired(project, study):
    folder = project.queue.folder(study)
    take_lease(folder, "claude")
    take_lease(folder, "claude")  # renewing your own is fine
    assert read_lease(folder)["by"] == "claude"
    with pytest.raises(LeaseHeld, match="being edited by claude"):
        take_lease(folder, GUI_EDITOR)
    release_lease(folder, GUI_EDITOR)  # not theirs to release
    assert read_lease(folder) is not None
    release_lease(folder, "claude")
    assert read_lease(folder) is None
    take_lease(folder, "claude", seconds=0.05)
    time.sleep(0.1)
    assert read_lease(folder) is None  # expired: no one is blocked
    take_lease(folder, GUI_EDITOR)


def test_edit_in_the_gui_is_blocked_while_an_agent_edits(project, study):
    bridge = Bridge(project=project)
    take_lease(project.queue.folder(study), "claude")
    (row,) = bridge.snapshot()["rows"]
    assert row["editing"].startswith("claude")
    with pytest.raises(GuiError, match="being edited by claude"):
        bridge.begin_edit("queue", study)
    with pytest.raises(GuiError, match="being edited by claude .*; nothing saved"):
        bridge.save_file(study, "task.yaml", "value: 2\n")
    release_lease(project.queue.folder(study), "claude")
    assert bridge.snapshot()["rows"][0]["editing"] == ""
    bridge.begin_edit("queue", study)
    assert read_lease(project.queue.folder(study))["by"] == GUI_EDITOR


def test_the_gui_s_own_lease_shows_no_tag(project, study):
    bridge = Bridge(project=project)
    bridge.begin_edit("queue", study)
    assert bridge.snapshot()["rows"][0]["editing"] == ""


def test_a_save_overtaken_by_an_outside_edit_is_refused(project, study):
    bridge = Bridge(project=project)
    bridge.begin_edit("queue", study)
    bridge.save_file(study, "task.yaml", "value: 2\n")  # our own: fine
    bridge.save_file(study, "task.yaml", "value: 3\n")  # and again
    path = project.queue.folder(study) / "task.yaml"
    path.write_text("value: 9\n", encoding="utf-8")  # someone ignoring the lease
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000))
    with pytest.raises(GuiError, match="changed outside the GUI"):
        bridge.save_file(study, "task.yaml", "value: 4\n")
    assert _task_text(project, study) == "value: 9\n"
    # Done editing, then Edit again: it starts from what is there now.
    bridge.end_edit()
    bridge.begin_edit("queue", study)
    bridge.save_file(study, "task.yaml", "value: 4\n")
    assert _task_text(project, study) == "value: 4\n"


def test_the_poll_renews_the_lease_and_drops_it_after_editing(project, study):
    bridge = Bridge(project=project)
    folder = project.queue.folder(study)
    editing = {"kind": "queue", "key": study, "edit": True}
    assert bridge.keep_editing(editing) == ""
    first = read_lease(folder)["until"]
    time.sleep(0.02)
    bridge.keep_editing(editing)
    assert read_lease(folder)["until"] > first
    assert bridge.keep_editing({"kind": "queue", "key": study}) == ""
    assert not (folder / LEASE_FILE).exists()


def test_a_lease_taken_from_the_gui_is_said_once(project, study):
    bridge = Bridge(project=project)
    folder = project.queue.folder(study)
    editing = {"kind": "queue", "key": study, "edit": True}
    bridge.keep_editing(editing)
    release_lease(folder, GUI_EDITOR)  # ours lapsed ...
    take_lease(folder, "claude")  # ... and the agent took it
    assert "being edited by claude" in bridge.keep_editing(editing)
    assert bridge.keep_editing(editing) == ""


def test_a_study_gone_while_edited_raises_nothing_every_poll(project, study):
    bridge = Bridge(project=project)
    editing = {"kind": "queue", "key": study, "edit": True}
    bridge.keep_editing(editing)
    project.queue.remove(study)
    assert bridge.keep_editing(editing) == ""


def test_the_lease_file_does_not_make_the_poll_re_read(project, study, monkeypatch):
    bridge = Bridge(project=project)
    bridge.snapshot()
    seen = []
    real = queue_state.load_job
    monkeypatch.setattr(
        queue_state, "load_job", lambda *a, **k: seen.append(a) or real(*a, **k)
    )
    take_lease(project.queue.folder(study), "claude")
    bridge.snapshot()
    assert seen == []


def _ref(project):
    adapter = project.root / "sequencer.py"
    adapter.write_text(ADAPTER, "utf-8")
    return str(adapter)


def test_the_command_line_writes_under_a_lease(project, study, capsys, monkeypatch):
    ref = _ref(project)
    monkeypatch.setattr("sys.stdin", io.StringIO("value: 5\n"))
    assert cli.main([ref, "write", study, "task.yaml", "--by", "claude"]) == 0
    assert _task_text(project, study) == "value: 5\n"
    assert f"wrote task.yaml of {study}" in capsys.readouterr().out
    assert read_lease(project.queue.folder(study)) is None  # released

    # While the GUI edits it, the command line is refused.
    bridge = Bridge(project=project)
    bridge.begin_edit("queue", study)
    source = project.root / "new_task.yaml"
    source.write_text("value: 6\n", encoding="utf-8")
    code = cli.main([ref, "write", study, "task.yaml", "--from", str(source)])
    assert code == 1
    assert "being edited by you (GUI)" in capsys.readouterr().err
    assert _task_text(project, study) == "value: 5\n"


def test_the_command_line_writes_a_held_study_and_reports_problems(
    project, capsys, monkeypatch
):
    ref = _ref(project)
    held = project.add(project.backend.catalog().tasks[0].key, hold=True)
    job = (project.hold.folder(held) / "job.yaml").read_text(encoding="utf-8")
    monkeypatch.setattr(
        "sys.stdin", io.StringIO(job + "settings: {no_such_setting: 1}\n")
    )
    assert cli.main([ref, "write", held, "job.yaml", "--by", "claude"]) == 2
    out = capsys.readouterr().out
    assert "cannot run yet" in out and "no_such_setting" in out


def test_a_lease_is_not_written_into_a_folder_that_is_gone(tmp_path):
    from qdesignoptimizer.sequencer.engine import lease as lease_module

    gone = tmp_path / "queue" / "s01-x"
    with pytest.raises(FileNotFoundError):
        lease_module._write_in(gone, {"by": "claude"})
    assert not gone.exists()  # not made anew next to a claimed study
