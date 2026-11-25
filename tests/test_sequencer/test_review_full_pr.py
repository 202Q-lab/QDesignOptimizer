"""Fixes from the review of the whole pull request (2026-11-17): edit
leases per browser tab and per command, needs that wait for a study
queued again, a write-back that raised after it was written, a busy
queue lock, and smaller ones."""

from __future__ import annotations

import io

import pytest

from qdesignoptimizer.sequencer.app.queue_state import Bridge, GuiError
from qdesignoptimizer.sequencer.app.server import _gone
from qdesignoptimizer.sequencer.engine import cli
from qdesignoptimizer.sequencer.engine import code_change as cc
from qdesignoptimizer.sequencer.engine import state as st
from qdesignoptimizer.sequencer.engine.backend import SequenceEntry, SequenceSpec
from qdesignoptimizer.sequencer.engine.expressions import Expression
from qdesignoptimizer.sequencer.engine.lease import (
    LEASE_FILE,
    read_lease,
    release_lease,
    take_lease,
)
from qdesignoptimizer.sequencer.engine.queue import QueueError
from qdesignoptimizer.sequencer.engine.runner import Runner

from .helpers import make_project

ADAPTER = (
    "from qdesignoptimizer.sequencer.backends.fake import FakeBackend\n"
    "from pathlib import Path\n"
    "def make_backend():\n"
    "    return FakeBackend(Path(__file__).parent)\n"
)


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    made = make_project(tmp_path / "p")
    (made.root / "sequencer.py").write_text(ADAPTER, "utf-8")
    return made


def _task(project):
    return project.backend.catalog().tasks[0].key


EDIT = {"kind": "queue", "edit": True}


def test_a_second_tab_neither_drops_nor_shares_the_first_tab_s_lease(project):
    study = project.add(_task(project))
    bridge = Bridge(project=project)
    folder = project.queue.folder(study)
    editing = {**EDIT, "key": study}
    assert bridge.keep_editing(editing, "tab-a") == ""
    # Tab B is open on something else: its poll leaves A's lease alone.
    assert bridge.keep_editing({"kind": "queue", "key": study}, "tab-b") == ""
    assert read_lease(folder) is not None
    # An agent is still kept out, and A's save goes through.
    with pytest.raises(Exception, match="being edited by you"):
        take_lease(folder, "claude")
    bridge.save_file(study, "task.yaml", "value: 2\n")
    # Tab B cannot edit it meanwhile, and says why.
    with pytest.raises(GuiError, match="another tab or window"):
        bridge.begin_edit("queue", study, "tab-b")
    # A is done: B may.
    bridge.end_edit("tab-a")
    bridge.begin_edit("queue", study, "tab-b")
    bridge.save_file(study, "task.yaml", "value: 3\n")


def test_two_agents_of_the_same_name_keep_each_other_out(project):
    study = project.add(_task(project))
    folder = project.queue.folder(study)
    take_lease(folder, "claude", holder="claude#1")
    with pytest.raises(Exception, match="being edited by claude"):
        take_lease(folder, "claude", holder="claude#2")
    release_lease(folder, "claude#1")
    take_lease(folder, "claude", holder="claude#2")


def test_park_and_remove_wait_for_someone_else_s_edit(project, capsys):
    study = project.add(_task(project))
    folder = project.queue.folder(study)
    take_lease(folder, "claude", seconds=60)
    bridge = Bridge(project=project)
    with pytest.raises(GuiError, match="being edited by claude"):
        bridge.park(study)
    with pytest.raises(GuiError, match="being edited by claude"):
        bridge.remove(study)
    assert project.queue.ids() == [study]
    release_lease(folder, "claude")
    # The GUI edits it: the command line may not park it ...
    bridge.begin_edit("queue", study)
    ref = str(project.root / "sequencer.py")
    assert cli.main([ref, "park", study]) == 1
    assert "being edited by you (GUI)" in capsys.readouterr().err
    # ... the GUI itself may, and its lease does not go along.
    bridge.park(study)
    assert project.hold.ids() == [study]
    assert not (project.hold.folder(study) / LEASE_FILE).exists()
    assert bridge._editing == {}


def test_a_need_waits_for_a_requeued_copy_not_for_the_old_run(project):
    first = project.add(_task(project), save=True)
    Runner(project).run()
    assert project.need_state(first) == "done"
    (done,) = project.queue.completed()
    project.queue.requeue(done, project.words)  # to run again
    assert project.need_state(first) == "queued"
    then = project.add(_task(project), needs=[first])
    project.queue.write_order([then, *[i for i in project.queue.ids() if i != then]])
    Runner(project).run()
    # The copy ran first, then the one that needs it.
    order = [p.name for p in sorted(project.queue.completed(), key=cc.started_key)]
    assert order[1].startswith(f"{first}_r") and order[2].startswith(then)


def test_a_write_back_that_raised_after_writing_is_recorded_written(project):
    project.add(_task(project), save=True)
    real = project.commit

    def commit_then_die(updates, campaign=""):
        real(updates, campaign)
        raise EOFError("the worker went away")

    project.commit = commit_then_die
    assert Runner(project).run().status == st.FINISHED
    (folder,) = project.queue.completed()
    record = st.result_of(folder)
    assert record["status"] == st.DONE
    assert record["committed"] == record["updates"] != {}


def test_a_write_back_that_never_landed_still_fails(project):
    project.add(_task(project), save=True, on_failure="continue")

    def die(updates, campaign=""):
        raise EOFError("the worker went away")

    project.commit = die
    Runner(project).run()
    (folder,) = project.queue.completed()
    record = st.result_of(folder)
    assert record["status"] == st.FAILED and record["committed"] == {}
    assert "write-back failed" in record["message"]


def test_a_busy_queue_lock_is_a_queue_error_with_a_message(project, monkeypatch):
    def held(*_a, **_k):
        raise st.LockHeld("order.lock held for 30 s")

    monkeypatch.setattr(st, "file_lock", held)
    with pytest.raises(QueueError, match="the queue is busy"):
        with st.order_guard():
            pass
    spec = SequenceSpec("tune", "Tune", (SequenceEntry(_task(project)),))
    with pytest.raises(st.QueueBusy):
        project.add_sequence(spec)
    assert project.queue.ids() == []


def test_sequence_settings_are_asked_for_before_the_order_lock(project, monkeypatch):
    asked = []
    real = project.backend.default_settings

    def default_settings(task):
        asked.append(st._order_depth)
        return real(task)

    monkeypatch.setattr(project.backend, "default_settings", default_settings)
    task = _task(project)
    project.add_sequence(SequenceSpec("tune", "Tune", (SequenceEntry(task),) * 2))
    assert asked == [0, 0]


def test_restart_leaves_a_study_being_edited_and_says_so(project, monkeypatch):
    monkeypatch.setattr(cc, "FILES_KEPT_S", 0.0)
    (project.root / "code.py").write_text("VERSION = 1\n", "utf-8")
    task = _task(project)
    spec = SequenceSpec("tune", "Tune", (SequenceEntry(task, save=True),) * 2)
    ids = project.add_sequence(spec)
    project.park(ids[1])
    Runner(project).run()
    project.unpark(ids[1])
    outside = project.add(task, needs=[ids[1]])
    take_lease(project.queue.folder(outside), "claude", seconds=60)
    (project.root / "code.py").write_text("VERSION = 2\n", "utf-8")
    message = Bridge(project=project).decide_code(ids[1], "restart")
    assert project.job(outside).needs == [ids[1]]  # left as it was
    assert outside in message and "point their needs at s002's" in message
    assert cc.not_repointed(project, "s001") == [outside]


def test_a_negative_base_to_a_fractional_power_is_an_error_not_complex():
    with pytest.raises(ValueError, match="not a real number"):
        Expression("(-8) ** 0.5")({}, {})
    assert Expression("(-8) ** 2")({}, {}) == 64.0


def test_remembered_projects_that_are_not_paths_are_kept(tmp_path):
    adapter = tmp_path / "sequencer.py"
    adapter.write_text(ADAPTER, "utf-8")
    assert not _gone(f"{adapter}:make_backend")
    assert not _gone(str(adapter))
    assert not _gone("qdesignoptimizer.sequencer.backends.fake:make_backend")
    assert not _gone("qdesignoptimizer.sequencer.backends.fake")
    assert _gone(str(tmp_path / "moved" / "sequencer.py"))
    assert _gone(f"{tmp_path / 'moved' / 'sequencer.py'}:make_backend")


def test_the_edit_bar_s_queue_button_is_not_the_row_s(project):
    from qdesignoptimizer.sequencer.app import layout

    project.add(_task(project), hold=True)
    bridge = Bridge(project=project)
    key = bridge.snapshot()["held"][0]["id"]
    snap = bridge.snapshot()
    ids = []

    def walk(node):
        node_id = getattr(node, "id", None)
        if isinstance(node_id, dict) and node_id.get("type") == "act":
            ids.append((node_id["action"], node_id["index"]))
        children = getattr(node, "children", None)
        if not isinstance(children, (list, tuple)):
            children = [children]
        for child in children:
            if child is not None and not isinstance(child, (str, int, float)):
                walk(child)

    walk(layout.held_table(snap, {"kind": "hold", "key": key, "edit": True}))
    walk(layout.details(bridge.details("hold", key), {}, True))
    assert len(ids) == len(set(ids)), ids
    assert ("unpark-bar", key) in ids and ("unpark", key) in ids


# -- the review of the fixes above -----------------------------------------


def test_a_tab_cannot_park_or_remove_another_tab_s_edit(project):
    study = project.add(_task(project))
    bridge = Bridge(project=project)
    bridge.begin_edit("queue", study, "tab-a")
    for act in (bridge.park, bridge.remove):
        with pytest.raises(GuiError, match="another tab or window"):
            act(study, "tab-b")
    assert project.queue.ids() == [study]
    bridge.park(study, "tab-a")  # its own edit: it may
    assert project.hold.ids() == [study]


def test_a_busy_lease_lock_refuses_a_move_with_a_queue_error(project, monkeypatch):
    from contextlib import contextmanager

    from qdesignoptimizer.sequencer.engine import project as project_module

    study = project.add(_task(project))

    @contextmanager
    def busy():
        raise st.LockHeld("lease.lock held for 10 s")
        yield  # pylint: disable=unreachable

    monkeypatch.setattr(project_module, "lease_guard", busy)
    with pytest.raises(QueueError, match="busy a moment"):
        project.park(study, holder="")
    assert project.queue.ids() == [study]


def test_values_already_in_the_store_are_not_a_landed_write_back(project):
    project.add(_task(project), save=True, on_failure="continue")
    Runner(project).run()  # writes its values
    project.queue.requeue(project.queue.completed()[0], project.words)

    def die(updates, campaign=""):
        raise EOFError("the worker went away")

    project.commit = die
    Runner(project).run()  # the same values, never written this time
    rerun = max(project.queue.completed(), key=cc.started_key)
    assert st.result_of(rerun)["status"] == st.FAILED


def test_unreadable_text_over_an_expression_is_kept_as_text():
    from qdesignoptimizer.sequencer.app.forms import parse

    assert parse("6.5", "=p['f'] / 1e9") == 6.5
    assert parse("[TBD", "=p['f'] / 1e9") == "[TBD"
