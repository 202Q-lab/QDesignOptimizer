"""Review of 2026-11-15: editors (GUI, CLI, other processes) racing the
runner."""

from __future__ import annotations

import json
import threading

import pytest

from qdesignoptimizer.sequencer.app.queue_state import TAKEN, Bridge, GuiError
from qdesignoptimizer.sequencer.backends import _store
from qdesignoptimizer.sequencer.engine import state as st
from qdesignoptimizer.sequencer.engine.queue import QueueError, keep_listed
from qdesignoptimizer.sequencer.engine.runner import Runner
from qdesignoptimizer.sequencer.engine.workspace import open_workspace

from .helpers import make_project, set_task, store


@pytest.fixture
def project(tmp_path):
    return make_project(tmp_path / "proj")


def _adapter(folder):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "sequencer.py").write_text(
        "from pathlib import Path\n"
        "from qdesignoptimizer.sequencer.backends.fake import FakeBackend\n"
        "def make_backend():\n"
        "    return FakeBackend(Path(__file__).parent)\n",
        "utf-8",
    )
    return str(folder / "sequencer.py")


# -- the order keeps a study the runner holds claimed ----------------------


def test_keep_listed_puts_a_held_id_back_where_it_was():
    assert keep_listed(["b", "c"], ["a", "b", "c"], {"a"}) == ["a", "b", "c"]
    assert keep_listed(["c", "b"], ["a", "b", "c"], {"a"}) == ["a", "c", "b"]
    assert keep_listed(["c"], ["b", "a", "c"], {"a"}) == ["c", "a"]
    assert keep_listed(["b"], ["b", "a"], {"a"}) == ["b", "a"]
    assert keep_listed(["b"], ["a", "b"], set()) == ["b"]


@pytest.mark.parametrize("edit", ["add", "reorder", "remove"])
def test_a_study_whose_check_fails_keeps_its_place_after_gui_edits(project, edit):
    """The GUI adds, drags or removes while the runner checks the first
    study; the check fails, and the study goes back first in line, not
    last (its dependents must not run before it)."""
    first = project.add("A.calibrate")
    second = project.add("A.measure")
    third = project.add("B.measure")
    real = project.problems

    def problems(job_id, deep=False, folder=None):
        if deep and folder is not None and ".claimed-" in str(folder):
            if edit == "add":
                project.add("B.calibrate")
            elif edit == "reorder":
                project.queue.save_order([third, second])
            else:
                project.queue.remove(third)
            return ["a typo"]
        return real(job_id, deep, folder)

    project.problems = problems
    final = Runner(project).run()
    assert final.status == st.ERROR
    assert project.queue.ids()[0] == first


def test_a_workspace_order_keeps_a_claimed_study(tmp_path):
    workspace = open_workspace(
        [_adapter(tmp_path / "a"), _adapter(tmp_path / "b")], isolate=False
    )
    try:
        a = workspace.projects["a"]
        b = workspace.projects["b"]
        first = workspace.added(a, [a.add("A.calibrate")])[0]
        second = workspace.added(b, [b.add("A.calibrate")])[0]
        claimed = a.queue.claim(first.split("/", 1)[1])
        workspace.added(b, [b.add("B.calibrate")])  # written meanwhile
        a.queue.release(claimed, first.split("/", 1)[1])
        assert workspace.ids()[:2] == [first, second]
    finally:
        workspace.close()


# -- campaign stores through a worker process --------------------------------


def test_a_campaign_study_run_through_a_worker_writes_its_campaigns_store(
    tmp_path,
):
    """The GUI and a multi-project runner talk to each project through a
    worker process; the campaign must travel with each call."""
    workspace = open_workspace([_adapter(tmp_path / "a"), _adapter(tmp_path / "b")])
    try:
        project = workspace.projects["a"]
        assert type(project.backend).__name__ == "RemoteBackend"
        folder = project.new_campaign("x")
        assert project.campaign_store("x") == folder / "store.json"
        job_id = project.add("A.calibrate", save=True, campaign="x")
        (project.queue.folder(job_id) / "task.yaml").write_text("value: 7.0\n")
        assert Runner(workspace).run().status == st.FINISHED
        assert json.loads((folder / "store.json").read_text())["A_value"] == 7.0
        assert json.loads((tmp_path / "a" / "store.json").read_text())["A_value"] == 0.0
        assert project.promote("x") == {"A_value": 7.0}
    finally:
        workspace.close()


def test_a_campaign_call_does_not_leak_into_another_threads_call(project):
    """One backend serves several GUI threads: a campaign's store is set
    for one call only, in its own thread."""
    project.new_campaign("x")
    project.campaign_store("x").write_text(json.dumps({"A_value": 9.0}), "utf-8")
    backend = project.backend
    seen = {}
    inside = threading.Event()
    done = threading.Event()

    def campaign_call():
        with backend.store_of("x"):
            seen["campaign"] = backend.store_values()["A_value"]
            inside.set()
            done.wait(5)

    thread = threading.Thread(target=campaign_call)
    thread.start()
    inside.wait(5)
    seen["project"] = backend.store_values()["A_value"]
    done.set()
    thread.join()
    assert seen == {"campaign": 9.0, "project": 0.0}
    assert backend.store_values()["A_value"] == 0.0  # reset after the call


# -- write-backs never lose another writer's values ---------------------------


def test_a_store_write_waits_for_another_writer(project, monkeypatch):
    """A write-back reading the store while a promote writes it: the
    lock makes the second wait, so neither value is lost."""
    path = project.backend.default_store
    real_read = _store.read_json
    entered = threading.Event()
    release = threading.Event()

    def slow_read(p):
        data = real_read(p)
        if threading.current_thread().name == "first":
            entered.set()
            release.wait(5)
        return data

    monkeypatch.setattr(_store, "read_json", slow_read)
    first = threading.Thread(
        target=lambda: _store.update_json(path, {"A_value": 1.0}), name="first"
    )
    first.start()
    entered.wait(5)
    second = threading.Thread(target=lambda: _store.update_json(path, {"B_value": 2.0}))
    second.start()
    second.join(0.3)
    assert second.is_alive()  # waits for the first to finish
    release.set()
    first.join()
    second.join()
    assert store(project) == {"A_value": 1.0, "B_value": 2.0}


# -- the runner and the GUI say what happened ---------------------------------


def test_a_study_held_open_stops_the_runner_with_a_reason(project, monkeypatch):
    job_id = project.add("A.calibrate")

    def refused(_job_id):
        raise PermissionError("[WinError 5] Access is denied")

    monkeypatch.setattr(project.queue, "claim", refused)
    final = Runner(project).run()
    assert final.status == st.ERROR
    assert "has its folder open" in final.message
    assert project.queue.ids() == [job_id]


def test_remove_and_delete_after_the_runner_took_the_study_say_so(project):
    job_id = project.add("A.calibrate")
    (project.queue.folder(job_id) / "design_variables.json").write_text("{}")
    project.queue.claim(job_id)
    with pytest.raises(QueueError, match="not queued"):
        project.queue.remove(job_id)
    with pytest.raises(QueueError, match="not queued"):
        project.queue.delete_file(job_id, "design_variables.json")


def test_a_save_just_before_the_runner_took_the_study_says_it_runs_with_it(
    project, monkeypatch
):
    bridge = Bridge(project=project)
    job_id = project.add("A.calibrate")
    set_task(project, job_id, value=1.0)
    real = project.queue.write_file

    def write_then_claimed(*args, **kwargs):
        real(*args, **kwargs)
        project.queue.claim(job_id)

    monkeypatch.setattr(project.queue, "write_file", write_then_claimed)
    message = bridge.save_file(job_id, "task.yaml", "value: 2.0\n")
    assert message == "saved task.yaml; " + TAKEN
    claimed = project.queue.queue_dir / f".claimed-{job_id}"
    assert "2.0" in (claimed / "task.yaml").read_text()
    monkeypatch.setattr(project.queue, "write_file", real)
    with pytest.raises(GuiError, match="no longer there"):
        bridge.save_file(job_id, "task.yaml", "value: 3.0\n")
