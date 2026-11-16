"""The project's code changing while studies run (2026-11-16): the
runner reloads it before the next study; a study sequence half run on
the old code waits until someone continues or restarts it; the GUI
reloads too and says its pictures are out of date."""

from __future__ import annotations

import io

import pytest

from qdesignoptimizer.sequencer.app import layout
from qdesignoptimizer.sequencer.app.queue_state import Bridge
from qdesignoptimizer.sequencer.engine import cli
from qdesignoptimizer.sequencer.engine import code_change as cc
from qdesignoptimizer.sequencer.engine import state as st
from qdesignoptimizer.sequencer.engine.backend import SequenceEntry, SequenceSpec
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
    monkeypatch.setattr(cc, "FILES_KEPT_S", 0.0)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    made = make_project(tmp_path / "p")
    (made.root / "code.py").write_text("VERSION = 1\n", "utf-8")
    (made.root / "sequencer.py").write_text(ADAPTER, "utf-8")
    return made


def _change_code(project, text="VERSION = 2\n"):
    (project.root / "code.py").write_text(text, "utf-8")


def _half_run_sequence(project, n=3):
    """A sequence of n studies whose first has run on code version 1."""
    task = project.backend.catalog().tasks[0].key
    spec = SequenceSpec(
        "tune", "Tune", tuple(SequenceEntry(task, save=True) for _ in range(n))
    )
    ids = project.add_sequence(spec)
    for job_id in ids[1:]:
        project.park(job_id)
    Runner(project).run()
    for job_id in ids[1:]:
        project.unpark(job_id)
    return ids


def test_a_run_records_the_code_it_ran_with(project):
    job_id = project.add(project.backend.catalog().tasks[0].key)
    Runner(project).run()
    (folder,) = project.queue.completed()
    assert st.result_of(folder)["code"] == cc.code_version(project) != ""
    assert job_id in folder.name


def test_the_runner_reloads_changed_code_before_the_next_study(project):
    task = project.backend.catalog().tasks[0].key
    first, second = project.add(task), project.add(task)
    run = project.backend.run

    def run_and_edit(folder, job):
        result = run(folder, job)
        if job.id == first:
            _change_code(project)  # saved while the first one solves
        return result

    project.backend.run = run_and_edit
    Runner(project).run()
    calls = [c for c in project.backend.calls if c[0] in ("run", "reload")]
    assert calls == [("run", first), ("reload", ""), ("run", second)]
    versions = {
        p.name.split("_")[0]: st.result_of(p)["code"] for p in project.queue.completed()
    }
    assert len(set(versions.values())) == 2  # each ran with its own


def test_a_half_run_sequence_waits_and_the_rest_moves_up(project):
    ids = _half_run_sequence(project)
    alone = project.add(project.backend.catalog().tasks[0].key)
    _change_code(project)
    final = Runner(project).run()
    assert final.status == st.FINISHED
    assert "waiting for a decision (the code changed): s01" in final.message
    assert project.queue.ids() == ids[1:]  # alone ran, the sequence waits
    assert any(alone in p.name for p in project.queue.completed())
    rows = {r["id"]: r for r in Bridge(project=project).snapshot()["rows"]}
    assert "the code changed since s01 began" in rows[ids[1]]["code_hold"]
    assert "code changed: continue or restart?" in str(
        layout.queue_table({"rows": list(rows.values())}, {})
    )


def test_continue_runs_the_rest_on_the_new_code(project, capsys):
    ids = _half_run_sequence(project)
    _change_code(project)
    assert cli.main([str(project.root / "sequencer.py"), "decide"]) == 0
    assert "s01: the code changed since it began; 2 to run" in capsys.readouterr().out
    assert (
        cli.main([str(project.root / "sequencer.py"), "decide", "s01", "continue"]) == 0
    )
    assert Runner(project).run().status == st.FINISHED
    assert project.queue.ids() == []
    assert all(any(i in p.name for p in project.queue.completed()) for i in ids)
    # Nothing of s01 is left to hold when the code changes again.
    _change_code(project, "VERSION = 3\n")
    assert cc.waiting(project) == {}


def test_restart_queues_the_sequence_again_and_holds_the_rest(project):
    ids = _half_run_sequence(project)
    outside = project.add(project.backend.catalog().tasks[0].key, needs=[ids[2]])
    _change_code(project)
    message = Bridge(project=project).decide_code(ids[1], "restart")
    # A study outside it that needed s01's third now needs s02's.
    assert (
        project.job(outside).needs
        == [i for i in project.queue.ids() if i.startswith("s02-")][2:3]
    )
    project.queue.remove(outside)
    assert "s01 restarts as s02 (3 studies" in message
    queued = project.queue.ids()
    assert len(queued) == 3 and all(i.startswith("s02-") for i in queued)
    assert sorted(project.hold.ids()) == sorted(ids[1:])
    jobs = [project.job(i) for i in queued]
    assert [j.raw.get("restarted_from") for j in jobs] == ["s01"] * 3
    assert [j.sequence["entry"] for j in jobs] == [1, 2, 3]
    # Its needs point at the new studies, not at s01's.
    assert jobs[1].needs == [queued[0]]
    assert jobs[2].needs == [queued[0], queued[1]]
    assert Runner(project).run().status == st.FINISHED
    assert project.queue.ids() == []


def test_a_sequence_without_code_files_never_waits(project):
    (project.root / "code.py").unlink()
    ids = _half_run_sequence(project, 2)
    assert Runner(project).run().status == st.FINISHED
    assert project.queue.ids() == []
    assert len(ids) == 2


def test_the_gui_reloads_changed_code_and_says_so(project):
    bridge = Bridge(project=project)
    assert bridge.snapshot()["code"]["rev"] == 0
    _change_code(project)
    snap = bridge.snapshot()
    assert snap["code"]["rev"] == 1
    assert snap["code"]["note"].startswith("code.py changed: reloaded at")
    assert ("reload", "") in project.backend.calls
    assert bridge.snapshot()["code"]["rev"] == 1  # once per change


# -- review fixes (2026-11-16) ----------------------------------------------------


def test_a_running_study_is_waited_for_and_not_failed(project, capsys):
    job_id = project.add(project.backend.catalog().tasks[0].key)
    folder = project.queue.start(project.queue.claim(job_id), job_id)
    st.write_json(folder / st.RESULT_FILE, {"id": job_id, "status": st.RUNNING})
    assert project.need_state(job_id) == "running"
    adapter = str(project.root / "sequencer.py")
    assert (
        cli.main([adapter, "wait", job_id, "--timeout", "0.05", "--every", "0.01"]) == 3
    )
    assert f"still queued: {job_id}" in capsys.readouterr().out


def test_code_that_does_not_load_makes_the_runner_wait(project):
    import threading
    import time

    task = project.backend.catalog().tasks[0].key
    first, second = project.add(task), project.add(task)
    run = project.backend.run

    def run_and_break(folder, job):
        result = run(folder, job)
        _change_code(project, "VERSION = (\n")  # a typo, saved mid-run
        return result

    def reload():
        raise SyntaxError("'(' was never closed")

    project.backend.run = run_and_break
    project.backend.reload = reload
    done = []
    thread = threading.Thread(target=lambda: done.append(Runner(project).run()))
    thread.start()
    deadline = time.monotonic() + 10
    while "does not load" not in st.RunnerState.load(project.out).message:
        assert time.monotonic() < deadline
        time.sleep(0.02)
    assert project.queue.ids() == [second]  # waits, did not end
    st.request_stop(project.out)
    thread.join(10)
    assert done[0].status == st.STOPPED
    assert first


def test_a_code_file_that_cannot_be_read_now_is_no_crash(project, monkeypatch):
    def refuse(self):
        raise PermissionError("held by a scanner")

    monkeypatch.setattr(type(project.root), "read_bytes", refuse)
    assert set(cc.code_hashes(project).values()) == {"unreadable"}


def test_restart_refuses_when_a_study_cannot_be_read(project):
    ids = _half_run_sequence(project)
    (project.queue.folder(ids[2]) / "job.yaml").write_text("id: [\n", "utf-8")
    _change_code(project)
    with pytest.raises(Exception, match="cannot be restarted"):
        cc.decide(project, "s01", "restart")
    assert project.queue.ids() == ids[1:]  # nothing moved


def test_the_gui_does_not_retry_code_that_does_not_load(project):
    bridge = Bridge(project=project)
    bridge.snapshot()
    tries = []

    def reload():
        tries.append(1)
        raise SyntaxError("bad")

    project.backend.reload = reload
    _change_code(project)
    note = bridge.snapshot()["code"]["note"]
    assert note.startswith("code.py changed but does not load (SyntaxError")
    bridge.snapshot()
    assert len(tries) == 1
