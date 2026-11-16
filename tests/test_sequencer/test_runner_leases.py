"""Which queued study the runner starts (2026-11-16): the first that may
start. One being edited, or one whose ``needs`` have not all run
successfully, is passed over and the studies after it move up."""

from __future__ import annotations

import io
import threading
import time

import pytest

from qdesignoptimizer.sequencer.app import layout
from qdesignoptimizer.sequencer.app.queue_state import Bridge
from qdesignoptimizer.sequencer.engine import cli
from qdesignoptimizer.sequencer.engine import runner as runner_module
from qdesignoptimizer.sequencer.engine import state as st
from qdesignoptimizer.sequencer.engine.backend import SequenceEntry, SequenceSpec
from qdesignoptimizer.sequencer.engine.job import parse_completed
from qdesignoptimizer.sequencer.engine.lease import release_lease, take_lease
from qdesignoptimizer.sequencer.engine.runner import Runner
from qdesignoptimizer.sequencer.engine.sequence import SequenceError, load_sequence

from .helpers import make_project, set_task

ADAPTER = (
    "from qdesignoptimizer.sequencer.backends.fake import FakeBackend\n"
    "from pathlib import Path\n"
    "def make_backend():\n"
    "    return FakeBackend(Path(__file__).parent)\n"
)


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_module, "WAIT_S", 0.02)
    return make_project(tmp_path / "p")


def _task(project):
    return project.backend.catalog().tasks[0].key


def _queue(project, n=3):
    return [project.add(_task(project)) for _ in range(n)]


def _ran(project) -> list[str]:
    """The studies finished, in the order they finished."""
    finished = [
        p
        for p in project.queue.completed()
        if st.result_of(p).get("status") in (st.DONE, st.FAILED)
    ]
    return [
        parse_completed(p.name)[0]
        for p in sorted(finished, key=lambda p: (p / "result.json").stat().st_mtime_ns)
    ]


def _run_in_background(project):
    done = []
    thread = threading.Thread(target=lambda: done.append(Runner(project).run()))
    thread.start()
    return thread, done


def _wait_for(condition, seconds=10.0):
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.02)


# -- being edited ---------------------------------------------------------------


def test_a_study_being_edited_is_passed_over_then_run(project):
    a, b, c = _queue(project)
    take_lease(project.queue.folder(a), "you (GUI)", seconds=60)
    thread, done = _run_in_background(project)
    _wait_for(lambda: len(_ran(project)) == 2)  # b and c, not a
    assert _ran(project) == [b, c]
    assert project.queue.ids() == [a]
    assert "waiting for edits" in st.RunnerState.load(project.out).message
    release_lease(project.queue.folder(a), "you (GUI)")
    thread.join(10)
    assert done and done[0].status == st.FINISHED
    assert _ran(project) == [b, c, a]


def test_stop_ends_a_wait(project):
    (a,) = _queue(project, 1)
    take_lease(project.queue.folder(a), "claude", seconds=60)
    thread, done = _run_in_background(project)
    _wait_for(lambda: "being edited" in st.RunnerState.load(project.out).message)
    st.request_stop(project.out)
    thread.join(10)
    assert done[0].status == st.STOPPED
    assert project.queue.ids() == [a]


def test_an_expired_lease_does_not_hold_the_runner(project):
    a, b = _queue(project, 2)
    take_lease(project.queue.folder(a), "claude", seconds=-1)  # long gone
    assert Runner(project).run().status == st.FINISHED
    assert _ran(project) == [a, b]


# -- needs ------------------------------------------------------------------------


def test_a_study_waits_for_what_it_needs_and_others_move_up(project):
    """b needs c, which is queued after it: c runs first, then b."""
    a = project.add(_task(project))
    c_id_later = []
    b = project.add(_task(project), needs=["placeholder"])
    c = project.add(_task(project))
    c_id_later.append(c)
    bridge = Bridge(project=project)
    bridge.save_job(b, {"needs": [c]})
    assert Runner(project).run().status == st.FINISHED
    assert _ran(project) == [a, c, b]


def test_a_study_whose_need_is_edited_waits_for_the_edit(project):
    a = project.add(_task(project))
    b = project.add(_task(project), needs=[a])
    take_lease(project.queue.folder(a), "claude", seconds=60)
    thread, done = _run_in_background(project)
    _wait_for(lambda: "waiting for edits" in st.RunnerState.load(project.out).message)
    time.sleep(0.1)
    assert _ran(project) == []  # b did not run before a
    release_lease(project.queue.folder(a), "claude")
    thread.join(10)
    assert done[0].status == st.FINISHED
    assert _ran(project) == [a, b]


def test_a_need_on_hold_lets_the_rest_run_then_says_so(project):
    held = project.add(_task(project), hold=True)
    b = project.add(_task(project), needs=[held])
    c = project.add(_task(project))
    final = Runner(project).run()
    assert final.status == st.FINISHED
    assert _ran(project) == [c]
    assert project.queue.ids() == [b]
    assert f"{b} needs {held} (hold)" in final.message
    # Queue the held one: both run.
    project.unpark(held)
    assert Runner(project).run().status == st.FINISHED
    assert _ran(project) == [c, held, b]


def test_a_failed_need_blocks_until_requeued(project):
    a = project.add(_task(project), on_failure="continue")
    set_task(project, a, fail=True)
    b = project.add(_task(project), needs=[a])
    final = Runner(project).run()
    assert _ran(project) == [a]  # failed; b not run
    assert f"{b} needs {a} (failed)" in final.message
    assert project.problems(b) == [f"needs {a}, whose runs failed; requeue it"]
    # A requeued copy (a_r2) counts once it succeeds.
    (failed,) = (p for p in project.queue.completed())
    copy = project.queue.requeue(failed, project.words)
    set_task(project, copy, fail=False)
    assert project.need_state(a) == "queued"
    assert Runner(project).run().status == st.FINISHED
    assert _ran(project)[-2:] == [copy, b]


def test_a_missing_need_is_a_problem(project):
    b = project.add(_task(project), needs=["no-such-study"])
    assert project.problems(b) == [
        "needs no-such-study, which is not queued, on hold or completed"
    ]
    (row,) = Bridge(project=project).snapshot()["rows"]
    assert row["waits_for"] == "no-such-study (missing)"


def test_the_queue_shows_what_a_study_waits_for(project):
    a = project.add(_task(project))
    b = project.add(_task(project), needs=[a])
    snap = Bridge(project=project).snapshot()
    rows = {r["id"]: r for r in snap["rows"]}
    assert rows[b]["waits_for"] == f"{a} (queued)"
    assert rows[a]["waits_for"] == ""
    seen = str(layout.queue_table(snap, {}))
    assert f"waits for {a} (queued)" in seen


def test_a_sequence_fills_in_what_its_studies_need(project):
    task = _task(project)
    spec = SequenceSpec(
        "tune",
        "Tune",
        (
            SequenceEntry(task, save=True),
            SequenceEntry(task, save=False),
            SequenceEntry(task, save=True),
            SequenceEntry(task, save=True, needs=(2,)),
            SequenceEntry(task, save=True, needs=()),
        ),
    )
    i = project.add_sequence(spec)
    assert [project.job(x).needs for x in i] == [
        [],
        [i[0]],  # every earlier study that writes back
        [i[0]],
        [i[1]],  # as the entry says
        [],
    ]


def test_a_sequence_file_names_needs_by_entry_number(project, tmp_path):
    task = _task(project)
    path = tmp_path / "s.yaml"
    path.write_text(
        f"version: 1\nstudies:\n- {project.words.task}: {task}\n- {project.words.task}: {task}\n  needs: [1]\n",
        encoding="utf-8",
    )
    spec = load_sequence(path, project.words)
    assert [e.needs for e in spec.entries] == [None, (1,)]
    path.write_text(
        f"version: 1\nstudies:\n- {project.words.task}: {task}\n  needs: [1]\n",
        encoding="utf-8",
    )
    with pytest.raises(SequenceError, match="earlier entries by number"):
        load_sequence(path, project.words)


def test_the_gui_form_and_the_command_line_set_needs(project, capsys, monkeypatch):
    a = project.add(_task(project))
    b = project.add(_task(project))
    bridge = Bridge(project=project)
    bridge.save_job(b, {"needs": [a]})
    assert project.job(b).needs == [a]
    assert bridge.details("queue", b)["job"]["needs"] == [a]
    with pytest.raises(Exception, match="cannot need itself"):
        bridge.save_job(b, {"needs": [b]})
    adapter = project.root / "sequencer.py"
    adapter.write_text(ADAPTER, "utf-8")
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert cli.main([str(adapter), "add", _task(project), "--needs", a]) == 0
    added = capsys.readouterr().out.split("queued ")[1].strip()
    assert project.job(added).needs == [a]
