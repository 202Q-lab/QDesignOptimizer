"""The runner and studies being edited (2026-11-16): one someone is
editing is passed over and run once the edit is done -- unless the study
after it needs it, then the runner waits for it."""

from __future__ import annotations

import threading
import time

import pytest

from qdesignoptimizer.sequencer.app.queue_state import Bridge
from qdesignoptimizer.sequencer.engine import runner as runner_module
from qdesignoptimizer.sequencer.engine import state as st
from qdesignoptimizer.sequencer.engine.backend import SequenceEntry, SequenceSpec
from qdesignoptimizer.sequencer.engine.job import parse_completed
from qdesignoptimizer.sequencer.engine.lease import release_lease, take_lease
from qdesignoptimizer.sequencer.engine.runner import Runner

from .helpers import make_project


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_module, "WAIT_S", 0.02)
    return make_project(tmp_path / "p")


def _queue(project, n=3, **fields):
    task = project.backend.catalog().tasks[0].key
    ids = [project.add(task) for _ in range(n)]
    for job_id, value in fields.items():
        _set(project, job_id, value)
    return ids


def _set(project, job_id, needed):
    folder = project.queue.folder(job_id)
    text = (folder / "job.yaml").read_text(encoding="utf-8")
    (folder / "job.yaml").write_text(
        text + f"needed_by_next: {str(needed).lower()}\n", encoding="utf-8"
    )


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


def test_a_study_being_edited_is_passed_over_then_run(project):
    a, b, c = _queue(project)
    take_lease(project.queue.folder(a), "you (GUI)", seconds=60)
    thread, done = _run_in_background(project)
    _wait_for(lambda: len(_ran(project)) == 2)  # b and c, not a
    assert _ran(project) == [b, c]
    assert project.queue.ids() == [a]
    state = st.RunnerState.load(project.out)
    assert state.message == "waiting: every queued study is being edited"
    release_lease(project.queue.folder(a), "you (GUI)")
    thread.join(10)
    assert done and done[0].status == st.FINISHED
    assert _ran(project) == [b, c, a]


def test_a_study_needed_by_the_next_is_waited_for(project):
    a, b = _queue(project, 2)
    _set(project, a, True)
    take_lease(project.queue.folder(a), "claude", seconds=60)
    thread, done = _run_in_background(project)
    _wait_for(
        lambda: "the study after it needs it"
        in st.RunnerState.load(project.out).message
    )
    time.sleep(0.1)
    assert _ran(project) == []  # b did not run past it
    release_lease(project.queue.folder(a), "claude")
    thread.join(10)
    assert done[0].status == st.FINISHED
    assert _ran(project) == [a, b]


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


def test_a_sequence_marks_what_its_later_studies_need(project):
    task = project.backend.catalog().tasks[0].key
    spec = SequenceSpec(
        "tune",
        "Tune",
        (
            SequenceEntry(task, save=True),
            SequenceEntry(task, save=False),
            SequenceEntry(task, save=True, needed_by_next=False),
            SequenceEntry(task, save=True),
        ),
    )
    ids = project.add_sequence(spec)
    assert [project.job(i).needed_by_next for i in ids] == [True, False, False, False]


def test_the_gui_form_saves_the_mark(project):
    (a,) = _queue(project, 1)
    bridge = Bridge(project=project)
    bridge.save_job(a, {"needed_by_next": True})
    assert project.job(a).needed_by_next is True
    assert bridge.details("queue", a)["job"]["needed_by_next"] is True
    bridge.save_job(a, {"needed_by_next": False})
    assert project.job(a).needed_by_next is False
