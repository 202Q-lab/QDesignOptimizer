"""Review round 3 (2026-11-14): engine and backend robustness."""

from __future__ import annotations

import json

import pytest

from qdesignoptimizer.sequencer.engine import project as project_module
from qdesignoptimizer.sequencer.engine import state as st
from qdesignoptimizer.sequencer.engine.queue import QueueError
from qdesignoptimizer.sequencer.engine.runner import Runner
from qdesignoptimizer.sequencer.engine.sequence import load_sequence_dir
from qdesignoptimizer.sequencer.engine.workspace import Workspace

from .helpers import make_project


@pytest.fixture
def project(tmp_path):
    return make_project(tmp_path / "proj")


def _cut_off(project, result_text):
    job_id = project.add("A.calibrate")
    claimed = project.queue.claim(job_id)
    folder = project.queue.start(claimed, job_id)
    (folder / st.RESULT_FILE).write_text(result_text, encoding="utf-8")
    return job_id, folder


def test_a_damaged_result_file_is_closed_as_cut_off(project):
    """An empty result.json (a power cut mid-write) neither stops the
    runner for good nor every add after it."""
    job_id, _ = _cut_off(project, "")
    project.add("B.calibrate")  # new ids can still be given out
    Runner(project).run()
    (closed,) = (p for p in project.queue.completed() if job_id in p.name)
    assert closed.name.endswith("_failed")
    record = json.loads((closed / st.RESULT_FILE).read_text("utf-8"))
    assert record["id"] == job_id and record["status"] == st.FAILED


def test_a_cut_off_study_without_a_result_keeps_its_id(project):
    job_id = project.add("A.calibrate")
    claimed = project.queue.claim(job_id)
    project.queue.start(claimed, job_id)  # died before result.json
    final = Runner(project).run()
    (closed,) = project.queue.completed()
    record = json.loads((closed / st.RESULT_FILE).read_text("utf-8"))
    assert record["id"] == job_id
    assert final.message.startswith(job_id)


@pytest.mark.parametrize(
    "text",
    [
        "defaults: [save]\nstudies: [A.calibrate]\n",
        "studies: 5\n",
        "studies:\n- {task: A.calibrate, settings: [1]}\n",
        "defaults: {settings: [1]}\nstudies: [A.calibrate]\n",
        "studies:\n- {task: A.calibrate, checks: {a: 1}}\n",
    ],
)
def test_a_malformed_sequence_file_is_one_problem_not_a_crash(tmp_path, text):
    (tmp_path / "bad.yaml").write_text(text, encoding="utf-8")
    (tmp_path / "good.yaml").write_text("studies: [A.calibrate]\n", encoding="utf-8")
    specs, problems = load_sequence_dir(tmp_path)
    assert [s.key for s in specs] == ["good"]
    assert problems and all(p.startswith("bad.yaml") for p in problems)


def test_deleting_a_file_of_a_claimed_study_is_refused(project):
    job_id = project.add("A.calibrate")
    project.queue.claim(job_id)
    with pytest.raises(QueueError, match="not queued"):
        project.queue.delete_file(job_id, "task.yaml")


def test_a_queued_sequence_skips_a_number_a_draft_keeps(project, monkeypatch):
    """A draft whose launch published nothing keeps s01; a sequence
    queued meanwhile takes s02, not s01."""
    name = project.new_draft("three")
    project.draft_add(name, "A.calibrate")
    real = project_module._rename
    calls = {"n": 0}

    def rename(source, dest, tries=50):
        calls["n"] += 1
        if calls["n"] == 2:  # the first publish
            raise PermissionError("held open")
        return real(source, dest, tries)

    monkeypatch.setattr(project_module, "_rename", rename)
    with pytest.raises(QueueError, match="as s01"):
        project.launch_draft(name)
    monkeypatch.undo()
    spec = project.backend.catalog().sequence("full_A")
    assert project.add_sequence(spec)[0].startswith("s02-")


def test_a_sequence_that_fails_half_way_queues_nothing(project, monkeypatch):
    spec = project.backend.catalog().sequence("full_A")
    real_add, calls = project.queue.add, {"n": 0}

    def add(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("disk full")
        return real_add(*args, **kwargs)

    monkeypatch.setattr(project.queue, "add", add)
    with pytest.raises(OSError):
        project.add_sequence(spec)
    assert project.queue.ids() == []


def test_project_names_do_not_depend_on_opening_order(tmp_path):
    a = make_project(tmp_path / "x" / "chip")
    b = make_project(tmp_path / "y" / "chip")
    one = Workspace([a, b], state_dir=tmp_path / "s1")
    two = Workspace([b, a], state_dir=tmp_path / "s2")
    assert one.projects["chip"] is a and two.projects["chip"] is a
    assert one.projects["chip-2"] is b and two.projects["chip-2"] is b


def test_a_draft_from_a_sequence_is_recorded_under_its_key(project):
    """Launched from the GUI unchanged, a sequence gets the ids and the
    name the CLI's ``sequence`` gives it, not its long label."""
    spec = project.backend.catalog().sequence("full_A")
    name = project.new_draft(from_spec=spec)
    ids = project.launch_draft(name)
    assert ids[0].startswith("s01-full_A-")
    assert project.job(ids[0]).sequence["name"] == "full_A"
