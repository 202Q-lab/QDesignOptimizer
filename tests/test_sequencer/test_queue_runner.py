"""The folder queue and the runner, on the fake backend."""

from __future__ import annotations

import json
import os

import pytest
import yaml

from qdesignoptimizer.sequencer.engine import state as st
from qdesignoptimizer.sequencer.engine.cli import main as cli
from qdesignoptimizer.sequencer.engine.job import JobError, parse_completed, parse_job
from qdesignoptimizer.sequencer.engine.plan import PlanError, load_plan
from qdesignoptimizer.sequencer.engine.queue import QueueError
from qdesignoptimizer.sequencer.engine.runner import Runner, RunRefused

from .helpers import make_project, set_task, store


@pytest.fixture
def project(tmp_path):
    return make_project(tmp_path)


def result(path):
    return json.loads((path / "result.json").read_text("utf-8"))


# -- queue --------------------------------------------------------------


def test_add_writes_a_self_contained_folder(project):
    job_id = project.add("A.calibrate", save=True)
    folder = project.queue.folder(job_id)
    assert job_id.startswith("x-") and job_id.endswith("-a_calibrate")
    assert (folder / "job.yaml").exists()
    assert (folder / "task.yaml").exists()
    job = project.job(job_id)
    assert job.task == "A.calibrate" and job.subject == "A" and job.save
    assert project.queue.ids() == [job_id]


def test_a_plan_is_tagged_and_numbered(project):
    spec = project.backend.catalog().plan("full_A")
    ids = project.add_plan(spec)
    assert ids == ["p01-full_A-a_calibrate", "p01-full_A-a_measure"]
    assert project.job(ids[0]).plan == {
        "id": "p01",
        "name": "full_A",
        "entry": 1,
    }
    assert project.add_plan(spec)[0].startswith("p02-full_A-")


def test_order_and_remove(project):
    a = project.add("A.calibrate")
    b = project.add("B.calibrate")
    project.queue.save_order([b, a])
    assert project.queue.ids() == [b, a]
    with pytest.raises(QueueError, match="changed"):
        project.queue.save_order([a])
    moved = project.queue.remove(b)
    assert moved.parent.name == "removed" and moved.exists()
    assert project.queue.ids() == [a]


def test_a_folder_from_before_the_rename_is_still_read(project):
    """``study.yaml`` was the queue entry's name until 2026-11-12."""
    job_id = project.add("A.calibrate")
    folder = project.queue.folder(job_id)
    (folder / "job.yaml").rename(folder / "study.yaml")
    assert project.queue.ids() == [job_id]
    assert project.job(job_id).task == "A.calibrate"


def test_a_hand_made_folder_joins_the_end(project):
    a = project.add("A.calibrate")
    folder = project.queue.folder(a)
    copy = folder.parent / "x-hand_made"
    copy.mkdir()
    for f in folder.iterdir():
        text = f.read_text("utf-8")
        if f.name == "job.yaml":
            text = text.replace(a, "x-hand_made")
        (copy / f.name).write_text(text, "utf-8")
    assert project.queue.ids() == [a, "x-hand_made"]
    assert project.problems("x-hand_made") == []


# -- running ------------------------------------------------------------


def test_a_study_runs_moves_and_writes_back(project):
    job_id = project.add("A.calibrate", save=True, checks=["converged"])
    set_task(project, job_id, value=5.0)
    final = Runner(project).run()
    assert final.status == st.FINISHED
    assert project.queue.ids() == []
    (done,) = project.queue.completed()
    # <id>__<start>: a name sort in Explorer groups by measurement plan
    assert done.name.startswith(f"{job_id}_20")
    rec = result(done)
    assert rec["status"] == "done" and rec["committed"] == {"A_value": 5.0}
    assert store(project)["A_value"] == 5.0
    # inputs/ as queued; the top level as it ran (filled in at start)
    assert (done / "inputs" / "task.yaml").exists()
    assert not (done / "inputs" / "design_variables.json").exists()
    ran = json.loads((done / "design_variables.json").read_text("utf-8"))
    assert ran == {"A_value": 0.0, "B_value": 0.0}


def test_a_later_study_starts_from_what_an_earlier_one_wrote(project):
    first = project.add("A.calibrate", save=True)
    second = project.add("A.measure")
    set_task(project, first, value=7.0)
    Runner(project).run()
    done = {parse_completed(p.name)[0]: p for p in project.queue.completed()}
    assert result(done[second])["outputs"]["started_from"] == 7.0


def test_values_set_in_the_folder_win_and_are_kept_for_requeue(project):
    job_id = project.add("A.measure")
    project.queue.write_file(
        job_id, "design_variables.json", json.dumps({"A_value": 3.0})
    )
    Runner(project).run()
    (done,) = project.queue.completed()
    assert result(done)["outputs"]["started_from"] == 3.0
    new_id = project.queue.requeue(done, project.words)
    assert new_id == f"{job_id}_r2"
    partial = json.loads(
        (project.queue.folder(new_id) / "design_variables.json").read_text("utf-8")
    )
    assert partial == {"A_value": 3.0}
    assert project.job(new_id).raw["requeued_from"] == done.name


def test_a_failed_study_is_marked_and_writes_nothing(project):
    bad = project.add("A.calibrate", save=True, on_failure="continue")
    after = project.add("B.calibrate")
    set_task(project, bad, value=9.0, fail=True)
    final = Runner(project).run()
    assert final.status == st.FINISHED
    names = {p.name for p in project.queue.completed()}
    assert any(n.startswith(f"{bad}_20") and n.endswith("_failed") for n in names)
    assert any(n.startswith(f"{after}_20") for n in names)
    assert store(project)["A_value"] == 0.0


def test_abort_stops_and_leaves_the_rest_queued(project):
    bad = project.add("A.calibrate")
    after = project.add("B.calibrate")
    set_task(project, bad, fail=True)
    final = Runner(project).run()
    assert final.status == st.ABORTED and bad in final.message
    assert project.queue.ids() == [after]


def test_a_failed_check_blocks_the_write_back(project):
    job_id = project.add(
        "A.calibrate", save=True, checks=["converged"], on_failure="continue"
    )
    set_task(project, job_id, value=4.0, converged=False)
    Runner(project).run()
    (done,) = project.queue.completed()
    rec = result(done)
    assert rec["status"] == "failed" and "did not converge" in rec["message"]
    assert rec["updates"] == {"A_value": 4.0} and rec["committed"] == {}
    assert store(project)["A_value"] == 0.0


def test_a_failed_write_back_fails_the_study(project):
    project.backend.fail_commit = True
    project.add("A.calibrate", save=True, on_failure="continue")
    Runner(project).run()
    (done,) = project.queue.completed()
    assert "write-back failed" in result(done)["message"]
    assert done.name.endswith("_failed")


def test_an_invalid_study_stays_queued_and_stops_the_runner(project):
    job_id = project.add("A.calibrate")
    set_task(project, job_id, colour="red")
    final = Runner(project).run()
    assert final.status == st.ERROR and "colour" in final.message
    assert project.queue.ids() == [job_id]
    assert project.queue.completed() == []


def test_the_queue_is_reread_before_every_study(project):
    project.add("A.calibrate")
    backend = project.backend
    original = backend.before_step

    def add_one(job):
        original(job)
        if job.task == "A.calibrate":
            project.add("B.calibrate")

    backend.before_step = add_one
    Runner(project).run()
    assert len(project.queue.completed()) == 2


def test_stop_ends_the_runner_between_studies(project):
    project.add("A.calibrate")
    later = project.add("B.calibrate")
    original = project.backend.before_step

    def stop(job):
        original(job)
        st.request_stop(project.out)

    project.backend.before_step = stop
    final = Runner(project).run()
    assert final.status == st.STOPPED
    assert project.queue.ids() == [later]
    assert not st.stop_requested(project.out)


def test_one_runner_at_a_time(project):
    lock = st.acquire_lock(project.out)
    try:
        with pytest.raises(RunRefused, match="already going"):
            Runner(project).run()
    finally:
        st.release_lock(lock)


def test_a_lock_left_by_a_dead_process_is_taken_over(project):
    project.out.mkdir(parents=True, exist_ok=True)
    (project.out / st.LOCK_FILE).write_text(
        json.dumps({"pid": 999999, "started": 1.0, "at": "then"})
    )
    project.add("A.calibrate")
    assert Runner(project).run().status == st.FINISHED


def test_runner_state_says_alive_while_it_runs(project):
    seen = {}
    first = project.add("A.calibrate")
    original = project.backend.before_step

    def look(job):
        original(job)
        state = st.RunnerState.load(project.out)
        seen["alive"], seen["current"] = (
            state.alive(project.out),
            state.current,
        )

    project.backend.before_step = look
    Runner(project).run()
    assert seen["alive"] is True and seen["current"].startswith(first + "_20")
    assert not st.RunnerState.load(project.out).alive(project.out)


# -- files --------------------------------------------------------------


def test_study_yaml_errors_are_collected():
    with pytest.raises(JobError) as err:
        parse_job(
            {
                "version": 1,
                "task": "A.calibrate",
                "save": "false",
                "on_failure": "retry",
                "colour": 1,
            }
        )
    text = "; ".join(err.value.problems)
    assert "true or false" in text and "retry" in text and "colour" in text


def test_plan_file(tmp_path, project):
    path = tmp_path / "night.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "name": "Night",
                "defaults": {"settings": {"seconds": 0}, "save": True},
                "studies": [
                    "A.calibrate",
                    {"task": "B.measure", "save": False},
                ],
            }
        )
    )
    spec = load_plan(path)
    assert [e.task for e in spec.entries] == ["A.calibrate", "B.measure"]
    assert spec.entries[1].save is False
    ids = project.add_plan(spec)
    assert project.job(ids[0]).save and not project.job(ids[1]).save
    path.write_text(
        yaml.safe_dump(
            {"version": 1, "studies": [{"task": "A.calibrate", "colour": 1}]}
        )
    )
    with pytest.raises(PlanError, match="colour"):
        load_plan(path)


def test_a_failing_generator_queues_nothing(project):
    spec = project.backend.catalog().plan("full_A")
    original = project.backend.generate

    def gen(task):
        if task == "A.measure":
            raise RuntimeError("generator broke")
        return original(task)

    project.backend.generate = gen
    with pytest.raises(QueueError, match="generator broke"):
        project.add_plan(spec)
    assert project.queue.ids() == []


# -- CLI ----------------------------------------------------------------


def test_cli(tmp_path, capsys):
    catalog = tmp_path / "catalog.py"
    catalog.write_text(
        "from pathlib import Path\n"
        "from qdesignoptimizer.sequencer.backends.fake import FakeBackend\n"
        "def make_backend():\n"
        "    return FakeBackend(Path(__file__).parent)\n"
    )
    assert cli([str(catalog), "add", "A.calibrate"]) == 0
    assert cli([str(catalog), "plan", "full_A"]) == 0
    assert cli([str(catalog), "check"]) == 0
    assert cli([str(catalog), "run"]) == 1
    assert "--yes" in capsys.readouterr().err
    assert cli([str(catalog), "run", "--yes"]) == 0
    assert cli([str(catalog), "list"]) == 0
    out = capsys.readouterr().out
    assert "queue (0)" in out and "done" in out


# -- review fixes (2026-11-10) -----------------------------------------


def test_a_single_check_name_in_a_plan_is_not_split(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "defaults": {"checks": "converged"},
                "studies": ["A.calibrate"],
            }
        )
    )
    assert load_plan(path).entries[0].checks == ("converged",)


def test_an_unknown_check_is_caught_before_the_run(project):
    job_id = project.add("A.calibrate", checks=["convergd"])
    assert any("convergd" in p for p in project.problems(job_id))
    final = Runner(project).run()
    assert final.status == st.ERROR
    assert project.queue.ids() == [job_id]


def test_a_claimed_study_cannot_be_edited_and_comes_back(project):
    job_id = project.add("A.calibrate")
    claimed = project.queue.claim(job_id)
    assert project.queue.ids() == []
    with pytest.raises(QueueError):
        project.queue.write_file(job_id, "task.yaml", "value: 3\n")
    project.queue.release(claimed, job_id)
    assert project.queue.ids() == [job_id]


def test_an_invalid_study_is_released_back_to_its_place(project):
    first = project.add("A.calibrate")
    second = project.add("B.calibrate")
    set_task(project, first, colour=1)
    Runner(project).run()
    assert project.queue.ids() == [first, second]


def test_ids_are_not_reused_after_a_remove(project):
    spec = project.backend.catalog().plan("full_A")
    for job_id in project.add_plan(spec):
        project.queue.remove(job_id)
    assert project.add_plan(spec)[0].startswith("p02-")


def test_the_lock_is_the_os_lock_not_the_file(project):
    """A lock file left behind (empty, garbage, a crashed runner's) is
    never held: only a live process's OS lock counts."""
    project.out.mkdir(parents=True, exist_ok=True)
    (project.out / st.LOCK_FILE).write_text("")
    assert not st.lock_held(project.out)
    lock = st.acquire_lock(project.out)
    assert st.lock_held(project.out)
    with pytest.raises(st.LockHeld):
        st.acquire_lock(project.out)
    st.release_lock(lock)
    assert not st.lock_held(project.out)


def test_a_dead_runner_reads_as_dead(project):
    st.RunnerState(status=st.BUSY, pid=os.getpid()).save(project.out)
    assert not st.RunnerState.load(project.out).alive(project.out)


def test_requeue_keeps_a_task_slug_that_ends_in_r2(project):
    job_id = project.add("A.calibrate")
    folder = project.queue.folder(job_id)
    renamed = folder.with_name("x-0000_readout_r2")
    folder.rename(renamed)
    text = (
        (renamed / "job.yaml").read_text("utf-8").replace(job_id, "x-0000_readout_r2")
    )
    (renamed / "job.yaml").write_text(text, "utf-8")
    Runner(project).run()
    (done,) = project.queue.completed()
    assert project.queue.requeue(done, project.words) == "x-0000_readout_r2_r2"


def test_a_check_that_crashes_releases_the_study(project):
    """Review 2026-11-10: an exception between claim and release left
    the study hidden as .claimed-<id> for good."""
    job_id = project.add("A.calibrate")
    original = project.backend.check_folder

    def boom(folder, job, deep):
        if deep:
            raise AttributeError("list has no get")
        return original(folder, job, deep)

    project.backend.check_folder = boom
    final = Runner(project).run()
    assert final.status == st.ERROR and "cannot be checked" in final.message
    assert project.queue.ids() == [job_id]
    assert not any(
        p.name.startswith(".claimed-") for p in project.queue.queue_dir.iterdir()
    )


def test_a_dead_runners_claim_is_recovered(project):
    job_id = project.add("A.calibrate")
    project.queue.claim(job_id)
    assert project.queue.ids() == []
    final = Runner(project).run()
    assert final.status == st.FINISHED
    assert len(project.queue.completed()) == 1


def test_a_study_cut_off_with_its_runner_is_closed(project):
    project.add("A.calibrate")
    project.add("B.calibrate")
    first = project.queue.ids()[0]
    claimed = project.queue.claim(first)
    folder = project.queue.start(claimed, first)
    st.write_json(folder / st.RESULT_FILE, {"id": first, "status": st.RUNNING})
    Runner(project).run()
    names = [p.name for p in project.queue.completed()]
    cut = next(n for n in names if first in n)
    assert cut.endswith("_failed")
    rec = json.loads(
        (project.queue.completed_dir / cut / "result.json").read_text("utf-8")
    )
    assert rec["status"] == "failed" and "interrupted" in rec["message"]


def _cut_off_first(project, on_failure):
    first = project.add("A.calibrate", on_failure=on_failure)
    second = project.add("B.calibrate")
    claimed = project.queue.claim(first)
    folder = project.queue.start(claimed, first)
    st.write_json(folder / st.RESULT_FILE, {"id": first, "status": st.RUNNING})
    return first, second


def test_a_crash_under_on_failure_abort_stops_the_queue(project):
    first, second = _cut_off_first(project, "abort")
    final = Runner(project).run()
    assert final.status == st.ABORTED
    assert first in final.message and "on_failure: abort" in final.message
    assert project.queue.ids() == [second]
    # Started again, the runner goes on: the cut-off study is closed.
    assert Runner(project).run().status == st.FINISHED
    assert project.queue.ids() == []


def test_a_crash_under_on_failure_continue_runs_the_rest(project):
    _, second = _cut_off_first(project, "continue")
    assert Runner(project).run().status == st.FINISHED
    assert project.queue.ids() == []
    assert any(second in p.name for p in project.queue.completed())


def test_the_store_write_retries_while_the_file_is_held(tmp_path, monkeypatch):
    from qdesignoptimizer.sequencer.backends import _store

    path = tmp_path / "dv.json"
    path.write_text(json.dumps({"L": "1um"}))
    real, calls = os.replace, []

    def flaky(src, dst):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError("held by a reader")
        real(src, dst)

    monkeypatch.setattr(_store.os, "replace", flaky)
    monkeypatch.setattr(_store.time, "sleep", lambda s: None)
    assert _store.update_json(path, {"L": "2um"}) == {"L": "2um"}
    assert json.loads(path.read_text()) == {"L": "2um"}

    monkeypatch.setattr(
        _store.os,
        "replace",
        lambda s, d: (_ for _ in ()).throw(PermissionError("held")),
    )
    with pytest.raises(PermissionError):
        _store.update_json(path, {"L": "3um"})
    assert json.loads(path.read_text()) == {"L": "2um"}
    assert [p.name for p in tmp_path.iterdir()] == ["dv.json"]


# -- review round 3 (2026-11-10) -----------------------------------------


def test_yaml_number_strings_become_numbers_in_settings(tmp_path):
    from qdesignoptimizer.sequencer.engine.job import parse_job

    job = parse_job(
        yaml.safe_load(
            "version: 1\ntask: A.calibrate\nsettings: {delta_f: 1e-3, n: '15'}\n"
        )
    )
    assert job.settings == {"delta_f": 0.001, "n": 15}
    path = tmp_path / "c.yaml"
    path.write_text(
        "version: 1\ndefaults: {settings: {delta_f: 1e-3}}\n" "studies: [A.calibrate]\n"
    )
    assert load_plan(path).entries[0].settings == {"delta_f": 0.001}


def test_any_malformed_study_yaml_is_a_job_error():
    from qdesignoptimizer.sequencer.engine.job import (
        JobError,
        parse_completed,
        parse_job,
    )

    for bad in (
        {"task": "A", "checks": 5},
        {"task": "A", "notes": [1]},
        {"task": "A", "checks": [1, 2]},
    ):
        with pytest.raises(JobError):
            parse_job(bad)


def test_a_removed_study_is_not_claimed(project):
    job_id = project.add("A.calibrate")
    project.queue.folder(job_id).rename(project.out / "elsewhere")
    with pytest.raises(QueueError):
        project.queue.claim(job_id)


def test_ctrl_c_is_recorded_and_the_study_closed_next_time(project):
    project.add("A.calibrate")
    original = project.backend.run

    def interrupt(folder, job):
        raise KeyboardInterrupt

    project.backend.run = interrupt
    with pytest.raises(KeyboardInterrupt):
        Runner(project).run()
    assert st.RunnerState.load(project.out).status == st.ERROR
    project.backend.run = original
    Runner(project).run()
    (done,) = project.queue.completed()
    assert done.name.endswith("_failed")
    assert (
        "interrupted"
        in json.loads((done / "result.json").read_text("utf-8"))["message"]
    )


def test_a_write_back_that_completed_is_recovered_as_done(project):
    job_id = project.add("A.calibrate", save=True)
    claimed = project.queue.claim(job_id)
    folder = project.queue.start(claimed, job_id)
    project.backend.commit({"A_value": 6.0})
    st.write_json(
        folder / st.RESULT_FILE,
        {"id": job_id, "status": st.COMMITTING, "updates": {"A_value": 6.0}},
    )
    other = project.add("B.calibrate", save=True)
    claimed = project.queue.claim(other)
    folder2 = project.queue.start(claimed, other)
    st.write_json(
        folder2 / st.RESULT_FILE,
        {"id": other, "status": st.COMMITTING, "updates": {"B_value": 9.0}},
    )
    Runner(project).run()
    by_id = {
        json.loads((p / "result.json").read_text("utf-8"))["id"]: p
        for p in project.queue.completed()
    }
    first = json.loads((by_id[job_id] / "result.json").read_text("utf-8"))
    assert first["status"] == "done" and "recovered" in first["message"]
    second = json.loads((by_id[other] / "result.json").read_text("utf-8"))
    assert second["status"] == "failed"
    assert by_id[other].name.endswith("_failed")


def test_a_study_started_without_a_result_is_closed(project):
    job_id = project.add("A.calibrate")
    claimed = project.queue.claim(job_id)
    project.queue.start(claimed, job_id)
    Runner(project).run()
    (done,) = project.queue.completed()
    assert done.name.endswith("_failed")


def test_the_lock_waits_out_a_brief_check(project):
    import threading

    lock = st.acquire_lock(project.out)
    threading.Timer(0.3, st.release_lock, args=(lock,)).start()
    st.release_lock(st.acquire_lock(project.out, wait=2.0))


# -- hold (2026-11-11) ------------------------------------------------------


def test_a_held_study_stops_the_runner_and_keeps_its_place(project):
    first = project.add("A.calibrate")
    held = project.add("B.calibrate")
    after = project.add("A.measure")
    project.set_hold(held, True)
    final = Runner(project).run()
    assert final.status == st.HELD and held in final.message
    assert project.queue.ids() == [held, after]
    assert len(project.queue.completed()) == 1
    project.set_hold(held, False)
    assert Runner(project).run().status == st.FINISHED
    assert project.queue.ids() == []
    assert first


def test_hold_is_written_only_when_on(project):
    job_id = project.add("A.calibrate")
    path = project.queue.folder(job_id) / "job.yaml"
    assert "hold" not in path.read_text("utf-8")
    project.set_hold(job_id, True)
    assert "hold: true" in path.read_text("utf-8")
    project.set_hold(job_id, False)
    assert "hold" not in path.read_text("utf-8")


def test_cli_hold_and_release(tmp_path, capsys):
    catalog = tmp_path / "sequencer.py"
    catalog.write_text(
        "from pathlib import Path\n"
        "from qdesignoptimizer.sequencer.backends.fake import FakeBackend\n"
        "def make_backend():\n"
        "    return FakeBackend(Path(__file__).parent)\n"
    )
    assert cli([str(catalog), "add", "A.calibrate"]) == 0
    job_id = capsys.readouterr().out.split()[-1]
    assert cli([str(catalog), "hold", job_id]) == 0
    assert cli([str(catalog), "run", "--yes"]) == 0
    assert "held" in capsys.readouterr().out
    assert cli([str(catalog), "release", job_id]) == 0


def test_completed_names_old_and_new_are_both_read(project):
    stamp = "2026-11-12T09-00-02"
    assert parse_completed(f"p01-tune_ATS1-ats1_kappa_{stamp}") == (
        "p01-tune_ATS1-ats1_kappa",
        stamp,
    )
    # a task whose name ends in "failed" is not confused with the suffix
    assert parse_completed(f"x-a7f3-t_failed_{stamp}_failed") == (
        "x-a7f3-t_failed",
        stamp,
    )
    assert parse_completed(f"x-a7f3-t_{stamp}-2") == (
        "x-a7f3-t",
        f"{stamp}-2",
    )
    assert parse_completed(f"ministudy_{stamp}_p01-01_a_b_failed") == (
        "p01-01_a_b",
        stamp,
    )
    # A run from before the rename is still listed, and requeues.
    job_id = project.add("A.calibrate")
    Runner(project).run()
    (done,) = project.queue.completed()
    old = done.with_name(f"ministudy_2026-11-10T08-00-00_{job_id}")
    done.rename(old)
    assert project.queue.completed() == [old]
    assert job_id in project.queue.taken()
    assert project.queue.requeue(old, project.words).startswith(job_id)


def test_a_name_sort_groups_a_plan(project):
    spec = project.backend.catalog().plan("full_A")
    project.add("B.measure")
    project.add_plan(spec)
    Runner(project).run()
    names = sorted(p.name for p in project.queue.completed())
    ids = [parse_completed(n)[0] for n in names]
    assert ids[:2] == [
        "p01-full_A-a_calibrate",
        "p01-full_A-a_measure",
    ]
    assert ids[2].startswith("x-") and ids[2].endswith("-b_measure")
