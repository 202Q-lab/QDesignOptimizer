"""Regressions for the findings of the review of drafts, write records
and studies made from scratch."""

from __future__ import annotations

import json

import pytest

from qdesignoptimizer.sequencer.app.queue_state import Bridge, GuiError
from qdesignoptimizer.sequencer.backends._store import (
    read_writes,
    sources_text,
    writes_path,
)
from qdesignoptimizer.sequencer.engine import project as project_module
from qdesignoptimizer.sequencer.engine import state as st
from qdesignoptimizer.sequencer.engine.project import STAGE
from qdesignoptimizer.sequencer.engine.queue import QueueError
from qdesignoptimizer.sequencer.engine.runner import Runner
from qdesignoptimizer.sequencer.engine.workspace import Workspace

from .helpers import make_project, set_task


@pytest.fixture
def project(tmp_path):
    return make_project(tmp_path)


def _draft(project, tasks=("A.calibrate", "A.measure", "B.measure")):
    name = project.new_draft("three")
    for task in tasks:
        project.draft_add(name, task)
    return name


def _failing_rename(monkeypatch, fail_on, error=PermissionError):
    """``_rename`` in project.py, failing on its ``fail_on``-th call."""
    real = project_module._rename
    calls = {"n": 0}

    def rename(source, dest, tries=50):
        calls["n"] += 1
        if calls["n"] == fail_on:
            raise error("held open")
        return real(source, dest, tries)

    monkeypatch.setattr(project_module, "_rename", rename)


def _staged(project):
    return [
        p.name for p in project.queue.queue_dir.iterdir() if p.name.startswith(STAGE)
    ]


# -- 1. a launch never strands a study -----------------------------------


def test_a_launch_failing_while_staging_puts_everything_back(project, monkeypatch):
    name = _draft(project)
    before = sorted(project.draft_queue(name).ids())
    _failing_rename(monkeypatch, fail_on=2)  # the second study's move
    with pytest.raises(QueueError, match="back in the draft"):
        project.launch_draft(name)
    assert project.queue.ids() == [] and _staged(project) == []
    draft = project.draft_queue(name)
    assert sorted(draft.ids()) == before
    for i in draft.ids():  # as they were: no campaign written into them
        assert project.problems(i, folder=draft.folder(i)) == []
        assert "campaign" not in (draft.folder(i) / "job.yaml").read_text()


def test_a_launch_failing_while_publishing_is_finished_by_a_relaunch(
    project, monkeypatch
):
    name = _draft(project)
    _failing_rename(monkeypatch, fail_on=3 + 2)  # 3 stagings, 2nd publish
    with pytest.raises(QueueError, match="launched 1 of 3 as c01"):
        project.launch_draft(name)
    assert project.queue.ids() == ["c01-three-a_calibrate"]
    assert _staged(project) == []
    assert len(project.draft_queue(name).ids()) == 2
    monkeypatch.undo()
    moved = project.launch_draft(name)
    assert moved == ["c01-three-a_measure", "c01-three-b_measure"]
    assert project.queue.ids() == ["c01-three-a_calibrate"] + moved
    assert project.drafts() == []


def test_a_launch_cut_off_while_publishing_is_finished_by_the_runner(
    project, monkeypatch
):
    name = _draft(project)
    _failing_rename(monkeypatch, fail_on=3 + 1, error=KeyboardInterrupt)
    with pytest.raises(KeyboardInterrupt):
        project.launch_draft(name)
    # One stage folder per launch, holding all three.
    assert project.queue.ids() == [] and len(_staged(project)) == 1
    monkeypatch.undo()
    final = Runner(project).run()
    assert len(final.done) == 3 and project.drafts() == []
    assert _staged(project) == []


def test_a_launch_cut_off_while_staging_is_undone_by_the_runner(project, monkeypatch):
    """A hard kill mid-staging: no undo runs, no record is written by
    hand -- the stage folder alone is enough for the runner to undo it."""
    name = _draft(project)
    monkeypatch.setattr(project, "_unstage", lambda *a, **k: None)
    _failing_rename(monkeypatch, fail_on=2, error=KeyboardInterrupt)
    with pytest.raises(KeyboardInterrupt):
        project.launch_draft(name)
    monkeypatch.undo()
    assert _staged(project) and len(project.draft_queue(name).ids()) == 2
    Runner(project).run()  # recovers before it looks at the queue
    assert _staged(project) == [] and project.queue.ids() == []
    assert len(project.draft_queue(name).ids()) == 3


def test_a_relaunch_after_a_crash_launches_every_study(project, monkeypatch):
    name = _draft(project)
    monkeypatch.setattr(project, "_unstage", lambda *a, **k: None)
    _failing_rename(monkeypatch, fail_on=2, error=KeyboardInterrupt)
    with pytest.raises(KeyboardInterrupt):
        project.launch_draft(name)
    monkeypatch.undo()
    project.draft_add(name, "B.calibrate")  # added after the crash
    moved = project.launch_draft(name)  # recovers first
    assert len(moved) == 4 and _staged(project) == []
    assert project.drafts() == []


def test_a_study_added_during_a_launch_is_refused_not_lost(project, monkeypatch):
    """The launch holds the lock while it checks; an add then is told to
    wait, and the launch goes on with what it staged."""
    name = _draft(project, tasks=("A.calibrate",))
    real = project.problems
    refused = []

    def problems(job_id, deep=False, folder=None):
        # The GUI adds a study while the launch checks the staged ones.
        try:
            project.draft_add(name, "B.measure")
        except QueueError as exc:
            refused.append(str(exc))
        return real(job_id, deep=deep, folder=folder)

    monkeypatch.setattr(project, "_launch_lock", _quick(project))
    monkeypatch.setattr(project, "problems", problems)
    moved = project.launch_draft(name)
    monkeypatch.undo()
    assert moved == ["c01-three-a_calibrate"]
    assert refused and "being launched" in refused[0]
    assert project.drafts() == []


def _quick(project):
    """``_launch_lock`` without the wait, so a refusal is immediate."""
    real = project._launch_lock

    def lock(wait):
        return real(wait=min(wait, 0.2))

    return lock


def test_launching_a_draft_twice_says_it_is_gone(project):
    name = _draft(project, tasks=("A.calibrate",))
    project.launch_draft(name)
    with pytest.raises(QueueError, match="no draft"):
        project.launch_draft(name)
    assert project.drafts() == []
    assert not (project.drafts_dir / name).exists()


def test_a_draft_cannot_be_discarded_or_launched_while_one_launches(
    project,
):
    name = _draft(project, tasks=("A.calibrate",))
    lock = st.acquire_lock(project.drafts_dir)
    try:
        with pytest.raises(QueueError, match="being launched"):
            project.discard_draft(name)
        with pytest.raises(QueueError, match="being launched"):
            project.launch_draft(name)
        assert project.recover_launches() == []  # leaves it alone
    finally:
        st.release_lock(lock)
    assert project.drafts()[0]["count"] == 1


def test_ids_a_launch_has_given_out_are_taken(project):
    stage = project.queue.queue_dir / f"{STAGE}some-draft"
    stage.mkdir(parents=True)
    st.write_json(
        stage / "launch.json",
        {
            "phase": "staging",
            "entries": [{"old_id": "x", "new_id": "c01-01_t"}],
        },
    )
    assert "c01-01_t" in project.queue.taken()
    assert project.queue.next_campaign_id() == "c02"
    stage.joinpath("launch.json").unlink()
    stage.rmdir()


def test_a_launch_checks_deeply(project, monkeypatch):
    name = _draft(project, tasks=("A.calibrate",))
    seen = []
    real = project.problems

    def problems(job_id, deep=False, folder=None):
        seen.append(deep)
        return ["only the deep check finds it"] if deep else []

    monkeypatch.setattr(project, "problems", problems)
    with pytest.raises(QueueError, match="only the deep check"):
        project.launch_draft(name)
    assert seen == [True]
    monkeypatch.setattr(project, "problems", real)


# -- 3. the write record -------------------------------------------------


def test_a_recovered_write_back_is_not_recorded_twice(project):
    job_id = project.add("A.calibrate", save=True)
    set_task(project, job_id, value=3.0)
    Runner(project).run()
    (done,) = project.queue.completed()
    record = st.result_of(done)
    record["status"] = st.COMMITTING  # as if cut off after the record
    st.write_json(done / st.RESULT_FILE, record)
    Runner(project).run()
    item = read_writes(project.backend.store_path)["A_value"]
    assert [h["study"] for h in item["history"]] == [job_id]
    assert item["latest"]["previous"] == 0.0


def test_a_recovered_write_back_keeps_its_campaign(project):
    spec = project.backend.catalog().campaign("full_A")
    job_id, _ = project.add_campaign(spec)
    claimed = project.queue.claim(job_id)
    folder = project.queue.start(claimed, job_id)
    project.backend.commit({"A_value": 6.0})
    st.write_json(
        folder / st.RESULT_FILE,
        {
            "id": job_id,
            "status": st.COMMITTING,
            "updates": {"A_value": 6.0},
            "before": {"A_value": 0.0},
        },
    )
    Runner(project).run()
    latest = read_writes(project.backend.store_path)["A_value"]["latest"]
    assert latest["recovered"] is True and latest["campaign"] == "c01"
    assert latest["previous"] == 0.0


def test_a_recovered_write_that_changed_nothing_is_not_credited(project):
    """Study A wrote 2.5; study B proposed 2.5 too and died before its
    write-back: B must not take A's place as the writer."""
    first = project.add("A.calibrate", save=True)
    set_task(project, first, value=2.5)
    Runner(project).run()
    second = project.add("A.calibrate", save=True)
    claimed = project.queue.claim(second)
    folder = project.queue.start(claimed, second)
    st.write_json(
        folder / st.RESULT_FILE,
        {
            "id": second,
            "status": st.COMMITTING,
            "updates": {"A_value": 2.5},
            "before": {"A_value": 2.5},
        },
    )
    Runner(project).run()
    latest = read_writes(project.backend.store_path)["A_value"]["latest"]
    assert latest["study"] == first


def test_a_record_that_is_not_a_mapping_is_set_aside(project):
    path = writes_path(project.backend.store_path)
    path.write_text("[1, 2]\n", encoding="utf-8")
    assert read_writes(project.backend.store_path) == {}
    project.add("A.calibrate", save=True)
    Runner(project).run()
    assert "A_value" in read_writes(project.backend.store_path)


def test_a_helper_that_gives_a_list_or_nothing(monkeypatch):
    from qdesignoptimizer.sequencer.backends import qdo_files as qf

    monkeypatch.setattr(qf, "data_to_target", lambda e: ["t1", "t2"])
    assert qf.build_targets([{"helper": "h"}]) == ["t1", "t2"]
    monkeypatch.setattr(qf, "data_to_target", lambda e: [])
    with pytest.raises(qf.CodecError, match="gives no target"):
        qf.build_targets([{"helper": "h"}])


def test_add_offers_only_what_builds_one_target():
    from qdesignoptimizer.sequencer.backends.qdo import QDOBackend

    backend = QDOBackend.__new__(QDOBackend)
    classes = backend.item_choices("ministudy.yaml")[
        ("args", "capacitance_matrix_studies")
    ]
    assert "ModeDecayStudy" not in {c["label"] for c in classes}
    helpers = backend.item_choices("opt_targets.yaml")[("targets",)]
    assert not any("targets_" in c["label"] for c in helpers)


def test_a_value_that_did_not_change_is_not_recorded(project):
    job_id = project.add("A.calibrate", save=True)
    set_task(project, job_id, value=0.0)  # what the store holds already
    Runner(project).run()
    assert not writes_path(project.backend.store_path).exists()


def test_a_store_that_cannot_be_read_does_not_fail_the_write(project, monkeypatch):
    project.add("A.calibrate", save=True)
    real = project.backend.store_values
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 2:  # the read for the record
            raise PermissionError("locked")
        return real()

    monkeypatch.setattr(project.backend, "store_values", flaky)
    Runner(project).run()
    (done,) = project.queue.completed()
    assert st.result_of(done)["status"] == "done"
    assert real()["A_value"] == 1.0


def test_a_damaged_record_is_set_aside_not_left_broken(project):
    path = writes_path(project.backend.store_path)
    path.write_text("<<<<<<< HEAD\n{}\n", encoding="utf-8")
    project.add("A.calibrate", save=True)
    Runner(project).run()
    assert "A_value" in read_writes(project.backend.store_path)
    assert list(path.parent.glob("store.writes.damaged-*.json"))


def test_a_value_edited_by_hand_says_so(project):
    project.add("A.calibrate", save=True)
    Runner(project).run()
    store = project.backend.store_path
    assert sources_text(store)["A_value"].startswith("x-")
    data = json.loads(store.read_text())
    data["A_value"] = 42.0
    store.write_text(json.dumps(data))
    assert sources_text(store)["A_value"].startswith("edited since")


# -- 5. a "~" in a project's folder name ---------------------------------


def test_a_tilde_in_a_project_name_is_not_a_draft(tmp_path):
    a = make_project(tmp_path / "chip~2")
    b = make_project(tmp_path / "~lead")
    workspace = Workspace([a, b], state_dir=tmp_path / "state")
    assert list(workspace.projects) == ["chip~2", "lead"]
    bridge = Bridge(workspace=workspace)
    key = bridge.add("task", "chip~2/A.calibrate").split()[-1]
    assert bridge.details("queue", key)["kind"] == "queue"
    assert "saved" in bridge.save_job(key, {"notes": "hi"})
    draft = bridge.new_draft(project="chip~2", label="d")
    assert draft.startswith("chip~2/~")
    study = bridge.draft_add(draft, "A.calibrate")
    assert bridge.details("draft", study)["kind"] == "draft"
    with pytest.raises(GuiError, match="no project"):
        bridge.details("draft", "nosuch/~d/x")


# -- minor ---------------------------------------------------------------


def test_a_target_file_that_is_a_list_says_what_is_wrong(project):
    from qdesignoptimizer.sequencer.backends.qdo import _targets_of

    assert _targets_of(["a"]) is None and _targets_of({"targets": []}) == []


# -- third review --------------------------------------------------------


def test_a_draft_is_not_changed_while_it_launches(project):
    """Adding to, renaming or blanking a draft waits for the launch
    lock, and refuses once the launch took the draft away -- it never
    writes a study into a draft folder that no longer exists."""
    name = _draft(project, tasks=("A.calibrate",))
    lock = st.acquire_lock(project.drafts_dir, wait=0)
    try:
        with pytest.raises(QueueError, match="being launched"):
            project.draft_add(name, "A.measure")
        with pytest.raises(QueueError, match="being launched"):
            project.set_draft_label(name, "renamed")
    finally:
        st.release_lock(lock)
    project.launch_draft(name)
    with pytest.raises(QueueError, match="no draft"):
        project.draft_add(name, "A.measure")
    assert not (project.drafts_dir / name).exists()
    # The launched draft is kept in removed/, not deleted.
    assert any(
        p.name.endswith(f"{name}_launched-c01")
        for p in project.queue.removed_dir.iterdir()
    )


def test_a_launch_that_published_nothing_keeps_its_campaign_number(
    project, monkeypatch
):
    first = _draft(project)
    _failing_rename(monkeypatch, fail_on=3 + 1)  # the first publish
    with pytest.raises(QueueError, match="launched 0 of 3 as c01"):
        project.launch_draft(first)
    monkeypatch.undo()
    assert project.queue.ids() == []
    # c01 is the first draft's: another campaign must not take it.
    other = project.new_draft("other")
    project.draft_add(other, "B.measure")
    assert project.launch_draft(other) == ["c02-other-b_measure"]
    # Renamed in between, the first keeps the name it launched with.
    project.set_draft_label(first, "renamed")
    assert project.launch_draft(first)[0] == "c01-three-a_calibrate"


def test_a_failed_launch_keeps_saying_why_until_the_popup_closes(project):
    bridge = Bridge(project=project)
    key = bridge.new_draft(label="bad")
    study = bridge.draft_add(key, "A.calibrate")
    bridge.save_file(study, "task.yaml", "nonsense: 1\n")
    bridge.start_launch(key)
    for _ in range(200):
        status = bridge.launch_status(key)
        if status["state"] != "running":
            break
        import time

        time.sleep(0.05)
    assert status["state"] == "failed"
    assert bridge.launch_shown(key) is True  # announced once
    assert bridge.launch_shown(key) is False
    assert bridge.launch_status(key)["state"] == "failed"  # still shown
    bridge.forget_shown_launches()  # the popup closed
    assert bridge.launch_status(key) is None


def test_a_study_whose_folder_would_be_too_long_is_warned_not_refused(
    project,
):
    """A solver that cannot write past ``path_limit`` characters: the
    study still queues and runs; the GUI and the add message say why
    its solver files may be missing."""
    job_id = project.add("A.calibrate")
    assert project.warnings(job_id) == []  # no limit declared
    backend = project.backend
    backend.path_limit = 40
    backend.solver_files = ("model.png",)
    backend.path_limit_note = "the snapshot would be missing"
    (warning,) = project.warnings(job_id)
    assert "model.png" in warning and "past the solver's 40" in warning
    assert "the snapshot would be missing" in warning
    assert project.problems(job_id) == []  # a warning, not a problem
    bridge = Bridge(project=project)
    assert "warning" in bridge.add("task", "A.measure")
    assert bridge.details("queue", job_id)["warnings"] == [warning]
    row = next(r for r in bridge.snapshot()["rows"] if r["id"] == job_id)
    assert row["warnings"] == [warning]
    backend.path_limit = 10_000
    assert project.warnings(job_id) == []
