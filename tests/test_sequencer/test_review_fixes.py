"""Regressions for the findings of the review of write records
and studies made from scratch."""

from __future__ import annotations

import json

import pytest

from qdesignoptimizer.sequencer.app.queue_state import Bridge, GuiError
from qdesignoptimizer.sequencer.backends._store import (
    meta_path,
    read_meta,
    sources_text,
)
from qdesignoptimizer.sequencer.engine import state as st
from qdesignoptimizer.sequencer.engine.runner import Runner
from qdesignoptimizer.sequencer.engine.workspace import Workspace

from .helpers import make_project, set_task


@pytest.fixture
def project(tmp_path):
    return make_project(tmp_path)


# -- 1. a launch never strands a study -----------------------------------


def _quick(project):
    """``_launch_lock`` without the wait, so a refusal is immediate."""
    real = project._launch_lock

    def lock(wait):
        return real(wait=min(wait, 0.2))

    return lock


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
    item = read_meta(project.backend.store_path)["A_value"]
    assert [h["study"] for h in item["history"]] == [job_id]
    assert item["latest"]["previous"] == 0.0


def test_a_recovered_write_back_keeps_its_sequence(project):
    spec = project.backend.catalog().sequence("full_A")
    job_id, _ = project.add_sequence(spec)
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
    latest = read_meta(project.backend.store_path)["A_value"]["latest"]
    assert latest["recovered"] is True and latest["sequence"] == "s001"
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
    latest = read_meta(project.backend.store_path)["A_value"]["latest"]
    assert latest["study"] == first


def test_a_record_that_is_not_a_mapping_is_set_aside(project):
    path = meta_path(project.backend.store_path)
    path.write_text("[1, 2]\n", encoding="utf-8")
    assert read_meta(project.backend.store_path) == {}
    project.add("A.calibrate", save=True)
    Runner(project).run()
    assert "A_value" in read_meta(project.backend.store_path)


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
    assert not meta_path(project.backend.store_path).exists()


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
    path = meta_path(project.backend.store_path)
    path.write_text("<<<<<<< HEAD\n{}\n", encoding="utf-8")
    project.add("A.calibrate", save=True)
    Runner(project).run()
    assert "A_value" in read_meta(project.backend.store_path)
    assert list(path.parent.glob("store.meta.damaged-*.json"))


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


def test_an_at_sign_in_a_project_name_is_not_a_study_on_hold(tmp_path):
    a = make_project(tmp_path / "chip@2")
    b = make_project(tmp_path / "@lead")
    workspace = Workspace([a, b], state_dir=tmp_path / "state")
    assert list(workspace.projects) == ["chip@2", "lead"]
    bridge = Bridge(workspace=workspace)
    key = bridge.add("task", "chip@2/A.calibrate").split()[-1]
    assert bridge.details("queue", key)["kind"] == "queue"
    assert "saved" in bridge.save_job(key, {"notes": "hi"})
    bridge.park(key)
    held = bridge.hold_key_of(key)
    assert held.startswith("chip@2/@hold/")
    assert bridge.details("hold", held)["kind"] == "hold"
    with pytest.raises(GuiError, match="no project"):
        bridge.details("hold", "nosuch/@hold/x")


# -- minor ---------------------------------------------------------------


def test_a_target_file_that_is_a_list_says_what_is_wrong(project):
    from qdesignoptimizer.sequencer.backends.qdo import _targets_of

    assert _targets_of(["a"]) is None and _targets_of({"targets": []}) == []


# -- third review --------------------------------------------------------


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
