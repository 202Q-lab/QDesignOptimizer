"""Which study wrote each value back: ``<store>.writes.json``."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from qdesignoptimizer.sequencer.app.queue_state import Bridge
from qdesignoptimizer.sequencer.backends._store import read_writes, writes_path
from qdesignoptimizer.sequencer.engine import state as st
from qdesignoptimizer.sequencer.engine.runner import Runner

from .helpers import make_project, set_task


@pytest.fixture
def project(tmp_path):
    return make_project(tmp_path)


def test_a_write_back_records_the_study_that_wrote_it(project):
    first = project.add("A.calibrate", save=True)
    set_task(project, first, value=2.5)
    Runner(project).run()
    record = read_writes(project.backend.store_path)
    assert set(record) == {"A_value"}  # only what the study tuned
    latest = record["A_value"]["latest"]
    assert latest["value"] == 2.5 and latest["previous"] == 0.0
    assert latest["study"] == first and latest["task"] == "A.calibrate"
    assert latest["folder"].startswith(f"out/completed/{first}_20")
    assert latest["written"]

    second = project.add("A.calibrate", save=True)
    set_task(project, second, value=4.0)
    Runner(project).run()
    item = read_writes(project.backend.store_path)["A_value"]
    assert [h["study"] for h in item["history"]] == [first, second]
    assert item["latest"]["previous"] == 2.5


def test_nothing_is_recorded_when_nothing_is_written(project):
    project.add("A.calibrate", save=False)
    failed = project.add(
        "B.calibrate",
        save=True,
        checks=["converged"],
        on_failure="continue",
    )
    set_task(project, failed, converged=False)
    Runner(project).run()
    (bad,) = (p for p in project.queue.completed() if "_failed" in p.name)
    assert st.result_of(bad)["committed"] == {}
    assert not writes_path(project.backend.store_path).exists()


def test_a_campaign_write_names_its_campaign(project):
    spec = project.backend.catalog().campaign("full_A")
    project.add_campaign(spec)
    Runner(project).run()
    latest = read_writes(project.backend.store_path)["A_value"]["latest"]
    assert latest["campaign"] == "c01"
    assert latest["campaign_name"] == "full_A"


def test_a_recovered_write_back_is_recorded_as_recovered(project):
    job_id = project.add("A.calibrate", save=True)
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
    assert latest["study"] == job_id and latest["recovered"] is True


def test_the_gui_shows_who_last_wrote_each_value(project):
    project.add("A.calibrate", save=True)
    Runner(project).run()
    bridge = Bridge(project=project)
    queued = project.add("B.calibrate")
    spec = bridge.file_spec(queued, "design_variables.json")
    assert spec["sources"]["A_value"].startswith("x-")
    assert "B_value" not in spec["sources"]


def test_the_qdo_backend_records_only_the_tuned_design_variable(tmp_path):
    demo = (
        Path(__file__).resolve().parents[2]
        / "tutorials"
        / "example_simulation_sequencer"
        / "demo"
    )
    root = tmp_path / "chip_qdo"
    shutil.copytree(
        demo / "chip_qdo",
        root,
        ignore=shutil.ignore_patterns("out", "design_variables.json"),
    )
    from qdesignoptimizer.sequencer.engine.workspace import open_workspace

    workspace = open_workspace([root], isolate=True)
    try:
        (project,) = workspace.projects.values()
        job_id = project.add("R1.frequency", save=True)
        folder = project.queue.folder(job_id)
        text = (folder / "job.yaml").read_text(encoding="utf-8")
        (folder / "job.yaml").write_text(
            text.replace("seconds: 5", "seconds: 0"), encoding="utf-8"
        )
        assert Runner(workspace).run().status == st.FINISHED
        record = read_writes(root / "design_variables.json")
        # The frequency study's target tunes the length; the coupling
        # length is in the study's variables but not its target.
        assert set(record) == {"design_var_length_resonator_R1"}
        latest = record["design_var_length_resonator_R1"]["latest"]
        assert latest["study"] == job_id
        assert latest["previous"] == "4000.0 um"
        # What produced it: the simulation, and the target it chased.
        assert latest["simulation"]["kind"] == "eigenmode"
        assert latest["simulation"]["runs"] == ["HFSS eigenmode"]
        assert latest["settings"]["seconds"] == 0
        (target,) = latest["target"]
        assert target["parameter"] == "resonator_R1_freq"
        assert target["type"] == "freq"
        assert target["target"] > 0 and target["achieved"] > 0
        assert isinstance(target["within_tolerance"], bool)
        sources = project.backend.value_sources("design_variables.json")
        assert "(eigenmode)" in sources["design_var_length_resonator_R1"]

        # The study keeps its own record: convergence and its log.
        (done,) = project.queue.completed()
        data = json.loads((done / "convergence.json").read_text("utf-8"))
        (item,) = data["targets"]
        assert item["parameter"] == "resonator_R1_freq"
        assert item["design_var"] == "design_var_length_resonator_R1"
        assert item["unit"] == "um"
        assert len(item["values"]) == data["iterations"] >= 1
        assert len(item["design_values"]) == data["iterations"]
        assert (done / "convergence.png").read_bytes()[:4] == b"\x89PNG"
        log_text = (done / "study.log").read_text("utf-8")
        assert "iteration 1/" in log_text  # the dummy solver's own lines
    finally:
        workspace.close()
