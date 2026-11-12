"""The demo backend and the demo projects in
tutorials/example_simulation_sequencer/demo/."""

from pathlib import Path

from qdesignoptimizer.sequencer.backends.demo import SOLVE_SECONDS, DemoBackend
from qdesignoptimizer.sequencer.engine import state as st
from qdesignoptimizer.sequencer.engine.project import Project
from qdesignoptimizer.sequencer.engine.runner import Runner
from qdesignoptimizer.sequencer.engine.workspace import find_adapters

_DEMO = (
    Path(__file__).resolve().parents[2]
    / "tutorials"
    / "example_simulation_sequencer"
    / "demo"
)


def test_the_demo_folder_opens_both_projects():
    found = [Path(f).parents[1].name for f in find_adapters(_DEMO)]
    assert found == ["chip_a", "chip_b", "chip_qdo"]


def test_a_demo_campaign_runs_and_writes_back(tmp_path):
    project = Project(DemoBackend(tmp_path, ("R1", "R2")))
    catalog = project.backend.catalog()
    assert {t.key for t in catalog.tasks} == {
        "R1.frequency",
        "R1.linewidth",
        "R2.frequency",
        "R2.linewidth",
    }
    ids = project.add_campaign(catalog.campaign("tune_R1"))
    assert project.job(ids[0]).settings == {"seconds": SOLVE_SECONDS}
    for job_id in ids:  # no waiting in a test
        folder = project.queue.folder(job_id)
        text = (folder / "job.yaml").read_text(encoding="utf-8")
        (folder / "job.yaml").write_text(
            text.replace(f"seconds: {SOLVE_SECONDS}", "seconds: 0"),
            encoding="utf-8",
        )
    assert Runner(project).run().status == st.FINISHED
    assert project.backend.store_values()["R1_value"] == 1.0


def test_the_qdo_demo_runs_a_campaign_on_the_dummy_solver(tmp_path):
    """chip_qdo end to end, in a copy and with 0 s solves: real QDO
    files queued, three dummy iterations, the tuned values written back
    and the targets hit."""
    import shutil

    from qdesignoptimizer.sequencer.engine.workspace import open_workspace

    root = tmp_path / "chip_qdo"
    shutil.copytree(
        _DEMO / "chip_qdo",
        root,
        ignore=shutil.ignore_patterns("out", "design_variables.json"),
    )
    workspace = open_workspace([root], isolate=True)
    try:
        (project,) = workspace.projects.values()
        catalog = project.backend.catalog()
        assert {t.key for t in catalog.tasks} >= {"R1.frequency"}
        ids = project.add_campaign(catalog.campaign("tune_R1"))
        for job_id in ids:
            folder = project.queue.folder(job_id)
            text = (folder / "job.yaml").read_text(encoding="utf-8")
            (folder / "job.yaml").write_text(
                text.replace("seconds: 5", "seconds: 0"), encoding="utf-8"
            )
            assert project.problems(job_id, deep=True) == []
        final = Runner(workspace).run()
        assert final.status == st.FINISHED, final.message
        store = project.backend.store_values()
        assert store["design_var_length_resonator_R1"] != "4000.0 um"
        assert store["design_var_coupl_length_resonator_R1_tee"] != ("500.0 um")
        for path in project.queue.completed():
            assert st.result_of(path)["checks"]["converged"]["ok"]
    finally:
        workspace.close()
