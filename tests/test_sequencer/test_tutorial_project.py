"""The tutorial project on the coupled-transmon chip, short of Ansys.

``tutorials/example_simulation_sequencer/coupled_transmon_chip`` solves in
Ansys, which no test has; what a test can hold is everything before the
solve: its templates and targets read, the studies they make, a measurement plan
queued and every study deep-checked against the built chip.
"""

import shutil
from pathlib import Path

import yaml

from qdesignoptimizer.sequencer.engine.workspace import open_workspace

_TUTORIALS = Path(__file__).resolve().parents[2] / "tutorials"
_PROJECT = _TUTORIALS / "example_simulation_sequencer" / "coupled_transmon_chip"
_CHIP = _TUTORIALS / "examples_coupled_transmon_chip"


def _copy(tmp_path: Path) -> Path:
    """The project and the chip it builds on, in the same layout."""
    shutil.copytree(_CHIP, tmp_path / _CHIP.name)
    root = tmp_path / "example_simulation_sequencer" / _PROJECT.name
    shutil.copytree(
        _PROJECT,
        root,
        ignore=shutil.ignore_patterns("out", "design_variables.json"),
    )
    return root


def _targets(folder: Path) -> dict:
    entries = yaml.safe_load((folder / "opt_targets.yaml").read_text("utf-8"))
    return {e["name"]: e.get("active", True) for e in entries["targets"]}


def test_the_tutorial_project_queues_its_plan_and_checks_clean(tmp_path):
    root = _copy(tmp_path)
    workspace = open_workspace([root], isolate=True)
    try:
        (project,) = workspace.projects.values()
        catalog = project.backend.catalog()
        assert project.backend.catalog_problems == []
        assert {t.key for t in catalog.tasks} == {
            "Q1.qubit_resonator",
            "Q1.qubit_charge_line",
            "Q2.qubit_charge_line",
            "Q1.resonator_kappa",
            "Q2.resonator_kappa",
        }
        ids = project.add_plan(catalog.plan("tune_Q1"))
        assert len(ids) == 3
        for job_id in ids:
            assert project.problems(job_id, deep=True) == []

        kappa, charge_line, eigenmode = (_targets(project.queue.folder(i)) for i in ids)
        # Each kind of study switches on the targets it produces.
        assert kappa == {
            "Q1_resonator_freq": False,
            "Q1_resonator_kappa": True,
            "Q1_resonator_kappa_eigenmode": False,
        }
        assert charge_line["Q1_charge_line_t1"] is True
        # Capacitance models without wire bonds, as in the notebooks.
        for job_id in ids[:2]:
            study = yaml.safe_load(
                (project.queue.folder(job_id) / "ministudy.yaml").read_text("utf-8")
            )
            (chained,) = study["args"]["capacitance_matrix_studies"]
            assert chained["args"]["render_qiskit_metal_kwargs"] == {
                "capacitance_or_surface_p_ratio": True
            }
        assert charge_line["Q1_qubit_freq"] is False
        assert eigenmode["Q1_resonator_kappa_eigenmode"] is True
        assert eigenmode["Q1_resonator_kappa"] is False
        assert eigenmode["Q1_chi"] is True
        # The store starts as a copy of the tutorial chip's.
        assert project.backend.store_values() == yaml.safe_load(
            (_CHIP / "design_variables.json").read_text("utf-8")
        )
    finally:
        workspace.close()


def test_the_deep_check_refuses_what_qdo_would_fail_on_after_a_solve(tmp_path):
    """A target switched on by hand that the study does not measure, and
    one whose parameter has no target value: both refused before Ansys."""
    root = _copy(tmp_path)
    workspace = open_workspace([root], isolate=True)
    try:
        (project,) = workspace.projects.values()
        (job_id,) = project.add_plan(project.backend.catalog().plan("tune_Q1"))[2:]
        folder = project.queue.folder(job_id)
        path = folder / "opt_targets.yaml"
        data = yaml.safe_load(path.read_text("utf-8"))
        for entry in data["targets"]:
            if entry["name"] in ("Q1_resonator_kappa", "Q1_charge_line_t1"):
                entry["active"] = True
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        problems = project.problems(job_id, deep=True)
        # Nothing in an eigenmode study measures a charge-line T1.
        assert any(
            "does not measure 'qubit_1_charge_line_limited_t1'" in p for p in problems
        ), problems
        # The capacitance-kind kappa is measured here too (ports), but it
        # tunes the variable the eigenmode one already tunes.
        assert any("both tune" in p for p in problems), problems

        targets = root / "parameter_targets.py"
        text = targets.read_text("utf-8")
        targets.write_text(
            text.replace("    param(n.QUBIT_1, n.FREQ): 4e9,\n", ""), encoding="utf-8"
        )
        problems = project.problems(job_id, deep=True)
        assert any("'qubit_1_freq'" in p for p in problems), problems
    finally:
        workspace.close()
