"""Integration tests running every tutorial optimization notebook against faked simulators.

See ``example_harness.py`` for how the notebooks are executed and how to
regenerate the snapshots after an intended change.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

import qdesignoptimizer
from qdesignoptimizer.design_analysis_types import SimulationResults
from qdesignoptimizer.utils.plotting import load_data_by_date, plot_optimization_results

from .conftest import EXAMPLE_CASES
from .example_harness import (
    assert_matches_snapshot,
    build_snapshot,
    notebook_code_cells,
    relative_errors_to_targets,
)

CASE_NAMES = [case.name for case in EXAMPLE_CASES]


def test_all_tutorial_notebooks_are_covered():
    """New example notebooks must be added to EXAMPLE_CASES in conftest.py."""
    from .example_harness import TUTORIALS

    notebooks = {
        path.resolve()
        for path in TUTORIALS.glob("examples_*/main_*.ipynb")
        if ".ipynb_checkpoints" not in path.parts
    }
    covered = {case.notebook.resolve() for case in EXAMPLE_CASES}
    assert notebooks == covered


@pytest.mark.parametrize("name", CASE_NAMES)
def test_simulator_calls_and_results_match_snapshot(name, run_example):
    """The same notebook settings must lead to the same HFSS/pyEPR/Q3D calls and results."""
    run = run_example(name)
    assert_matches_snapshot(
        name, build_snapshot(run.session, run.optimization_results, run.analysis)
    )


@pytest.mark.parametrize(
    "name", [case.name for case in EXAMPLE_CASES if case.converges]
)
def test_optimization_reaches_targets(name, run_example):
    """With a simulator that follows the targets' own relations, every example
    must converge to its targets (which checks the loop, not the relations)."""
    run = run_example(name)
    errors = relative_errors_to_targets(run.analysis, run.optimization_results)
    assert errors, "the example has no optimization targets"
    not_converged = {key: err for key, err in errors.items() if err > 1e-4}
    assert not not_converged, f"targets not reached: {not_converged}"


@pytest.mark.parametrize("name", CASE_NAMES)
def test_saved_results_can_be_reloaded_and_replotted(name, run_example):
    """The .npy written to out/ round-trips and works with the plotting utilities."""
    run = run_example(name)
    out = run.workdir / "out"
    files = load_data_by_date(
        str(out), "multi_transmon_chip", "20000101-000000", "21000101-000000"
    )
    assert files, "no results found by load_data_by_date"
    newest = max(files, key=lambda file: Path(file).stat().st_mtime)

    loaded = SimulationResults.from_dict(np.load(newest, allow_pickle=True)[0])
    assert loaded.system_target_params == run.analysis.system_target_params
    assert loaded.design_analysis_version == qdesignoptimizer.__version__
    assert set(loaded.plot_settings) == set(
        run.analysis.plot_settings or run.namespace["PLOT_SETTINGS"]
    )
    expected = [_essentials(result) for result in run.optimization_results]
    assert [_essentials(result) for result in loaded.optimization_results] == expected

    design_vars_file = Path(newest[: -len(".npy")] + "_design_variables.json")
    saved_design_vars = json.loads(design_vars_file.read_text())
    final_design_vars = run.optimization_results[-1]["design_variables"]
    for target in run.analysis.opt_targets:
        assert (
            saved_design_vars[target.design_var] == final_design_vars[target.design_var]
        )

    plot_optimization_results(
        [newest],
        plot_variance=False,
        save_figures=False,
    )
    plt.close("all")


def _essentials(result: dict) -> dict:
    return {
        "design_variables": dict(result["design_variables"]),
        "system_optimized_params": dict(result["system_optimized_params"]),
    }


@pytest.mark.parametrize("name", CASE_NAMES)
def test_result_getters_return_the_requested_iteration(name, run_example):
    run = run_example(name)
    analysis = run.analysis
    results = analysis.optimization_results
    for iteration, result in enumerate(results):
        if "eig_results" in result:
            assert analysis.get_eigenmode_results(iteration) is result["eig_results"]
            assert analysis.get_cross_kerr_matrix(iteration) is result["cross_kerrs"]
        else:
            assert analysis.get_eigenmode_results(iteration) is None
        if result.get("capacitance_matrix"):
            assert (
                analysis.get_capacitance_matrix(1, iteration)
                is result["capacitance_matrix"][0]
            )
            assert analysis.get_capacitance_matrix(2, iteration) is None
    if len(results) > 1 and "eig_results" in results[0]:
        assert not analysis.get_eigenmode_results(0).equals(
            analysis.get_eigenmode_results(-1)
        )


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason="Known notebook bug: the partitioned example creates its DesignAnalysis "
    "objects with save_path=None, so its overwrite_parameters() cell raises.",
)
def test_partitioned_example_skipped_cells_still_fail(run_example, monkeypatch):
    """Once the notebook is fixed this XPASSes; then remove skip_cells_containing."""
    name = "tunable_coupler__eigenmode_two_qubit_resonator_partitioned"
    run = run_example(name)
    case = run.case
    skipped = [
        cell
        for cell in notebook_code_cells(case.notebook)
        if any(marker in cell for marker in case.skip_cells_containing)
    ]
    namespace = dict(run.namespace)  # do not alter the shared run
    monkeypatch.chdir(run.workdir)  # the notebook reads/writes relative paths
    for cell in skipped:
        exec(
            compile(cell, case.notebook.name, "exec"), namespace
        )  # pylint: disable=exec-used
