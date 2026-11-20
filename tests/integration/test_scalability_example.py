"""Scaled-down run of tutorials/example_clusters_anmod_scaliability/main.py.

The script optimizes 3000 parameters, which is too slow for the test suite, so
this test repeats its optimization loop with the tutorial's own ScaledSystem and
helpers for 20 clusters (60 parameters) and seed 2, the script's default.
"""

import sys

import pytest

from qdesignoptimizer.anmod_optimizer import ANModOptimizer

from .example_harness import TUTORIALS, assert_matches_snapshot
from .fake_ansys import to_jsonable

EXAMPLE_DIR = TUTORIALS / "example_clusters_anmod_scaliability"
LOCAL_MODULES = (
    "check_convergence",
    "plot_convergence",
    "plot_settings",
    "scaled_system_definition",
    "utils",
)


@pytest.fixture
def scalability_modules(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    for module in LOCAL_MODULES:
        monkeypatch.delitem(sys.modules, module, raising=False)
    import check_convergence  # pylint: disable=import-outside-toplevel
    import scaled_system_definition  # pylint: disable=import-outside-toplevel
    import utils  # pylint: disable=import-outside-toplevel

    yield check_convergence, scaled_system_definition, utils
    for module in LOCAL_MODULES:
        sys.modules.pop(module, None)


def test_scaled_down_cluster_optimization(scalability_modules):
    check_convergence_module, system_module, utils_module = scalability_modules

    # Same settings as main.py except for the number of clusters.
    system = system_module.ScaledSystem(
        n_clusters=20,
        m_per_cluster=3,
        epsilon=0.05,
        exponent_approx_to_1_over=4,
        seed=2,
        sample_range_alpha_ij_eq_k=[(2.0, 2.5)],
        sample_range_alpha_ij_neq_k=[(-1, -0.1)],
        sample_range_beta=[(-0.5, -0.1), (0.1, 0.5)],
        sample_range_gamma=[(-1.5, -0.1), (0.1, 1.5)],
    )
    opt_targets = [
        utils_module.get_opt_target(i, j, system)
        for i in range(system.n_clusters)
        for j in range(system.m_per_cluster)
    ]
    anmod = ANModOptimizer(
        opt_targets=opt_targets,
        system_target_params=system.get_flattened_y_target(),
        adjustment_rate=1,
        minimization_tol=1e-16,
    )

    optimization_results: list = []
    for _ in range(10):
        system.gather_info_for_y_given_x()
        updated_design_vars, minimization_results = anmod.calculate_target_design_var(
            system_optimized_params=system.get_flattened_yk(),
            variables_with_units=system.get_flattened_x(),
        )
        system.set_updated_design_vars(updated_design_vars)
        utils_module.aggregate_results(
            optimization_results, system, minimization_results
        )

    convergence_iteration, convergence_status = (
        check_convergence_module.check_convergence(
            optimization_results, system.get_flattened_y_target(), tolerance=0.001
        )
    )
    converged_counts = [
        status["converged_count"] for status in convergence_status.values()
    ]
    assert converged_counts[-1] == 60, converged_counts
    # one minimization per cluster (targets are grouped by cluster index)
    assert len(optimization_results[-1]["minimization_results"]) == 20

    assert_matches_snapshot(
        "scalability__20_clusters_seed_2",
        {
            "convergence_iteration": convergence_iteration,
            "converged_counts": converged_counts,
            "iterations": [
                {
                    "design_variables": to_jsonable(dict(result["design_variables"])),
                    "system_optimized_params": to_jsonable(
                        dict(result["system_optimized_params"])
                    ),
                }
                for result in optimization_results
            ],
        },
    )
