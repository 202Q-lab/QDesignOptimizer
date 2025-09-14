from copy import deepcopy

from check_convergence import check_convergence, print_convergence_summary
from plot_convergence import plot_all_convergence_ratios
from plot_settings import get_plot_settings
from scaled_system_definition import ScaledSystem, get_prop_to

from qdesignoptimizer.anmod_optimizer import ANModOptimizer
from qdesignoptimizer.design_analysis_types import OptTarget
from qdesignoptimizer.utils.names_parameters import mode


def get_opt_target(i, j, sys: ScaledSystem) -> OptTarget:
    """Return the optimization target for parameter with composite index a=(i,j)."""
    mode_name = mode(f"{i},{j}")
    return OptTarget(
        target_param_type="",
        involved_modes=[mode_name],
        design_var=f"dv_{i},{j}",
        design_var_constraint={"larger_than": 1e-8, "smaller_than": 100},
        prop_to=get_prop_to(i, j, sys),
        independent_target=i,
    )


def aggregate_results(
    optimization_results: list, sys: ScaledSystem, minimization_results: any
):
    design_variables = deepcopy(sys.get_flattened_x())
    system_optimized_params = deepcopy(sys.get_flattened_y())
    h_factor = deepcopy(sys.get_flattened_h())
    g_approx_over_g_factor = deepcopy(sys.get_flattened_g_approx_over_g())

    iteration_result = {}
    iteration_result["design_variables"] = design_variables
    iteration_result["system_optimized_params"] = system_optimized_params
    iteration_result["minimization_results"] = minimization_results
    iteration_result["h_factor"] = h_factor
    iteration_result["g_approx_over_g_factor"] = g_approx_over_g_factor

    optimization_results.append(iteration_result)


# ---------------- Example usage ----------------
if __name__ == "__main__":
    sys = ScaledSystem(
        n_clusters=1000,
        m_per_cluster=3,
        epsilon=0.05,
        exponent_approx_to_1_over=4,
        seed=42,
        sample_range_alpha_ij_eq_k=[(2.0, 2.5)],
        sample_range_alpha_ij_neq_k=[(-1, -0.1)],
        sample_range_beta=[(-0.5, -0.1), (0.1, 0.5)],
        sample_range_gamma=[(-1.5, -0.1), (0.1, 1.5)],
    )

    all_opt_targets = [
        get_opt_target(i, j, sys)
        for i in range(sys.n_clusters)
        for j in range(sys.m_per_cluster)
    ]

    anmod = ANModOptimizer(
        opt_targets=all_opt_targets,
        system_target_params=sys.get_flattened_y_target(),
        adjustment_rate=1,
        minimization_tol=1e-16,
    )

    optimization_results = []
    system_target_params = sys.get_flattened_y_target()

    NBR_ITERATIONS = 10
    for it in range(NBR_ITERATIONS):
        sys.gather_info_for_y_given_x()
        system_optimized_params = sys.get_flattened_y()
        updated_design_vars, minimization_results = anmod.calculate_target_design_var(
            system_optimized_params=system_optimized_params,
            variables_with_units=sys.get_flattened_x(),
        )
        sys.set_updated_design_vars(updated_design_vars)
        aggregate_results(optimization_results, sys, minimization_results)

    plot_all_convergence_ratios(optimization_results, system_target_params)

    convergence_iteration, convergence_status = check_convergence(
        optimization_results, system_target_params, tolerance=0.001
    )
    print_convergence_summary(
        convergence_iteration, convergence_status, tolerance=0.001
    )

    # plot_settings = get_plot_settings(sys.n_clusters, sys.m_per_cluster)
    # simulation = [
    #     {
    #         "optimization_results": optimization_results,
    #         "system_target_params": system_target_params,
    #         "plot_settings": plot_settings,
    #         "design_analysis_version": anmod.anmod_version,
    #     }
    # ]

    # plot_progress(
    #         [optimization_results],
    #         system_target_params,
    #         plot_settings,
    #         block_plots=True
    #     )
