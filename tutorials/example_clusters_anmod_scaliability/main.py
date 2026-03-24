from check_convergence import check_convergence, print_convergence_summary
from plot_convergence import plot_all_convergence_ratios
from plot_settings import get_plot_settings
from scaled_system_definition import ScaledSystem
from utils import aggregate_results, get_opt_target

from qdesignoptimizer.anmod_optimizer import ANModOptimizer

# ---------------- Example usage ----------------

# This tutorial demonstrates how the ANMod optimization method can scale to
# large nonlinear systems. The ScaledSystem acts as a fictive nonlinear
# "full" model that is evaluated each iteration, while ANMod computes updates
# from an approximate version of that full model.
# This mirrors realistic workflows where direct optimization on the full model
# alone would be too expensive.

# There are 3000 parameters to optimize in total (1000 clusters with 3 parameters each).
# looping over seeds = 0-99 gives the following number of convergences within 10 iterations
# [2993, 3000, 3000, 2992, 0, 0, 3000, 3000, 3000, 2997, 0, 0, 2997, 3000, 3000, 0, 2878,
# 2080, 3000, 2992, 2995, 2998, 0, 3000, 3000, 2965, 3000, 2981, 2996, 2990, 3000, 3000,
# 2995, 3000, 2996, 2997, 2996, 2995, 2998, nan, 2991, 2997, 2997, 3000, 2996, nan, 3000,
# 2992, 2997, 2997, 3000, 2982, 2997, 2996, 2997, 2996, 2974, 2996, 0, 0, 2996, 2997, 2952,
# 2997, 3000, 3000, 2986, 2997, 2997, 2997, 2996, 2997, 2998, 3000, 2997, 2974, 2997, 2989,
# 2998, 2995, 2997, nan, 3000, 2991, 2995, 3000, 2997, 2997, 0, 3000, 2995, 0, 2994, 2995,
# 3000, 2999, 2998, 0, 3000, 2999]
# 0 diverges and nan gets stuck in evaluation


# -------- User settings --------

NBR_ITERATIONS = 10
SHOW_PLOT = True
RUN_MULTIPLE_SEEDS = False  # True: Runs multiple seeds with the results summarized above. False: Run with seed=2 for quick testing.

# The generated plots will end up in tutorials/example_clusters_anmod_scaliability/out

# -------- End of user settings --------


all_convergence_status = []

if RUN_MULTIPLE_SEEDS:
    all_seeds = range(100)
else:
    all_seeds = [2]

for seed in all_seeds:
    if seed in [39, 44, 79]:
        continue  # getting stuck i.e. diverging

    sys = ScaledSystem(
        n_clusters=1000,
        m_per_cluster=3,
        epsilon=0.05,
        exponent_approx_to_1_over=4,
        seed=seed,
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

    for it in range(NBR_ITERATIONS):
        sys.gather_info_for_y_given_x()
        system_optimized_params = sys.get_flattened_yk()
        updated_design_vars, minimization_results = anmod.calculate_target_design_var(
            system_optimized_params=system_optimized_params,
            variables_with_units=sys.get_flattened_x(),
        )
        print(f"--- Iteration {it+1}/{NBR_ITERATIONS} ---")
        sys.set_updated_design_vars(updated_design_vars)
        aggregate_results(optimization_results, sys, minimization_results)

    plot_all_convergence_ratios(
        optimization_results, system_target_params, seed=seed, show_plot=SHOW_PLOT
    )

    convergence_iteration, convergence_status = check_convergence(
        optimization_results, system_target_params, tolerance=0.001
    )
    print_convergence_summary(
        convergence_iteration, convergence_status, tolerance=0.001
    )

    all_convergence_status.append(convergence_status)

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

# print("Converge counts ", [c[NBR_ITERATIONS-1]["converged_count"] for c in all_convergence_status])
