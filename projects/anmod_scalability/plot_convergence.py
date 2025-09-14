import matplotlib.pyplot as plt
import numpy as np

from qdesignoptimizer.sim_plot_progress import plot_progress


def plot_all_convergence_ratios(optimization_results, system_target_params):
    """
    Plot convergence analysis in a 2x2 matrix of subplots.

    Parameters
    ----------
    optimization_results : List[Dict]
        List of optimization results from each iteration
    system_target_params : Dict[str, float]
        Target parameter values
    """
    # Get all parameter names from the first iteration
    param_names = list(optimization_results[0]["system_optimized_params"].keys())
    n_iterations = len(optimization_results)
    iterations = range(n_iterations)

    # Create 2x2 subplot matrix
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # ===== Upper Right: y/y_target convergence =====
    ax1.set_title("Convergence: y/y_target vs Iterations", fontsize=14)

    for param_name in param_names:
        ratios = []
        target_val = system_target_params[param_name]

        for iter_result in optimization_results:
            optimized_val = iter_result["system_optimized_params"][param_name]
            ratio = optimized_val / target_val
            ratios.append(ratio)

        ax1.plot(
            iterations,
            ratios,
            "o-",
            linewidth=1.5,
            markersize=4,
            label=param_name,
            alpha=0.7,
        )

    ax1.set_xlabel("Iteration", fontsize=12)
    ax1.set_ylabel("y/y_target Ratio", fontsize=12)
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    # ===== Lower Left: Histogram of last iteration's h_factors =====
    ax2.set_title("Distribution of g_ij^approx/g_ij (Last Iteration)", fontsize=14)

    # Get last iteration's h_factors
    last_g_approx_over_g_factors = []
    for param_name in param_names:
        last_g_approx_over_g_factors.append(
            optimization_results[-1]["g_approx_over_g_factor"][param_name]
        )

    # Create histogram
    ax2.hist(
        last_g_approx_over_g_factors,
        bins=min(30, len(param_names) // 2),
        alpha=0.7,
        edgecolor="black",
    )
    ax2.set_xlabel("g_ij^approx/g_ij Value", fontsize=12)
    ax2.set_ylabel("Number of occurrences", fontsize=12)
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    # ===== Lower Left: Histogram of last iteration's h_factors =====
    ax3.set_title("Distribution of h_factors (Last Iteration)", fontsize=14)

    # Get last iteration's h_factors
    last_h_factors = []
    for param_name in param_names:
        last_h_factors.append(optimization_results[-1]["h_factor"][param_name])

    # Create histogram
    ax3.hist(
        last_h_factors,
        bins=min(30, len(param_names) // 2),
        alpha=0.7,
        edgecolor="black",
    )
    ax3.set_xlabel("h_ij Value", fontsize=12)
    ax3.set_ylabel("Number of occurrences", fontsize=12)
    ax3.set_yscale("log")
    ax3.grid(True, alpha=0.3)

    # ===== Lower Left: Histogram of last iteration's h_factors =====
    ax4.set_title("Distribution of h_factors (Last Iteration)", fontsize=14)

    # Get last iteration's h_factors
    last_conbined_factors = []
    for param_name in param_names:
        last_conbined_factors.append(
            optimization_results[-1]["g_approx_over_g_factor"][param_name]
            * optimization_results[-1]["h_factor"][param_name]
        )

    # Create histogram
    ax4.hist(
        last_conbined_factors,
        bins=min(30, len(param_names) // 2),
        alpha=0.7,
        edgecolor="black",
    )
    ax4.set_xlabel("g_ij^approx/g * h_ij Value", fontsize=12)
    ax4.set_ylabel("Number of occurrences", fontsize=12)
    ax4.set_yscale("log")
    ax4.grid(True, alpha=0.3)

    # Adjust layout and show
    plt.tight_layout()
    plt.show(block=True)
