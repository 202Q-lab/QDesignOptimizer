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
    dv_names = list(optimization_results[0]["design_variables"].keys())
    n_iterations = len(optimization_results)
    iterations = range(n_iterations)

    # Create 2x2 subplot matrix
    fig, ((ax_1, ax_2), (ax_3, ax_4), (ax_5, ax_6), (ax_7, ax_8)) = plt.subplots(
        4, 2, figsize=(16, 12)
    )  #    plot evolution of design variables
    # plot f_k+1 / f_k
    ax7, ax8, ax1, ax2, ax3, ax4, ax5, ax6 = (
        ax_1,
        ax_2,
        ax_3,
        ax_4,
        ax_5,
        ax_6,
        ax_7,
        ax_8,
    )

    # ===== Upper Right: y/y_target convergence =====
    ax7.set_title("y vs Iterations", fontsize=14)

    for param_name in param_names:
        ratios = []
        target_val = system_target_params[param_name]

        for iter_result in optimization_results:
            optimized_val = iter_result["system_optimized_params"][param_name]
            ratio = optimized_val
            ratios.append(ratio)

        ax7.plot(
            iterations,
            ratios,
            "o-",
            linewidth=1.5,
            markersize=4,
            label=param_name,
            alpha=0.7,
        )

    ax7.set_xlabel("Iteration", fontsize=12)
    ax7.set_ylabel("Parameters y", fontsize=12)
    ax7.set_yscale("log")
    ax7.grid(True, alpha=0.3)

    ax8.set_title("Convergence: x vs Iterations", fontsize=14)

    for dv_name in dv_names:
        ratios = []

        for iter_result in optimization_results:
            optimized_val = iter_result["design_variables"][dv_name]
            ratio = optimized_val
            ratios.append(ratio)

        ax8.plot(
            iterations,
            ratios,
            "o-",
            linewidth=1.5,
            markersize=4,
            label=dv_name,
            alpha=0.7,
        )

    ax8.set_xlabel("Iteration", fontsize=12)
    ax8.set_ylabel("Design variables x", fontsize=12)
    ax8.set_yscale("log")
    ax8.grid(True, alpha=0.3)

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

    ax2.set_title("Convergence: x/x_9 vs Iterations", fontsize=14)

    for dv_name in dv_names:
        ratios = []
        final_val = optimization_results[-1]["design_variables"][dv_name]

        for iter_result in optimization_results:
            optimized_val = iter_result["design_variables"][dv_name]
            ratio = optimized_val / final_val
            ratios.append(ratio)

        ax2.plot(
            iterations,
            ratios,
            "o-",
            linewidth=1.5,
            markersize=4,
            label=dv_name,
            alpha=0.7,
        )

    ax2.set_xlabel("Iteration", fontsize=12)
    ax2.set_ylabel("x/x_target Ratio", fontsize=12)
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    ax3.set_title("Distribution of g_ij^approx/g_ij (Last Iteration)", fontsize=14)

    last_g_approx_factors = []
    for param_name in param_names:
        last_g_approx_factors.append(
            optimization_results[-1]["g_ij_approx_factor_at_yk_xk"][param_name]
            / optimization_results[-1]["g_ij_factor_at_yk_xk"][param_name]
        )

    ax3.hist(
        last_g_approx_factors,
        bins=min(30, len(param_names) // 2),
        alpha=0.7,
        edgecolor="black",
    )
    ax3.set_xlabel("g_ij^approx(yk, xk)/g_ij(yk, xk) Value", fontsize=12)
    ax3.set_ylabel("Number of occurrences", fontsize=12)
    ax3.set_yscale("log")
    ax3.grid(True, alpha=0.3)

    ax4.set_title("Distribution of h_factors (Last Iteration)", fontsize=14)

    last_h_factors = []
    for param_name in param_names:
        last_h_factors.append(
            optimization_results[-1]["h_ij_factor_at_yk_xk"][param_name]
        )

    ax4.hist(
        last_h_factors,
        bins=min(30, len(param_names) // 2),
        alpha=0.7,
        edgecolor="black",
    )
    ax4.set_xlabel("h_ij(yk, xk) Value", fontsize=12)
    ax4.set_ylabel("Number of occurrences", fontsize=12)
    ax4.set_yscale("log")
    ax4.grid(True, alpha=0.3)

    ax5.set_title("Distribution of h_factors (Last Iteration)", fontsize=14)

    last_conbined_factors = []
    for param_name in param_names:
        last_conbined_factors.append(
            optimization_results[-1]["g_ij_approx_factor_at_yk_xk"][param_name]
            / (
                optimization_results[-1]["g_ij_factor_at_yk_xk"][param_name]
                * optimization_results[-1]["h_ij_factor_at_yk_xk"][param_name]
            )
        )

    ax5.hist(
        last_conbined_factors,
        bins=min(30, len(param_names) // 2),
        alpha=0.7,
        edgecolor="black",
    )
    ax5.set_xlabel("g_ij^approx(yk, xk) / (g_ij(yk, xk)*h_ij(yk, xk))", fontsize=12)
    ax5.set_ylabel("Number of occurrences", fontsize=12)
    ax5.set_yscale("log")
    ax5.grid(True, alpha=0.3)

    ax6.set_title("Distribution of h_factors (Last Iteration)", fontsize=14)

    # Compare the error made by the model given by
    # f^approx_ij,(y^target, x^(k+1)) * f^eps(y^k,x^k) / ( f^_{ij,(y^(k+1), x^(k+1))} * f^eps(y^k+1),x^(k+1)) )
    for param_name in param_names:
        target_val = system_target_params[param_name]
        approx_ratio = []
        for iter in range(n_iterations):
            if iter == n_iterations - 1:
                continue  # Skip last iteration for k+1 access
            approx_ratio.append(
                optimization_results[iter + 1]["g_ij_approx_factor_at_ytarget_xk"][
                    param_name
                ]
                * optimization_results[iter]["system_optimized_params"][param_name]
                / (
                    optimization_results[iter]["g_ij_approx_factor_at_yk_xk"][
                        param_name
                    ]
                    * optimization_results[iter + 1]["system_optimized_params"][
                        param_name
                    ]
                )
            )

        ax6.plot(
            iterations[0:-1],
            approx_ratio,
            "o-",
            linewidth=1.5,
            markersize=4,
            label=param_name,
            alpha=0.7,
        )
    ax6.set_xlabel(
        "f^approx_ij,(y^target, x^(k+1)) * f^eps(y^k,x^k) / f^approx_{ij,(y^(k+1), x^(k+1))} * f^eps(y^k+1),x^(k+1))",
        fontsize=12,
    )
    ax6.set_ylabel("Number of occurrences", fontsize=12)
    ax6.set_yscale("log")
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show(block=True)
