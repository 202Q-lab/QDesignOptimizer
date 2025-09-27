import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import EngFormatter, FuncFormatter, LogFormatter, LogLocator

from qdesignoptimizer.sim_plot_progress import plot_progress


def plot_all_convergence_ratios(optimization_results, system_target_params):
    """
    Plot convergence analysis.

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
    iterations = np.array(range(n_iterations)) + 1
    plt.rcParams["axes.formatter.useoffset"] = False
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 35,
            "mathtext.fontset": "stix",
        }
    )
    # Create 2x2 subplot matrix
    fig, ((ax_1, ax_2, ax_3, ax_4), (ax_5, ax_6, ax_7, ax_8)) = plt.subplots(
        2, 4, figsize=(3.5 * 6.7, 3.5 * 3.35)
    )  #    plot evolution of design variables
    gs = fig.add_gridspec(
        1, 1, wspace=0.55, hspace=0.15, left=0.16, right=0.99, bottom=0.28, top=0.97
    )
    axes = [ax_1, ax_2, ax_3, ax_4, ax_5, ax_6, ax_7, ax_8]

    ax_1.text(0.8, 0.85, "(a)", transform=ax_1.transAxes)
    ax_2.text(0.8, 0.85, "(b)", transform=ax_2.transAxes)
    ax_3.text(0.8, 0.85, "(c)", transform=ax_3.transAxes)
    ax_4.text(0.8, 0.85, "(d)", transform=ax_4.transAxes)
    ax_5.text(0.8, 0.85, "(e)", transform=ax_5.transAxes)
    ax_6.text(0.8, 0.85, "(f)", transform=ax_6.transAxes)
    ax_7.text(0.8, 0.85, "(g)", transform=ax_7.transAxes)
    ax_8.text(0.8, 0.85, "(h)", transform=ax_8.transAxes)
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
            linewidth=1.5 * 2,
            markersize=4 * 2,
            label=param_name,
            alpha=0.7,
        )

    ax7.set_xlabel("Iteration $k$")
    ax7.set_ylabel("Parameters $y_{i,j}$")
    ax7.set_yscale("log")

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
            linewidth=1.5 * 2,
            markersize=4 * 2,
            label=dv_name,
            alpha=0.7,
        )

    ax8.set_xlabel("Iteration $k$")
    ax8.set_ylabel("Design variables $x_{i,j}$")
    ax8.set_yscale("log")
    ax8.set_yticks([0.5, 1.0])

    ax8.grid(True, alpha=0.3)

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
            linewidth=1.5 * 2,
            markersize=4 * 2,
            label=param_name,
            alpha=0.7,
        )

    ax1.set_xlabel("Iteration $k$")
    ax1.set_ylabel(r"Ratio $y_{i,j}\ /\ y_{i,j}^{target}$")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

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
            linewidth=1.5 * 2,
            markersize=4 * 2,
            label=dv_name,
            alpha=0.7,
        )

    ax2.set_xlabel("Iteration $k$")
    ax2.set_ylabel(r"Ratio $x_{i,j}\ / \ x_{i,j}^{k=10}$")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)
    ax2.set_yticks([0.5, 1.0])

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
    ax3.set_xlabel(r"Local $g_{i,j}^{approx, k=10}\ /\ g_{i,j}^{k=10}$")
    ax3.set_ylabel("Count")
    ax3.set_yscale("log")
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks([0.8, 1.0, 1.2])

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
    ax4.set_xlabel(r"Global $h_{i,j}^{k=10}$")
    ax4.set_ylabel("Count")
    ax4.set_yscale("log")
    ax4.grid(True, alpha=0.3)

    last_conbined_factors = []
    for param_name in param_names:
        last_conbined_factors.append(
            (
                optimization_results[-1]["g_ij_factor_at_yk_xk"][param_name]
                * optimization_results[-1]["h_ij_factor_at_yk_xk"][param_name]
            )
            / optimization_results[-1]["g_ij_approx_factor_at_yk_xk"][param_name]
        )

    largest_approximation_slope = 0
    largest_approximation_slope_for_iteration_all = []
    for iter in range(n_iterations):
        largest_approximation_slope_for_iteration = 0
        for param_name in param_names:
            slope = abs(
                optimization_results[iter]["g_ij_factor_at_yk_xk"][param_name]
                * optimization_results[iter]["h_ij_factor_at_yk_xk"][param_name]
                / optimization_results[iter]["g_ij_approx_factor_at_yk_xk"][param_name]
                - 1
            )
            if slope > largest_approximation_slope:
                largest_approximation_slope = slope
            if slope > largest_approximation_slope_for_iteration:
                largest_approximation_slope_for_iteration = slope
        largest_approximation_slope_for_iteration_all.append(
            {
                "iteration": iter,
                "param": param_name,
                "slope": largest_approximation_slope_for_iteration,
            }
        )
    ax5.hist(
        last_conbined_factors,
        bins=min(30, len(param_names) // 2),
        alpha=0.7,
        edgecolor="black",
    )
    ax5.set_xlabel(r"Error factor $f_{i,j}^{error, k=10}$")
    ax5.set_ylabel("Count")
    ax5.set_yscale("log")
    ax5.grid(True, alpha=0.3)
    ax5.set_xticks([1.0, 1.2, 1.4])

    max_start = 0
    # Compare the error made by the model given by
    for param_name in param_names:
        target_val = system_target_params[param_name]
        approx_ratio = []
        for iter in range(n_iterations):
            error_ratio = (
                abs(
                    optimization_results[iter]["system_optimized_params"][param_name]
                    - target_val
                )
                / target_val
            )
            max_start = max(max_start, error_ratio)
            approx_ratio.append(error_ratio)

        ax6.plot(
            iterations,
            approx_ratio,
            "o-",
            linewidth=1.5 * 2,
            markersize=4 * 2,
            label=param_name,
            alpha=0.7,
        )
    largest_slope_last_iteration = largest_approximation_slope_for_iteration_all[-1][
        "slope"
    ]
    ax6.plot(
        iterations,
        4 * max_start * largest_slope_last_iteration ** (iterations - 1),
        "k--",
        linewidth=1.5 * 2,
        label="Estimated slope",
    )
    ax6.set_xlabel("Iteration $k$")
    ax6.set_ylabel(r"Ratio |$y_{i,j}^{k}-y_{i,j}^{target} $| / $y_{i,j}^{target}$ ")
    ax6.set_yscale("log")
    ax6.grid(True, alpha=0.3)
    ax6.set_yticks([1e-8, 1e-4, 1])

    from matplotlib.ticker import FuncFormatter

    def plain_number(y, _pos):
        return f"{y:g}"  # prints 0.9, 1, 10, 100, ...

    for ax in [ax_1, ax_2, ax_3, ax_4, ax_5, ax_6, ax_7, ax_8]:
        # pick one:
        # ax.yaxis.set_major_locator(dense_locator)   # or base_locator
        if ax not in [ax_8]:
            ax.yaxis.set_major_formatter(FuncFormatter(plain_number))
        # hide minor tick labels
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))

    print(largest_approximation_slope_for_iteration_all)
    plt.tight_layout()
    plt.show(block=True)

    plt.plot(
        iterations,
        [item["slope"] for item in largest_approximation_slope_for_iteration_all],
        "o-",
    )
    plt.show(block=True)
