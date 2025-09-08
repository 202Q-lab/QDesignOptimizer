import matplotlib.pyplot as plt

from qdesignoptimizer.sim_plot_progress import plot_progress


def plot_all_convergence_ratios(optimization_results, system_target_params):
    """
    Plot all y/y_target ratios in the same graph as a function of iterations.

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

    # Plot each parameter's ratio
    for param_name in param_names:

        ratios = []
        target_val = system_target_params[param_name]

        for iter_result in optimization_results:
            optimized_val = iter_result["system_optimized_params"][param_name]
            ratio = optimized_val / target_val
            ratios.append(ratio)

        plt.plot(
            iterations,
            ratios,
            "o-",
            linewidth=1.5,
            markersize=4,
            label=param_name,
            alpha=0.7,
        )

    plt.axhline(
        y=1.0,
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.8,
        label="Target (ratio=1)",
    )
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Ratio y/y_target", fontsize=12)
    plt.yscale("log")
    plt.title("Convergence: All Parameters y/y_target vs Iterations", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show(block=True)
