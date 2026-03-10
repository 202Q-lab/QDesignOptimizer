def check_convergence(optimization_results, system_target_params, tolerance=0.001):
    """
    Check if all parameters are within the specified tolerance of their targets.

    Parameters
    ----------
    optimization_results : List[Dict]
        List of optimization results from each iteration
    system_target_params : Dict[str, float]
        Target parameter values
    tolerance : float, default=0.001
        Convergence tolerance (0.001 = 0.1%)

    Returns
    -------
    convergence_iteration : int or None
        Iteration number when all parameters first converged, None if not converged
    convergence_status : Dict
        Dictionary with convergence information per iteration
    """
    convergence_status = {}
    convergence_iteration = None

    for iteration, result in enumerate(optimization_results):
        system_optimized_params = result["system_optimized_params"]

        # Calculate relative errors for all parameters
        relative_errors = {}
        max_relative_error = 0.0
        converged_count = 0
        total_params = len(system_target_params)

        for param_name, target_val in system_target_params.items():
            optimized_val = system_optimized_params[param_name]

            # Calculate relative error: |optimized - target| / |target|
            if target_val != 0:
                relative_error = abs(optimized_val - target_val) / abs(target_val)
            else:
                # Handle zero target case
                relative_error = abs(optimized_val)

            relative_errors[param_name] = relative_error
            max_relative_error = max(max_relative_error, relative_error)

            # Check if this parameter has converged
            if relative_error <= tolerance:
                converged_count += 1

        # Store convergence status for this iteration
        convergence_status[iteration] = {
            "max_relative_error": max_relative_error,
            "converged_count": converged_count,
            "total_params": total_params,
            "convergence_percentage": (converged_count / total_params) * 100,
            "all_converged": converged_count == total_params,
            "relative_errors": relative_errors,
        }

        # Check if this is the first iteration where all parameters converged
        if convergence_iteration is None and converged_count == total_params:
            convergence_iteration = iteration

        # Print progress
        print(
            f"Iteration {iteration}: {converged_count}/{total_params} parameters converged "
            f"({convergence_status[iteration]['convergence_percentage']:.1f}%), "
            f"max error: {max_relative_error:.6f}"
        )

    return convergence_iteration, convergence_status


def print_convergence_summary(
    convergence_iteration, convergence_status, tolerance=0.001
):
    """Print a summary of convergence results."""
    print("\n" + "=" * 60)
    print("CONVERGENCE SUMMARY")
    print("=" * 60)

    if convergence_iteration is not None:
        print(f"✅ ALL PARAMETERS CONVERGED at iteration {convergence_iteration}")
        print(f"   Tolerance: {tolerance*100:.1f}% relative error")
        print(f"   Total iterations to convergence: {convergence_iteration + 1}")
    else:
        print(f"❌ NOT ALL PARAMETERS CONVERGED")
        print(f"   Tolerance: {tolerance*100:.1f}% relative error")

        # Find the iteration with best convergence
        best_iteration = max(
            convergence_status.keys(),
            key=lambda k: convergence_status[k]["converged_count"],
        )
        best_status = convergence_status[best_iteration]

        print(f"   Best iteration: {best_iteration}")
        print(
            f"   Best convergence: {best_status['converged_count']}/{best_status['total_params']} "
            f"({best_status['convergence_percentage']:.1f}%)"
        )
        print(
            f"   Max error in best iteration: {best_status['max_relative_error']:.6f}"
        )

    # Show final iteration stats
    final_iteration = max(convergence_status.keys())
    final_status = convergence_status[final_iteration]
    print(f"\nFinal iteration ({final_iteration}) results:")
    print(
        f"   Converged parameters: {final_status['converged_count']}/{final_status['total_params']} "
        f"({final_status['convergence_percentage']:.1f}%)"
    )
    print(f"   Max relative error: {final_status['max_relative_error']:.6f}")
    print("=" * 60)
