from copy import deepcopy

from scaled_system_definition import ScaledSystem, get_prop_to

from qdesignoptimizer.design_analysis_types import OptTarget
from qdesignoptimizer.utils.names_parameters import UNITLESS, mode


def get_opt_target(i, j, sys: ScaledSystem) -> OptTarget:
    """Return the optimization target for parameter with composite index a=(i,j)."""
    mode_name = mode(f"{i},{j}")
    return OptTarget(
        target_param_type=UNITLESS,
        involved_modes=[mode_name],
        design_var=f"dv_{i},{j}",
        design_var_constraint={"larger_than": 1e-8, "smaller_than": 100},
        prop_to=get_prop_to(i, j, sys),
        independent_target=f"{i}",
    )


def aggregate_results(
    optimization_results: list, sys: ScaledSystem, minimization_results: any
):
    design_variables = deepcopy(sys.get_flattened_x())
    system_optimized_params = deepcopy(sys.get_flattened_yk())
    h_ij_factor_at_yk_xk = deepcopy(sys.get_flattened_h_ij_factor_at_yk_xk())
    g_ij_approx_factor_at_yk_xk = deepcopy(
        sys.get_flattened_g_ij_approx_factor_at_yk_xk()
    )
    g_ij_approx_factor_at_ytarget_xk = deepcopy(
        sys.get_flattened_g_ij_approx_factor_at_ytarget_xk()
    )
    g_ij_factor_at_yk_xk = deepcopy(sys.get_flattened_g_ij_factor_at_yk_xk())

    iteration_result = {}
    iteration_result["design_variables"] = design_variables
    iteration_result["system_optimized_params"] = system_optimized_params
    iteration_result["minimization_results"] = minimization_results
    iteration_result["h_ij_factor_at_yk_xk"] = h_ij_factor_at_yk_xk
    iteration_result["g_ij_approx_factor_at_yk_xk"] = g_ij_approx_factor_at_yk_xk
    iteration_result["g_ij_approx_factor_at_ytarget_xk"] = (
        g_ij_approx_factor_at_ytarget_xk
    )
    iteration_result["g_ij_factor_at_yk_xk"] = g_ij_factor_at_yk_xk

    optimization_results.append(iteration_result)
