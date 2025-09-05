from copy import deepcopy
from typing import List

import scipy
import scipy.optimize

from qdesignoptimizer.design_analysis_types import OptTarget
from qdesignoptimizer.logger import log
from qdesignoptimizer.utils.names_parameters import (
    CAPACITANCE,
    NONLIN,
    DesignVariable,
    Parameter,
    param,
    param_capacitance,
    param_nonlin,
)
from qdesignoptimizer.utils.utils import get_value_and_unit


class ANModOptimizer:
    """
    Approximate Nonlinear Model-driven Optimizer (ANModOptimizer)


    """

    def __init__(
        self,
        opt_targets: List[OptTarget],
        system_target_params: dict[Parameter, float | int],
        adjustment_rate: float = 1,
        minimization_tol: float = 1e-12,
    ):
        self.opt_targets = opt_targets
        self.system_target_params = system_target_params
        self.adjustment_rate = adjustment_rate
        self.minimization_tol = minimization_tol

    def _minimize_for_design_vars(
        self,
        targets_to_minimize_for: List[OptTarget],
        all_design_var_current: dict,
        all_design_var_updated: dict,
        all_parameters_current: dict,
        all_parameters_targets_met: dict,
    ):
        """Minimize the cost function to find the optimal design variables to reach the target.
        The all_design_var_updated variable is automatically updated with the optimal design variables during the minimization.
        """
        design_var_names_to_minimize = [
            target.design_var for target in targets_to_minimize_for
        ]
        bounds_for_targets = [
            (
                get_value_and_unit(target.design_var_constraint["larger_than"])[0],
                get_value_and_unit(target.design_var_constraint["smaller_than"])[0],
            )
            for target in targets_to_minimize_for
        ]

        init_design_var = []
        init_design_var = [
            all_design_var_current[name] for name in design_var_names_to_minimize
        ]

        def cost_function(design_var_vals_updated):
            """Cost function to minimize.

            Args:
                ordered_design_var_vals_updated (List[float]): list of updated design variable values
            """
            for idx, name in enumerate(design_var_names_to_minimize):
                all_design_var_updated[name] = design_var_vals_updated[idx]
            cost = 0
            for target in targets_to_minimize_for:
                Q_k1_i = (
                    self._get_parameter_value(target, all_parameters_current)
                    * target.prop_to(all_parameters_targets_met, all_design_var_updated)
                    / target.prop_to(all_parameters_current, all_design_var_current)
                )
                cost += (
                    (
                        Q_k1_i
                        / self._get_parameter_value(target, all_parameters_targets_met)
                    )
                    - 1
                ) ** 2

            return cost

        min_result = scipy.optimize.minimize(
            cost_function,
            init_design_var,
            tol=self.minimization_tol,
            bounds=bounds_for_targets,
        )

        for idx, name in enumerate(design_var_names_to_minimize):
            if (
                all_design_var_updated[name] == bounds_for_targets[idx][0]
                or all_design_var_updated[name] == bounds_for_targets[idx][1]
            ):
                log.warning(
                    f"The optimized value for the design variable {name}: {all_design_var_updated[name]} is at the bounds. Consider changing the bounds or making the initial design closer to the optimal one."
                )

        final_cost = cost_function(
            [all_design_var_updated[name] for name in design_var_names_to_minimize]
        )
        return {
            "result": min_result,
            "targets_to_minimize_for": [
                target.design_var for target in targets_to_minimize_for
            ],
            "final_cost": final_cost,
        }

    def calculate_target_design_var(
        self,
        system_optimized_params: dict[Parameter, float | int],
        variables_with_units: dict[DesignVariable, str],
    ) -> dict:
        """Calculate the new design value for the optimization targets."""
        minimization_results: list[dict] = []

        # TODO: Refactor to avoid deepcopies
        system_params_current = deepcopy(system_optimized_params)
        system_params_targets_met = self._get_system_params_targets_met(
            system_optimized_params
        )

        # TODO: Refactor to avoid deepcopies
        design_vars_current_str = deepcopy(variables_with_units)

        if not self.is_system_optimized_params_initialized:
            self.is_system_optimized_params_initialized = True
            return design_vars_current_str

        # Fetch the numeric values of the design variables
        design_vars_current = {}
        design_vars_updated = {}
        units = {}
        for design_var, val_unit in design_vars_current_str.items():
            val, unit = get_value_and_unit(val_unit)
            design_vars_current[design_var] = val
            design_vars_updated[design_var] = val
            units[design_var] = unit

        independent_targets = [
            target for target in self.opt_targets if target.independent_target
        ]

        if independent_targets:
            for independent_target in independent_targets:
                minimization_result = self._minimize_for_design_vars(
                    [independent_target],
                    design_vars_current,
                    design_vars_updated,
                    system_params_current,
                    system_params_targets_met,
                )
                minimization_results.append(minimization_result)

        dependent_targets = [
            target for target in self.opt_targets if not target.independent_target
        ]
        if dependent_targets:
            minimization_result = self._minimize_for_design_vars(
                dependent_targets,
                design_vars_current,
                design_vars_updated,
                system_params_current,
                system_params_targets_met,
            )
            minimization_results.append(minimization_result)

        # Stitch back the unit of the design variable values
        design_vars_updated_constrained_str = {}
        for target in self.opt_targets:
            design_var_name = target.design_var
            design_vars_updated_val_and_unit = (
                f"{design_vars_updated[design_var_name]} {units[design_var_name]}"
            )
            constrained_val_and_unit = self._constrain_design_value(
                design_vars_current_str[design_var_name],
                design_vars_updated_val_and_unit,
                target.design_var_constraint,
            )
            design_vars_updated_constrained_str[design_var_name] = (
                constrained_val_and_unit
            )
        return design_vars_updated_constrained_str, minimization_results

    def _constrain_design_value(
        self,
        design_value_old: str,
        design_value_new: str,
        design_var_constraint: dict[str, str],
    ) -> str:
        """Constrain design value.

        Args:
            design_value (str): design value to be constrained
            design_var_constraint (dict[str, str]): design variable constraint, example {'min': '10 um', 'max': '100 um'}
        """
        d_val_o, d_unit = get_value_and_unit(design_value_old)
        d_val_n, d_unit = get_value_and_unit(design_value_new)

        d_val = self._apply_adjustment_rate(d_val_n, d_val_o, self.adjustment_rate)

        c_val_to_be_smaller_than, c_unit_to_be_smaller_than = get_value_and_unit(
            design_var_constraint["smaller_than"]
        )
        c_val_to_be_larger_than, c_unit_to_be_larger_than = get_value_and_unit(
            design_var_constraint["larger_than"]
        )
        assert (
            d_unit == c_unit_to_be_smaller_than == c_unit_to_be_larger_than
        ), f"Units of design_value {design_value_old} and constraint {design_var_constraint} must match"
        if d_val > c_val_to_be_smaller_than:
            design_value = c_val_to_be_smaller_than
        elif d_val < c_val_to_be_larger_than:
            design_value = c_val_to_be_larger_than
        else:
            design_value = d_val

        return f"{design_value} {d_unit}"

    @staticmethod
    def _apply_adjustment_rate(
        new_val: float | int, old_val: float | int, rate: float | int
    ) -> float:
        """Low pass filter for adjustment rate.

        Args:
            new_val (float): new value
            old_val (float): old value
            rate (float): rate of adjustment
        """
        return rate * new_val + (1 - rate) * old_val

    @staticmethod
    def _get_parameter_value(target: OptTarget, system_params: dict) -> float:
        """Return value of parameter from target specification."""
        if target.target_param_type == NONLIN:
            mode1, mode2 = target.involved_modes
            current_value = system_params[param_nonlin(mode1, mode2)]
        elif target.target_param_type == CAPACITANCE:
            capacitance_name_1, capacitance_name_2 = target.involved_modes
            current_value = system_params[
                param_capacitance(capacitance_name_1, capacitance_name_2)
            ]
        else:
            mode = target.involved_modes[0]
            current_value = system_params[param(mode, target.target_param_type)]  # type: ignore
        return current_value

    def _get_system_params_targets_met(
        self, system_optimized_params: dict[Parameter, float | int]
    ) -> dict[str, float]:
        """Return organized dictionary of parameters given target specifications and current status."""
        system_params_targets_met = deepcopy(system_optimized_params)
        for target in self.opt_targets:
            if target.target_param_type == NONLIN:
                mode1, mode2 = target.involved_modes
                system_params_targets_met[param_nonlin(mode1, mode2)] = (
                    self._get_parameter_value(target, self.system_target_params)
                )
            elif target.target_param_type == CAPACITANCE:
                capacitance_name_1, capacitance_name_2 = target.involved_modes
                system_params_targets_met[
                    param_capacitance(capacitance_name_1, capacitance_name_2)
                ] = self._get_parameter_value(target, self.system_target_params)
            else:
                mode_name = target.involved_modes[0]
                system_params_targets_met[
                    param(mode_name, target.target_param_type)  # type: ignore
                ] = self._get_parameter_value(target, self.system_target_params)
        return system_params_targets_met
