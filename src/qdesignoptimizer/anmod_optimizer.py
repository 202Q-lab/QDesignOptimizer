from copy import deepcopy
from typing import List

import scipy
import scipy.optimize

import qdesignoptimizer
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
        self.anmod_version = qdesignoptimizer.__version__

    def _minimize_for_control_vars(
        self,
        targets_to_minimize_for: List[OptTarget],
        all_control_vars_current: dict,
        all_control_vars_updated: dict,
        all_parameters_current: dict,
        all_parameters_targets_met: dict,
    ):
        """Minimize the cost function to find the optimal control variables to reach the target.
        The all_control_vars_updated variable is automatically updated with the optimal control variables during the minimization.
        """
        control_var_names_to_minimize = [
            target.control_var for target in targets_to_minimize_for
        ]
        bounds_for_targets = [
            (
                get_value_and_unit(target.control_var_constraint["larger_than"])[0],
                get_value_and_unit(target.control_var_constraint["smaller_than"])[0],
            )
            for target in targets_to_minimize_for
        ]

        init_control_vars = []
        init_control_vars = [
            all_control_vars_current[name] for name in control_var_names_to_minimize
        ]

        def cost_function(control_var_vals_updated):
            """Cost function to minimize.

            Args:
                control_var_vals_updated (List[float]): list of updated control variable values
            """
            for idx, name in enumerate(control_var_names_to_minimize):
                all_control_vars_updated[name] = control_var_vals_updated[idx]
            cost = 0
            for target in targets_to_minimize_for:
                Q_k1_i = (
                    self._get_parameter_value(target, all_parameters_current)
                    * target.prop_to(all_parameters_targets_met, all_control_vars_updated)
                    / target.prop_to(all_parameters_current, all_control_vars_current)
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
            init_control_vars,
            tol=self.minimization_tol,
            bounds=bounds_for_targets,
        )

        for idx, name in enumerate(control_var_names_to_minimize):
            if (
                all_control_vars_updated[name] == bounds_for_targets[idx][0]
                or all_control_vars_updated[name] == bounds_for_targets[idx][1]
            ):
                log.warning(
                    f"The optimized value for the control variable {name}: {all_control_vars_updated[name]} is at the bounds. Consider changing the bounds or making the initial values closer to the optimal ones."
                )

        final_cost = cost_function(
            [all_control_vars_updated[name] for name in control_var_names_to_minimize]
        )
        return {
            "result": min_result,
            "targets_to_minimize_for": [
                target.control_var for target in targets_to_minimize_for
            ],
            "final_cost": final_cost,
        }

    def calculate_target_control_vars(
        self,
        system_optimized_params: dict[Parameter, float | int],
        variables_with_units: dict[DesignVariable, str],
    ) -> tuple[dict, list[dict]]:
        """Calculate the new control variable values for the optimization targets."""
        minimization_results: list[dict] = []

        # TODO: Refactor to avoid deepcopies
        system_params_current = deepcopy(system_optimized_params)
        system_params_targets_met = self._get_system_params_targets_met(
            system_optimized_params
        )

        # TODO: Refactor to avoid deepcopies
        control_vars_current_str = deepcopy(variables_with_units)

        # Fetch the numeric values of the control variables
        control_vars_current = {}
        control_vars_updated = {}
        units = {}
        for control_var, val_unit in control_vars_current_str.items():
            val, unit = get_value_and_unit(val_unit)
            control_vars_current[control_var] = val
            control_vars_updated[control_var] = val
            units[control_var] = unit

        minimization_target_groups = self.group_targets(self.opt_targets)

        for targets in minimization_target_groups:
            minimization_result = self._minimize_for_control_vars(
                targets,
                control_vars_current,
                control_vars_updated,
                system_params_current,
                system_params_targets_met,
            )
            minimization_results.append(minimization_result)

        # Stitch back the unit of the control variable values
        control_vars_updated_constrained_str = {}
        for target in self.opt_targets:
            control_var_name = target.control_var
            control_vars_updated_val_and_unit = (
                f"{control_vars_updated[control_var_name]} {units[control_var_name]}"
            )
            constrained_val_and_unit = self._constrain_control_value(
                control_vars_current_str[control_var_name],
                control_vars_updated_val_and_unit,
                target.control_var_constraint,
            )
            control_vars_updated_constrained_str[control_var_name] = (
                constrained_val_and_unit
            )
        return control_vars_updated_constrained_str, minimization_results

    @staticmethod
    def group_targets(optimization_targets: List[OptTarget]) -> List[List[OptTarget]]:
        """Group optimization targets based on their independent_target attribute."""
        target_groups: dict[str, List[OptTarget]] = {}
        dependent_targets: list[OptTarget] = []
        minimization_targets: list[list[OptTarget]] = []

        for target in optimization_targets:
            if target.independent_target is True:
                minimization_targets.append([target])
            elif target.independent_target is False:
                dependent_targets.append(target)
            elif isinstance(target.independent_target, str):
                if target.independent_target not in target_groups:
                    target_groups[target.independent_target] = []
                target_groups[target.independent_target].append(target)
            else:
                raise ValueError(
                    f"Invalid value for independent_target: {target.independent_target}. Must be bool or str."
                )

        minimization_targets.extend(list(target_groups.values()))
        if len(dependent_targets) > 0:
            minimization_targets.append(dependent_targets)

        return minimization_targets

    def _constrain_control_value(
        self,
        value_old: str,
        value_new: str,
        control_var_constraint: dict[str, str],
    ) -> str:
        """Constrain control variable value.

        Args:
            value_old (str): old control variable value to be constrained
            value_new (str): new control variable value to be constrained
            control_var_constraint (dict[str, str]): control variable constraint, example {'larger_than': '10 um', 'smaller_than': '100 um'}
        """
        val_o, unit = get_value_and_unit(value_old)
        val_n, unit = get_value_and_unit(value_new)

        val = self._apply_adjustment_rate(val_n, val_o, self.adjustment_rate)

        c_val_to_be_smaller_than, c_unit_to_be_smaller_than = get_value_and_unit(
            control_var_constraint["smaller_than"]
        )
        c_val_to_be_larger_than, c_unit_to_be_larger_than = get_value_and_unit(
            control_var_constraint["larger_than"]
        )
        assert (
            unit == c_unit_to_be_smaller_than == c_unit_to_be_larger_than
        ), f"Units of control value {value_old} and constraint {control_var_constraint} must match"
        if val > c_val_to_be_smaller_than:
            constrained_value = c_val_to_be_smaller_than
        elif val < c_val_to_be_larger_than:
            constrained_value = c_val_to_be_larger_than
        else:
            constrained_value = val

        return f"{constrained_value} {unit}"

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
