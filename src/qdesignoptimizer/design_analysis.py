import json
from copy import deepcopy
from typing import List, Optional

import numpy as np
import pandas as pd
import pyEPR as epr
import scipy
import scipy.optimize
from pyaedt import Hfss
from qiskit_metal.analyses.quantization import EPRanalysis
from qiskit_metal.analyses.quantization.energy_participation_ratio import EPRanalysis

from qdesignoptimizer.design_analysis_types import (
    DesignAnalysisState,
    MeshingMap,
    MiniStudy,
    OptTarget,
)
from qdesignoptimizer.logging import dict_log_format, log
from qdesignoptimizer.sim_capacitance_matrix import CapacitanceMatrixStudy
from qdesignoptimizer.sim_plot_progress import plot_progress
from qdesignoptimizer.utils.names_parameters import (
    CAPACITANCE_MATRIX_ELEMENTS,
    FREQ,
    KAPPA,
    NONLIN,
    PURCELL_LIMIT_T1,
    mode,
    param,
    param_capacitance,
    param_nonlin,
)
from qdesignoptimizer.utils.utils import get_value_and_unit, get_version_from_pyproject


class DesignAnalysis:
    """Class for DesignAnalysis.

    Args:
        state (DesignAnalysisSetup): DesignAnalysisState object
        mini_study (MiniStudy): MiniStudy object
        opt_targets (List[OptTarget]): list of OptTarget objects
        save_path (str): path to save results
        update_design_variables (bool): update parameters
        plot_settings (dict): plot settings for progress plots
        meshing_map (List[MeshingMap]): meshing map

    """

    def __init__(
        self,
        state: DesignAnalysisState,
        mini_study: MiniStudy,
        opt_targets: List[OptTarget] = None,
        save_path: str = None,
        update_design_variables: bool = True,
        plot_settings: dict = None,
        meshing_map: List[MeshingMap] = None,
        minimization_tol=1e-12,
    ):
        self.design_analysis_version = get_version_from_pyproject()
        self.design = state.design
        self.eig_solver = EPRanalysis(self.design, "hfss")
        self.eig_solver.sim.setup.name = "Resonator_setup"
        self.renderer = self.eig_solver.sim.renderer
        log.info(
            "self.eig_solver.sim.setup %s", dict_log_format(self.eig_solver.sim.setup)
        )
        self.eig_solver.setup.sweep_variable = "dummy"

        self.mini_study = mini_study
        self.opt_targets = opt_targets
        self.all_design_vars = [target.design_var for target in opt_targets]
        self.render_qiskit_metal = state.render_qiskit_metal
        self.system_target_params = state.system_target_params

        self.is_system_optimized_params_initialized = False
        if state.system_optimized_params is not None:
            sys_opt_param = state.system_optimized_params
            self.is_system_optimized_params_initialized = True
        else:

            def fill_leaves_with_none(nested_dict):
                for key, value in nested_dict.items():
                    if isinstance(value, dict):
                        fill_leaves_with_none(value)
                    else:
                        nested_dict[key] = None
                return nested_dict

            sys_opt_param = fill_leaves_with_none(deepcopy(state.system_target_params))
        self.system_optimized_params = sys_opt_param

        self.save_path = save_path
        self.update_design_variables = update_design_variables
        self.plot_settings = plot_settings
        self.meshing_map = meshing_map
        self.minimization_tol = minimization_tol

        self.optimization_results = []

        self.renderer.start()
        self.renderer.activate_ansys_design(self.mini_study.design_name, "eigenmode")

        self.pinfo = self.renderer.pinfo
        self.setup = self.pinfo.setup
        self.setup.n_modes = len(self.mini_study.modes)
        self.setup.passes = self.mini_study.nbr_passes
        self.setup.delta_f = self.mini_study.delta_f
        self.renderer.options["x_buffer_width_mm"] = self.mini_study.x_buffer_width_mm
        self.renderer.options["y_buffer_width_mm"] = self.mini_study.y_buffer_width_mm
        self.renderer.options["max_mesh_length_port"] = (
            self.mini_study.max_mesh_length_port
        )
        self.renderer.options["keep_originals"] = True

        self._validate_opt_targets()
        assert (
            not self.system_target_params is self.system_optimized_params
        ), "system_target_params and system_optimized_params may not be references to the same object"

    def update_nbr_passes(self, nbr_passes):
        self.mini_study.nbr_passes = nbr_passes
        self.setup.passes = nbr_passes

    def update_nbr_passes_capacitance_ministudies(self, nbr_passes):
        """Updates the number of passes for capacitance matrix studies."""
        if self.mini_study.capacitance_matrix_studies:
            for cap_study in self.mini_study.capacitance_matrix_studies:
                cap_study.nbr_passes = nbr_passes

    def update_delta_f(self, delta_f):
        self.mini_study.delta_f = delta_f
        self.setup.delta_f = delta_f

    def _validate_opt_targets(self):
        """Validate opt_targets."""
        if not self.opt_targets is None:
            for target in self.opt_targets:
                assert (
                    target.design_var in self.design.variables
                ), f"Design variable {target.design_var} not found in design variables."

                if target.system_target_param == PURCELL_LIMIT_T1:
                    assert (
                        len(self.mini_study.capacitance_matrix_studies) != 0
                    ), "capacitance_matrix_studies in ministudy must be populated for Charge line T1 decay study."

                if target.system_target_param == CAPACITANCE_MATRIX_ELEMENTS:
                    capacitance_1 = target.involved_modes[0]
                    capacitance_2 = target.involved_modes[1]
                    assert (
                        CAPACITANCE_MATRIX_ELEMENTS in self.system_target_params
                    ), f"Target for {CAPACITANCE_MATRIX_ELEMENTS} requires {CAPACITANCE_MATRIX_ELEMENTS} in system_target_params."
                    assert (
                        len(target.involved_modes) == 2
                    ), f"Target for {target.system_target_param} expects 2 capacitance names, but {len(target.involved_modes)} were given."
                    assert isinstance(
                        capacitance_1, str
                    ), f"First capacitance name {capacitance_1} must be a string."
                    assert isinstance(
                        capacitance_2, str
                    ), f"Second capacitance name {capacitance_2} must be a string."

                    if (capacitance_2, capacitance_1) in self.system_target_params[
                        CAPACITANCE_MATRIX_ELEMENTS
                    ]:
                        tip = f" The reversed ordered key {(capacitance_2, capacitance_1)} exists, but the order must be consistent."
                    else:
                        tip = ""
                    assert (capacitance_1, capacitance_2) in self.system_target_params[
                        CAPACITANCE_MATRIX_ELEMENTS
                    ], (
                        f"Capacitance names key {(capacitance_1, capacitance_2)} not found in system_target_params[{CAPACITANCE_MATRIX_ELEMENTS}]."
                        + tip
                    )
                elif target.system_target_param == NONLIN:
                    assert (
                        len(target.involved_modes) == 2
                    ), f"Target for {target.system_target_param} expects 2 modes."
                    assert len(self.mini_study.modes) >= len(
                        target.involved_modes
                    ), f"Target for {target.system_target_param} expects \
                        {len(target.involved_modes)} modes but only {self.setup.n_modes} modes will be simulated."
                    for mode in target.involved_modes:
                        assert (
                            mode in self.mini_study.modes
                        ), f"Target mode {mode} \
                            not found in modes which will be simulated."
                else:
                    assert len(self.mini_study.modes) >= len(
                        target.involved_modes
                    ), f"Target for {target.system_target_param} expects \
                        {len(target.involved_modes)} modes but only {self.setup.n_modes} modes will be simulated."
                    for mode in target.involved_modes:
                        assert (
                            mode in self.mini_study.modes
                        ), f"Target mode {mode} \
                            not found in modes which will be simulated."

            design_variables = [target.design_var for target in self.opt_targets]
            assert len(design_variables) == len(
                set(design_variables)
            ), "Design variables must be unique."

    def update_var(self, updated_design_vars: dict, system_optimized_params: dict):
        """Update junction and design variables in mini_study, design, pinfo and."""

        for key, val in {**self.design.variables, **updated_design_vars}.items():
            self.pinfo.design.set_variable(key, val)
            self.design.variables[key] = val

        self.eig_solver.sim.setup.vars = self.design.variables
        self.eig_solver.setup.junctions = self.mini_study.jj_setup

        self.system_optimized_params = {
            **self.system_optimized_params,
            **system_optimized_params,
        }

    def get_fine_mesh_names(self):
        """The fine mesh for the eigenmode study of HFSS can be set in two different ways. First, via an attribute in the component class with function name "get_meshing_names". This function should return the list of part names, e.g. {name}_flux_line_left. The second option is to specify the part names via the meshing_map as keyword in the DesginAnalysis class."""
        finer_mesh_names = []
        for component in self.mini_study.qiskit_component_names:
            if hasattr(self.design.components[component], "get_meshing_names"):
                finer_mesh_names += self.design.components[
                    component
                ].get_meshing_names()
            elif self.meshing_map != None:
                for map in self.meshing_map:
                    if isinstance(
                        self.design.components[component], map.component_class
                    ):
                        finer_mesh_names += map.mesh_names(component)
            else:
                log.info("No fine mesh map was found for " + component)

        return finer_mesh_names

    def get_port_gap_names(self):
        return [f"endcap_{comp}_{name}" for comp, name, _ in self.mini_study.port_list]

    def run_eigenmodes(self):
        """Simulate eigenmodes."""
        self.update_var({}, {})
        self.pinfo.validate_junction_info()

        hfss = Hfss()
        self.renderer.clean_active_design()
        self.render_qiskit_metal(
            self.design, **self.mini_study.render_qiskit_metal_eigenmode_kw_args
        )
        # set hfss wire bonds properties
        self.renderer.options["wb_size"] = self.mini_study.hfss_wire_bond_size
        self.renderer.options["wb_threshold"] = self.mini_study.hfss_wire_bond_threshold
        self.renderer.options["wb_offset"] = self.mini_study.hfss_wire_bond_offset

        # render design in HFSS
        self.renderer.render_design(
            selection=self.mini_study.qiskit_component_names,
            port_list=self.mini_study.port_list,
            open_pins=self.mini_study.open_pins,
        )

        # set custom air bridges
        for component_name in self.mini_study.qiskit_component_names:
            if hasattr(
                self.design.components[component_name], "get_air_bridge_coordinates"
            ):
                for coord in self.design.components[
                    component_name
                ].get_air_bridge_coordinates():
                    hfss.modeler.create_bondwire(
                        coord[0],
                        coord[1],
                        h1=0.005,
                        h2=0.000,
                        alpha=90,
                        beta=45,
                        diameter=0.005,
                        bond_type=0,
                        name="mybox1",
                        matname="aluminum",
                    )

        # set fine mesh
        fine_mesh_names = self.get_fine_mesh_names()
        restrict_mesh = (
            (not not fine_mesh_names)
            and self.mini_study.build_fine_mesh
            and len(self.mini_study.port_list) > 0
        )

        if restrict_mesh:
            self.renderer.modeler.mesh_length(
                "fine_mesh",
                fine_mesh_names,
                MaxLength=self.mini_study.max_mesh_length_lines_to_ports,
                RefineInside=True,
            )

        # run eigenmode analysis
        self.setup.analyze()
        eig_results = self.eig_solver.get_frequencies()
        eig_results["Kappas (kHz)"] = (
            eig_results["Freq. (GHz)"] * 1e9 / eig_results["Quality Factor"] / 1e3
        )
        eig_results["Freq. (Hz)"] = eig_results["Freq. (GHz)"] * 1e9
        eig_results["Kappas (Hz)"] = eig_results["Kappas (kHz)"] * 1e3

        self._update_optimized_params(eig_results)

        return eig_results

    def run_epr(self):
        """Run EPR, requires design with junctions."""
        no_junctions = (
            self.mini_study.jj_setup is None or len(self.mini_study.jj_setup) == 0
        )

        jj_setups_to_include_in_epr = {}
        junction_found = False
        linear_element_found = False
        for key, val in self.mini_study.jj_setup.items():
            # experimental implementation. to be generatized in the future to arbitrary junction potentials
            # this is a simple way to implement a linear potential only
            if "type" in val and val["type"] == "linear":
                linear_element_found = True
                continue
            jj_setups_to_include_in_epr[key] = val
            junction_found = True
        if junction_found:
            self.eig_solver.setup.junctions = jj_setups_to_include_in_epr

            if not no_junctions:
                try:
                    self.eprd = epr.DistributedAnalysis(self.pinfo)
                    self.eig_solver.clear_data()

                    self.eig_solver.get_stored_energy(no_junctions)
                    self.eprd.do_EPR_analysis()
                    self.epra = epr.QuantumAnalysis(self.eprd.data_filename)
                    self.epra.analyze_all_variations(
                        cos_trunc=self.mini_study.cos_trunc,
                        fock_trunc=self.mini_study.fock_trunc,
                    )
                    self.epra.plot_hamiltonian_results()
                    freqs = self.epra.get_frequencies(numeric=True)
                    chis = self.epra.get_chis(numeric=True)
                except AttributeError:
                    log.error(
                        "Please install a more recent version of pyEPR (>=0.8.5.3)"
                    )

            self.eig_solver.setup.junctions = (
                self.mini_study.jj_setup
            )  # reset jj_setup for linear HFSS simulation

            self._update_optimized_params_epr(freqs, chis)
            return chis
        else:
            add_msg = ""
            if linear_element_found:
                add_msg = " However, a linear element was found."
            log.warning("No junctions found, skipping EPR analysis." + add_msg)
            return

    def get_simulated_modes_sorted(self):
        """Get simulated modes sorted on value.

        Returns:
            List[tuple]: list of modes sorted on freq_value
        """
        simulated_modes = self.mini_study.modes
        simulated_mode_freq_value = [
            (mode, self.system_target_params[param(mode, FREQ)])
            for mode in simulated_modes
        ]
        simulated_modes_sorted_on_freq_value = sorted(
            simulated_mode_freq_value, key=lambda x: x[1]
        )
        return simulated_modes_sorted_on_freq_value

    def _update_optimized_params(self, eig_result: pd.DataFrame):

        for idx, (mode, freq) in enumerate(self.get_simulated_modes_sorted()):
            freq = eig_result["Freq. (Hz)"][idx]
            decay = eig_result["Kappas (Hz)"][idx]

            self.system_optimized_params[param(mode, FREQ)] = freq
            if param(mode, KAPPA) in self.system_target_params:
                self.system_optimized_params[param(mode, KAPPA)] = decay

    def _get_mode_idx_map(self):
        """Get mode index map.

        Returns:
            dict: object {mode: idx}
        """
        all_modes = self.get_simulated_modes_sorted()
        mode_idx = {}
        for idx_i, (mode, _) in enumerate(all_modes):
            mode_idx[mode] = idx_i
        return mode_idx

    def _update_optimized_params_epr(
        self, freq_ND_results: pd.DataFrame, epr_result: pd.DataFrame
    ):

        MHz = 1e6
        mode_idx = self._get_mode_idx_map()
        log.info("freq_ND_results%s", dict_log_format(freq_ND_results.to_dict()))
        freq_column = 0

        for mode, _ in mode_idx.items():
            self.system_optimized_params[param(mode, FREQ)] = (
                freq_ND_results.iloc[mode_idx[mode]][freq_column] * MHz
            )

        for mode_i in self.mini_study.modes:
            for mode_j in self.mini_study.modes:
                self.system_optimized_params[param_nonlin(mode_i, mode_j)] = (
                    epr_result[mode_idx[mode_i]].iloc[mode_idx[mode_j]] * MHz
                )

    def _update_optimized_params_capacitance_simulation(
        self,
        capacitance_matrix: pd.DataFrame,
        capacitance_study: CapacitanceMatrixStudy,
    ):
        if CAPACITANCE_MATRIX_ELEMENTS in self.system_target_params:
            for key_capacitances in self.system_target_params[
                CAPACITANCE_MATRIX_ELEMENTS
            ].keys():
                try:
                    self.system_optimized_params[CAPACITANCE_MATRIX_ELEMENTS][
                        key_capacitances
                    ] = capacitance_matrix.loc[key_capacitances[0], key_capacitances[1]]
                except KeyError:
                    log.warning(
                        f"Warning: capacitance {key_capacitances} not found in capacitance matrix"
                    )
        if PURCELL_LIMIT_T1 in self.system_target_params:
            log.info("Computing T1 limit from decay in charge line.")
            self.system_optimized_params[capacitance_study.mode][
                capacitance_study.mode
            ] = capacitance_study.get_t1_limit_due_to_decay_into_charge_line()

    @staticmethod
    def _apply_adjustment_rate(new_val, old_val, rate):
        """Low pass filter for adjustment rate.

        Args:
            new_val (float): new value
            old_val (float): old value
            rate (float): rate of adjustment
        """
        return rate * new_val + (1 - rate) * old_val

    def _constrain_design_value(
        self,
        design_value_old: str,
        design_value_new: str,
        design_var_constraint: object,
    ):
        """Constrain design value.

        Args:
            design_value (str): design value to be constrained
            design_var_constraint (object): design variable constraint, example {'min': '10 um', 'max': '100 um'}
        """
        d_val_o, d_unit = get_value_and_unit(design_value_old)
        d_val_n, d_unit = get_value_and_unit(design_value_new)
        rate = self.mini_study.adjustment_rate
        d_val = self._apply_adjustment_rate(d_val_n, d_val_o, rate)

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
    def get_parameter_value(target: OptTarget, system_params: dict):
        if target.system_target_param == NONLIN:
            mode1, mode2 = target.involved_modes
            current_value = system_params[param_nonlin(mode1, mode2)]
        elif target.system_target_param == CAPACITANCE_MATRIX_ELEMENTS:
            capacitance_name_1, capacitance_name_2 = target.involved_modes
            current_value = system_params[
                param_capacitance(capacitance_name_1, capacitance_name_2)
            ]
        else:
            mode = target.involved_modes[0]
            current_value = system_params[param(mode, target.system_target_param)]
        return current_value

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
        ordered_design_var_names_to_minimize = [
            target.design_var for target in targets_to_minimize_for
        ]

        def cost_function(ordered_design_var_vals_updated):
            """Cost function to minimize.

            Args:
                ordered_design_var_vals_updated (List[float]): list of updated design variable values
            """
            for idx, name in enumerate(ordered_design_var_names_to_minimize):
                all_design_var_updated[name] = ordered_design_var_vals_updated[idx]
            cost = 0
            for target in targets_to_minimize_for:
                Q_k1_i = (
                    self.get_parameter_value(target, all_parameters_current)
                    * target.prop_to(all_parameters_targets_met, all_design_var_updated)
                    / target.prop_to(all_parameters_current, all_design_var_current)
                )
                cost += (
                    (
                        Q_k1_i
                        / self.get_parameter_value(target, all_parameters_targets_met)
                    )
                    - 1
                ) ** 2
            return cost

        init_design_var = []
        init_design_var = [
            all_design_var_current[name]
            for name in ordered_design_var_names_to_minimize
        ]

        scipy.optimize.minimize(
            cost_function, init_design_var, tol=self.minimization_tol
        )

    def get_system_params_targets_met(self):
        system_params_targets_met = deepcopy(self.system_optimized_params)
        for target in self.opt_targets:
            if target.system_target_param == NONLIN:
                mode1, mode2 = target.involved_modes
                system_params_targets_met[param_nonlin(mode1, mode2)] = (
                    self.get_parameter_value(target, self.system_target_params)
                )
            elif target.system_target_param == CAPACITANCE_MATRIX_ELEMENTS:
                capacitance_name_1, capacitance_name_2 = target.involved_modes
                system_params_targets_met[
                    param_capacitance(capacitance_name_1, capacitance_name_2)
                ] = self.get_parameter_value(target, self.system_target_params)
            else:
                mode_name = target.involved_modes[0]
                system_params_targets_met[
                    param(mode_name, target.system_target_param)
                ] = self.get_parameter_value(target, self.system_target_params)
        return system_params_targets_met

    def _calculate_target_design_var(self) -> dict:
        """Calculate the new design value for the optimization targets."""

        system_params_current = deepcopy(self.system_optimized_params)
        system_params_targets_met = self.get_system_params_targets_met()

        design_vars_current_str = deepcopy(self.design.variables)

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
        if independent_targets is not []:
            for independent_target in independent_targets:
                self._minimize_for_design_vars(
                    [independent_target],
                    design_vars_current,
                    design_vars_updated,
                    system_params_current,
                    system_params_targets_met,
                )

        dependent_targets = [
            target for target in self.opt_targets if not target.independent_target
        ]
        if len(dependent_targets) != 0:
            self._minimize_for_design_vars(
                dependent_targets,
                design_vars_current,
                design_vars_updated,
                system_params_current,
                system_params_targets_met,
            )

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

        # TODO AXEL document that the user must make sure that if they use e.g. sums or differences of design variables, they must make sure they are the same dimensions
        return design_vars_updated_constrained_str

    def optimize_target(
        self, updated_design_vars_input: dict, system_optimized_params: dict
    ):
        """Optimize with respect to provided targets.
        !!! Assumes that all simulated frequencies have same order as the target mode freqs, to associate them correctly. !!!
        The modes will be incorrectly assigned if the frequencies are not in the same order.
        Tip: simulate the modes individually first to get them close to their target frequencies.

        Args:
            updated_design_vars (dict): updated design variables
            system_optimized_params (dict): updated system optimized parameters
        """
        if not system_optimized_params == {}:
            self.is_system_optimized_params_initialized = True
        self.update_var(updated_design_vars_input, system_optimized_params)

        updated_design_vars = self._calculate_target_design_var()
        log.info("Updated_design_vars%s", dict_log_format(updated_design_vars))
        self.update_var(updated_design_vars, {})

        iteration_result = {}
        if self.mini_study is not None and len(self.mini_study.modes) > 0:
            # Eigenmode analysis for frequencies
            self.eig_result = self.run_eigenmodes()
            iteration_result["eig_results"] = deepcopy(self.eig_result)

            # EPR analysis for nonlinearities
            self.cross_kerrs = self.run_epr()

            iteration_result["cross_kerrs"] = deepcopy(self.cross_kerrs)

        if self.mini_study.capacitance_matrix_studies is not None:
            iteration_result["capacitance_matrix"] = []
            for capacitance_study in self.mini_study.capacitance_matrix_studies:
                capacitance_study.set_render_qiskit_metal(self.render_qiskit_metal)
                log.info("Simulating capacitance matrix study.")
                capacitance_matrix = capacitance_study.simulate_capacitance_matrix(
                    self.design
                )
                log.info("CAPACITANCE MATRIX\n%s", capacitance_matrix.to_string())
                self._update_optimized_params_capacitance_simulation(
                    capacitance_matrix, capacitance_study
                )

                iteration_result["capacitance_matrix"].append(
                    deepcopy(capacitance_matrix)
                )

        iteration_result["design_variables"] = deepcopy(self.design.variables)
        iteration_result["system_optimized_params"] = deepcopy(
            self.system_optimized_params
        )

        self.optimization_results.append(iteration_result)
        simulation = [
            {
                "optimization_results": self.optimization_results,
                "system_target_params": self.system_target_params,
                "plot_settings": self.plot_settings,
                "design_analysis_version": self.design_analysis_version,
            }
        ]
        if self.save_path is not None:
            np.save(self.save_path, simulation, allow_pickle=True)

            with open(self.save_path + "_design_variables.json", "w") as outfile:
                json.dump(updated_design_vars, outfile, indent=4)

        if self.update_design_variables is True:
            self.overwrite_parameters()

        if self.plot_settings is not None:
            plot_progress(
                self.optimization_results,
                self.system_target_params,
                self.plot_settings,
            )

    def overwrite_parameters(self):
        if self.save_path is None:
            raise Exception("A path must be specified to fetch results.")

        with open(self.save_path + "_design_variables.json") as in_file:
            updated_design_vars = json.load(in_file)

        with open("design_variables.json") as in_file:
            rewrite_parameters = json.load(in_file)

        for key, item in updated_design_vars.items():
            if key in rewrite_parameters:
                rewrite_parameters[key] = item

        with open("design_variables.json", "w") as outfile:
            json.dump(rewrite_parameters, outfile, indent=4)

        log.info("Overwritten parameters%s", dict_log_format(updated_design_vars))

    def get_cross_kerr_matrix(self, iteration: int = -1) -> pd.DataFrame:
        """Get cross kerr matrix from EPR analysis.

        Args:
            iteration (int): simulation iteration number defaults to the most recent iteration

        Returns:
            pd.DataFrame: cross kerr matrix
        """
        if "cross_kerrs" in self.optimization_results[iteration]:
            return self.optimization_results[iteration]["cross_kerrs"]

    def get_eigenmode_results(self, iteration: int = -1) -> pd.DataFrame:
        """Get eigenmode results.

        Args:
            iteration (int): simulation iteration number defaults to the most recent iteration

        Returns:
            pd.DataFrame: eigenmode results
        """
        if "eig_results" in self.optimization_results[iteration]:
            return self.optimization_results[iteration]["eig_results"]

    def get_capacitance_matrix(
        self, capacitance_study_number: int, iteration: int = -1
    ) -> Optional[pd.DataFrame]:
        """Get capacitance matrix from stati simulation.

        Args:
            capacitance_study_number (int): which of the capacitance studies in the MiniStudy to retrive the matrix from (1 being the first)
            iteration (int): simulation iteration number defaults to the most recent iteration

        Returns:
            List[pd.DataFrame]: capacitance matrix
        """
        if "capacitance_matrix" in self.optimization_results[iteration]:
            capacitance_matrices = self.optimization_results[iteration][
                "capacitance_matrix"
            ]

        if 0 <= capacitance_study_number - 1 < len(capacitance_matrices):
            return capacitance_matrices[capacitance_study_number - 1]

        return None

    def screenshot(self, gui, run=None):
        if self.save_path is None:
            raise Exception("A path must be specified to save screenshot.")
        gui.autoscale()
        name = self.save_path + f"_{run+1}" if run is not None else self.save_path
        gui.screenshot(name=name, display=False)
