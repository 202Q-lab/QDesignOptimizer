"""Core class for managing, optimizing and analyzing quantum circuit designs using electromagnetic simulations."""

import io
import json
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
from typing import List, Optional

import numpy as np
import pandas as pd
import pyEPR as epr
from pyaedt import Hfss
from pyEPR._config_default import config
from qiskit_metal.analyses.quantization import EPRanalysis

import qdesignoptimizer
from qdesignoptimizer.anmod_optimizer import ANModOptimizer
from qdesignoptimizer.design_analysis_types import (
    DesignAnalysisState,
    MeshingMap,
    MiniStudy,
    OptTarget,
)
from qdesignoptimizer.logger import dict_log_format, log
from qdesignoptimizer.sim_capacitance_matrix import (
    CapacitanceMatrixStudy,
    ModeDecayStudy,
    ResonatorDecayIntoWaveguideStudy,
)
from qdesignoptimizer.sim_plot_progress import plot_progress  # type: ignore
from qdesignoptimizer.utils.names_parameters import (
    CAPACITANCE,
    CHARGE_LINE_LIMITED_T1,
    FREQ,
    KAPPA,
    NONLIN,
    param,
    param_capacitance,
    param_nonlin,
    param_participation_ratio,
)


class DesignAnalysis:
    """Manager for quantum circuit design optimization and electromagnetic simulation.

    Handles the workflow of:

        - Simulation setup for eigenmode and capacitance calculations
        - Parameter extraction from simulations
        - Design variable optimization based on physical targets
        - Result tracking and visualization

    This class integrates HFSS/ANSYS simulations with Qiskit Metal designs to automate
    the optimization of superconducting circuit parameters.

    Args:
        state (DesignAnalysisState): State object containing design, render function, and target parameters.
        mini_study (MiniStudy): Simulation parameters including component names, modes, and passes.
        opt_targets (List[OptTarget]): List of target physical parameters to optimize and their design variable relationships.
        save_path (str): Location to save optimization results.
        update_design_variables (bool): Whether to automatically update design files with optimized values.
        plot_settings (dict): Configuration for progress visualization plots.
        meshing_map (List[MeshingMap]): Custom mesh refinement configurations for specific components.
        minimization_tol (float): tolerance used to terminate the solution of an optimization step.

    """

    def __init__(
        self,
        state: DesignAnalysisState,
        mini_study: MiniStudy,
        opt_targets: Optional[List[OptTarget]] = None,
        save_path: str = "analysis_result",
        update_design_variables: bool = True,
        plot_settings: Optional[dict] = None,
        meshing_map: Optional[List[MeshingMap]] = None,
        minimization_tol=1e-12,
    ):
        self.design_analysis_version = qdesignoptimizer.__version__
        self.design = state.design
        self.eig_solver = EPRanalysis(self.design, "hfss")
        self.eig_solver.sim.setup.name = "Resonator_setup"
        self.renderer = self.eig_solver.sim.renderer
        log.info(
            "self.eig_solver.sim.setup %s", dict_log_format(self.eig_solver.sim.setup)
        )
        self.eig_solver.setup.sweep_variable = "dummy"

        self.mini_study = mini_study
        self.opt_targets: List[OptTarget] = opt_targets or []
        self.all_design_vars = [target.design_var for target in self.opt_targets]
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
        self.meshing_map: List[MeshingMap] = meshing_map or []
        self.minimization_tol = minimization_tol

        self.anmod_optimizer = ANModOptimizer(
            opt_targets=self.opt_targets,
            system_target_params=self.system_target_params,
            adjustment_rate=self.mini_study.adjustment_rate,
            minimization_tol=self.minimization_tol,
        )

        self.optimization_results: list[dict] = []
        self.minimization_results: list[dict] = []

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
        self._validate_opt_targets()
        self.extracted_junctions_for_epr()

        assert (
            not self.system_target_params is self.system_optimized_params
        ), "system_target_params and system_optimized_params cannot be references to the same object"

    def update_nbr_passes(self, nbr_passes: int):
        """Update the number of simulation passes."""
        self.mini_study.nbr_passes = nbr_passes
        self.setup.passes = nbr_passes

    def update_nbr_passes_capacitance_ministudies(self, nbr_passes: int):
        """Updates the number of passes for capacitance matrix studies."""
        if self.mini_study.capacitance_matrix_studies:
            for cap_study in self.mini_study.capacitance_matrix_studies:
                cap_study.nbr_passes = nbr_passes

    def update_delta_f(self, delta_f: float):
        """Update eigenmode convergence tolerance."""
        self.mini_study.delta_f = delta_f
        self.setup.delta_f = delta_f

    def extracted_junctions_for_epr(self):
        """Extract junctions for EPR analysis."""
        self.no_junctions = (
            self.mini_study.jj_setup is None or len(self.mini_study.jj_setup) == 0
        )

        self.jj_setups_to_include_in_epr = {}
        for key, val in self.mini_study.jj_setup.items():
            # experimental implementation. to be generalized in the future to arbitrary junction potentials
            # this is a simple way to implement a linear potential only
            if "junction_type" in val and val["junction_type"] == "linear":
                continue
            self.jj_setups_to_include_in_epr[key] = val


    def _validate_opt_targets(self):
        """Validate opt_targets."""
        if self.opt_targets:
            for target in self.opt_targets:
                assert (
                    target.design_var in self.design.variables
                ), f"Design variable {target.design_var} not found in design variables."
                if target.target_param_type == CHARGE_LINE_LIMITED_T1:
                    assert (
                        len(self.mini_study.capacitance_matrix_studies) != 0
                    ), "capacitance_matrix_studies in ministudy must be populated for Charge line T1 decay study."
                elif target.target_param_type == KAPPA:
                    if not any(
                        isinstance(study, ResonatorDecayIntoWaveguideStudy)
                        for study in self.mini_study.capacitance_matrix_studies
                    ):
                        assert len(self.mini_study.modes) >= len(
                            target.involved_modes
                        ), f"Target for {target.target_param_type} expects \
                        {len(target.involved_modes)} modes but only {self.setup.n_modes} modes will be simulated."
                elif target.target_param_type == CAPACITANCE:
                    capacitance_1 = target.involved_modes[0]
                    capacitance_2 = target.involved_modes[1]
                    assert (
                        param_capacitance(*target.involved_modes)
                        in self.system_target_params
                    ), f"Target for '{CAPACITANCE}' requires {param_capacitance(*target.involved_modes)} in system_target_params."
                    assert (
                        len(target.involved_modes) == 2
                    ), f"Target for {target.target_param_type} expects 2 capacitance names, but {len(target.involved_modes)} were given."
                    assert isinstance(
                        capacitance_1, str
                    ), f"First capacitance name {capacitance_1} must be a string."
                    assert isinstance(
                        capacitance_2, str
                    ), f"Second capacitance name {capacitance_2} must be a string."

                elif target.target_param_type == NONLIN:
                    assert (
                        len(target.involved_modes) == 2
                    ), f"Target for {target.target_param_type} expects 2 modes."
                    assert len(self.mini_study.modes) >= len(
                        target.involved_modes
                    ), f"Target for {target.target_param_type} expects \
                        {len(target.involved_modes)} modes but only {self.setup.n_modes} modes will be simulated."
                    for mode in target.involved_modes:
                        assert (
                            mode in self.mini_study.modes
                        ), f"Target mode {mode} \
                            not found in modes which will be simulated."
                else:
                    assert len(self.mini_study.modes) >= len(
                        target.involved_modes
                    ), f"Target for {target.target_param_type} expects \
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
        """The fine mesh for the eigenmode study of HFSS can be set in two different ways. First, via an attribute in the component class with function name "get_meshing_names".
        This function should return the list of part names, e.g. {name}_flux_line_left. The second option is to specify the part names via the meshing_map as keyword in the DesginAnalysis class.
        """
        finer_mesh_names = []
        for component in self.mini_study.qiskit_component_names:
            if hasattr(self.design.components[component], "get_meshing_names"):
                finer_mesh_names += self.design.components[
                    component
                ].get_meshing_names()
            elif self.meshing_map is not None:
                for m_map in self.meshing_map:
                    if isinstance(
                        self.design.components[component], m_map.component_class
                    ):
                        finer_mesh_names += m_map.mesh_names(component)
            else:
                log.info("No fine mesh map was found for %s", component)

        return finer_mesh_names

    def get_port_gap_names(self):
        """Get names of all endcap ports in the design."""
        return [f"endcap_{comp}_{name}" for comp, name, _ in self.mini_study.port_list]

    def run_eigenmodes(self):
        """Simulate eigenmodes."""
        self.update_var({}, {})
        self.pinfo.validate_junction_info()

        self.hfss = Hfss()
        self.renderer.clean_active_design()
        self.render_qiskit_metal(
            self.design, **self.mini_study.render_qiskit_metal_eigenmode_kw_args
        )
        # set hfss wire bonds properties
        self.renderer.options["wb_size"] = self.mini_study.hfss_wire_bond_size
        self.renderer.options["wb_threshold"] = self.mini_study.hfss_wire_bond_threshold
        self.renderer.options["wb_offset"] = self.mini_study.hfss_wire_bond_offset

        # check for no hfss wire bonds in surface participation ratio analysis
        if self.mini_study.surface_properties:
            for component in self.mini_study.qiskit_component_names:
                if "hfss_wire_bonds" in self.design.components[component].options:
                    assert not self.design.components[component].options[
                        "hfss_wire_bonds"
                    ], f"hfss_wire_bonds in {component} must be set to False for surface participation ratio analysis."

        # render design in HFSS
        self.renderer.render_design(
            selection=self.mini_study.qiskit_component_names,
            port_list=self.mini_study.port_list,
            open_pins=self.mini_study.open_pins,
        )

        # set custom air bridges (only if no interfaces are defined in mini_study)
        if not self.mini_study.surface_properties:
            for component_name in self.mini_study.qiskit_component_names:
                if hasattr(
                    self.design.components[component_name], "get_air_bridge_coordinates"
                ):
                    for coord in self.design.components[
                        component_name
                    ].get_air_bridge_coordinates():
                        self.hfss.modeler.create_bondwire(
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

        # interfaces will be rendered if interfaces are defined in mini_study
        if self.mini_study.surface_properties:
            self._surface_rendering_for_surface_participation_ratios()

        # set fine mesh
        fine_mesh_names = self.get_fine_mesh_names()
        restrict_mesh = (
            (fine_mesh_names)
            and self.mini_study.build_fine_mesh
            and len(self.mini_study.port_list) > 0
        )

        if restrict_mesh:
            if self.mini_study.surface_properties:
                log.error("Interfaces must be empty when using fine mesh. ")
            else:
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
        
        if self.jj_setups_to_include_in_epr:
            self.eig_solver.setup.junctions = self.jj_setups_to_include_in_epr

            if not self.no_junctions:
                self.eprd = epr.DistributedAnalysis(self.pinfo)
                self.eig_solver.clear_data()
                self.eig_solver.get_stored_energy(self.no_junctions)
                self.eprd.do_EPR_analysis()
                self.epra = epr.QuantumAnalysis(self.eprd.data_filename)
                self.epra.analyze_all_variations(
                    cos_trunc=self.mini_study.cos_trunc,
                    fock_trunc=self.mini_study.fock_trunc,
                )
                self.epra.plot_hamiltonian_results()
                freqs = self.epra.get_frequencies(numeric=True)
                chis = self.epra.get_chis(numeric=True)
                participation_ratio = self.epra.get_participations() # normalized by default

                self._update_optimized_params_epr(freqs, chis, participation_ratio)

            self.eig_solver.setup.junctions = self.mini_study.jj_setup

            return chis, participation_ratio
        

    def _surface_rendering_for_surface_participation_ratios(self):
        """Render surfaces for surface participation ratio analysis.
        This function creates groups for different surfaces based on the interfaces defined in the mini_study.
        It handles the creation of groups for substrate-air, metal-substrate, metal-air, and underside surfaces.
        It also assigns materials to the metal-air group based on the sheet material defined in the mini_study.
        It also assigns interface properties using InterfaceProperties dataclass to pinfo for the Qanalysis.
        """

        metal = self.hfss.modeler.get_objects_in_group("Perfect E")
        self.hfss.modeler.ungroup(
            ["substrate_air", "metal_substrate", "underside_air", "metal_air"]
        )

        # Filter out JJ_rect_Lj objects once
        filtered_metal = [obj for obj in metal if not obj.startswith("JJ_rect_Lj")]

        # substrate-air
        self.hfss.modeler.section("main", "XY")
        cloned_polygon_names = []
        for obj in filtered_metal:
            self.hfss.modeler.clone(obj)
            cloned_polygon_names.append(obj + "1")
            self.hfss.modeler.subtract("main_Section1", cloned_polygon_names[-1], False)
        self.hfss.modeler.subtract("main_Section1", filtered_metal, True)
        self.hfss.modeler.create_group("main_Section1", group_name="substrate_air")

        # metal-substrate
        cloned_polygon_name = []
        for obj in filtered_metal:
            self.hfss.modeler.clone(obj)
            cloned_polygon_name.append(obj + "2")
        self.hfss.modeler.unite(cloned_polygon_name, False)
        self.hfss.modeler.create_group(
            cloned_polygon_name, group_name="metal_substrate"
        )

        # metal-air
        cloned_polygon_name = []
        for obj in filtered_metal:
            cloned_polygon_name.append(obj)
        metal_air = self.hfss.modeler.unite(cloned_polygon_name, False)
        metal_air = self.hfss.modeler.thicken_sheet(
            metal_air,
            self.mini_study.surface_properties.sheet_thickness,
            bBothSides=True,
        )
        metal_air = self.hfss.modeler.move(
            metal_air, [0, 0, self.mini_study.surface_properties.sheet_thickness / 2]
        )
        self.hfss.modeler.create_group(cloned_polygon_name, group_name="metal_air")
        objects = self.hfss.modeler.get_objects_in_group("metal_air")
        metal_air = self.hfss.assign_material(
            objects, self.mini_study.surface_properties.sheet_material
        )

        # underside surface
        self.hfss.modeler.section("main", "XY")
        self.hfss.modeler.move(
            "main_Section2", [0, 0, self.design._chips["main"]["size"]["size_z"]]
        )
        self.hfss.modeler.create_group(["main_Section2"], group_name="underside_air")

        # Assign interface properties using InterfaceProperties dataclass
        if self.mini_study.surface_properties.interfaces:
            self.pinfo.dissipative["dielectric_surfaces"] = {}
            for interface_name in self.mini_study.surface_properties.interfaces.keys():
                interface_props = getattr(
                    self.mini_study.surface_properties.interfaces, interface_name
                )
                for obj_name in self.hfss.modeler.get_objects_in_group(interface_name):
                    self.pinfo.dissipative["dielectric_surfaces"][obj_name] = asdict(
                        interface_props
                    )

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
        self, freq_ND_results: pd.DataFrame, epr_result: pd.DataFrame, participation_ratio: pd.DataFrame
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

        for mode in self.mini_study.modes:
            for jj_idx, junction in enumerate(self.jj_setups_to_include_in_epr):
                self.system_optimized_params[param_participation_ratio(mode, junction)] = (
                    participation_ratio.iloc[(mode_idx[mode], jj_idx)]
                )

    def _update_optimized_params_capacitance_simulation(
        self,
        capacitance_matrix: pd.DataFrame,
        capacitance_study: CapacitanceMatrixStudy,
    ):
        capacitance_names_all_targets = [
            target.involved_modes
            for target in self.opt_targets
            if target.target_param_type == CAPACITANCE
        ]

        for capacitance_names in capacitance_names_all_targets:
            try:
                self.system_optimized_params[param_capacitance(*capacitance_names)] = (
                    capacitance_matrix.loc[capacitance_names[0], capacitance_names[1]]
                )
            except KeyError:
                log.warning(
                    "Warning: capacitance %s not found in capacitance matrix with names %s",
                    capacitance_names,
                    capacitance_matrix.columns,
                )

        # Check if this is a ModeDecayStudy and update the appropriate parameter
        if isinstance(capacitance_study, ModeDecayStudy):
            param_type = capacitance_study.get_decay_parameter_type()
            log.info(f"Computing {param_type} from decay study.")
            if param_type == KAPPA:
                log.warning(
                    "Parameter KAPPA from capacitance matrix simulation will overwrite eigenmode result."
                )
            param_value = capacitance_study.get_decay_parameter_value()
            log.info(f"Computed {param_type} value: {param_value}")
            self.system_optimized_params[param(capacitance_study.mode, param_type)] = (
                param_value
            )

    def optimize_target(
        self, updated_design_vars_input: dict, system_optimized_params: dict, save_figures: bool = False
    ):
        """Run full optimization iteration to adjust design variables toward target parameters.

        Performs these steps in sequence:

        1. Updates design variables and system parameters
        2. Runs eigenmode simulation to extract frequencies and decay rates
        3. Performs energy participation ratio (EPR) analysis for nonlinearities
        4. Executes capacitance matrix studies if configured
        5. Saves results and generates visualization plots

        Args:
            updated_design_vars_input (dict): Manual design variable overrides
            system_optimized_params (dict): Manual system parameter overrides

        Note:
            Assumes modes are simulated in the same order as target frequencies for correct assignment.
            Mismatched orders will cause incorrect parameter mapping.
        """
        if not system_optimized_params == {}:
            self.is_system_optimized_params_initialized = True
        self.update_var(updated_design_vars_input, system_optimized_params)

        if not self.is_system_optimized_params_initialized:
            # bootstrap with initial design variables if no system_optimized_params exist
            updated_design_vars = deepcopy(self.design.variables)
            minimization_results: list[dict] = []
            self.is_system_optimized_params_initialized = True
        else:
            updated_design_vars, minimization_results = (
                self.anmod_optimizer.calculate_target_design_var(
                    self.system_optimized_params, self.design.variables
                )
            )
        log.info("Updated_design_vars%s", dict_log_format(updated_design_vars))
        self.update_var(updated_design_vars, {})

        iteration_result = {}
        if (
            self.mini_study is not None
            and len(self.mini_study.modes) > 0
            and not self.mini_study.run_capacitance_studies_only
        ):
            # Eigenmode analysis for frequencies
            self.eig_result = self.run_eigenmodes()
            iteration_result["eig_results"] = deepcopy(self.eig_result)

            # EPR analysis for nonlinearities
            self.cross_kerrs = self.run_epr()

            iteration_result["cross_kerrs"] = deepcopy(self.cross_kerrs)

            # Surface participation ratio
            if self.mini_study.surface_properties:
                self.surface_p_ratio = self.get_surface_p_ratio()
            else:
                self.surface_p_ratio = None

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

        iteration_result["design_variables"] = dict(deepcopy(self.design.variables))
        iteration_result["system_optimized_params"] = deepcopy(
            self.system_optimized_params
        )
        iteration_result["minimization_results"] = minimization_results

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
            import os
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            np.save(self.save_path, np.array(simulation), allow_pickle=True)

            with open(self.save_path + "_design_variables.json", "w") as outfile:
                json.dump(updated_design_vars, outfile, indent=4)

        if self.update_design_variables is True:
            self.overwrite_parameters()

        if self.plot_settings is not None:
            plot_progress(
                [self.optimization_results],
                self.system_target_params,
                self.plot_settings,
                save_figures=save_figures,
                save_path=self.save_path,
            )

    def overwrite_parameters(self):
        """Overwirte the original design_variables.json file with new values."""
        if self.save_path is None:
            raise ValueError("A path must be specified to fetch results.")

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

    def get_cross_kerr_matrix(self, iteration: int = -1) -> Optional[pd.DataFrame]:
        """Get cross kerr matrix from EPR analysis.

        Args:
            iteration (int): simulation iteration number defaults to the most recent iteration

        Returns:
            pd.DataFrame: cross kerr matrix
        """
        if "cross_kerrs" in self.optimization_results[iteration]:
            return self.optimization_results[iteration]["cross_kerrs"]
        log.warning("No cross kerr matrix in optimization results.")
        return None

    def get_eigenmode_results(self, iteration: int = -1) -> Optional[pd.DataFrame]:
        """Get eigenmode results.

        Args:
            iteration (int): simulation iteration number defaults to the most recent iteration

        Returns:
            pd.DataFrame: eigenmode results
        """
        if "eig_results" in self.optimization_results[iteration]:
            return self.optimization_results[iteration]["eig_results"]
        log.warning("No eigenmode matrix in optimization results.")
        return None

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
        """Take and save a screenshot of the current qiskit-metal design.

        Useful for tracking how the geometry is updated during optimization.
        """
        if self.save_path is None:
            raise ValueError("A path must be specified to save screenshot.")
        gui.autoscale()
        name = self.save_path + f"_{run+1}" if run is not None else self.save_path
        gui.screenshot(name=name, display=False)

    def _get_dielectric_p_ratio(self):
        """Get dielectric participation ratio taking into account the dielectric loss tangent."""

        p_dielectric = {}
        for mode in range(int(self.pinfo.setup.n_modes)):
            self.eprd.set_mode(mode)
            with io.StringIO() as buf, redirect_stdout(buf):
                q_dielectric = self.eprd.get_Qdielectric(
                    dielectric="main", mode=mode, variation=None
                )[0]
            p_dielectric[mode] = 1 / (q_dielectric * config.dissipation.tan_delta_sapp)

        return p_dielectric

    def _get_surface_p_ratio(self, name, mode, variation=None):
        """Get surface participation ratio."""
        with io.StringIO() as buf, redirect_stdout(buf):
            psurf = 1 / self.eprd.get_Qsurface(
                mode=mode, variation=variation, name=name
            )  # we set tand = 1 in the design file
        return psurf.iloc[
            0
        ]  # Return the second element which is the participation ratio value

    def get_surface_p_ratio(self):
        """Computes the surfaces participation ratio for all given interfaces. And also for every junction."""
        p_ratio_dict = {}

        # Initialize the structure for interfaces
        for interface_name in self.mini_study.surface_properties.interfaces.keys():
            p_ratio_dict[interface_name] = {}
            for mode in range(int(self.pinfo.setup.n_modes)):
                self.eprd.set_mode(mode)
                p_ratio_dict[interface_name][mode] = self._get_surface_p_ratio(
                    name=self.hfss.modeler.get_objects_in_group(interface_name)[0],
                    mode=mode,
                )

        # Handle junction data
        # can only handle a single junction type for now
        p_ratio_dict["Junction(inductive energy)"] = {}
        for mode in range(int(self.pinfo.setup.n_modes)):
            self.eprd.set_mode(mode)
            j_ratio = self.eprd.calc_p_junction_single(mode=mode, variation=None)
            for key in j_ratio:
                p_ratio = j_ratio[key]
            p_ratio_dict["Junction(inductive energy)"][mode] = p_ratio

        # Handle dielectric data
        p_ratio_dict["dielectric"] = self._get_dielectric_p_ratio()

        return p_ratio_dict
