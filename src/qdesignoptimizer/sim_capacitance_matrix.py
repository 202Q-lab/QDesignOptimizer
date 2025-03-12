"""Study classes for capacitance matrix based simulations."""

from abc import ABC, abstractmethod
from typing import Callable, List, Literal, Optional, Union

import numpy as np
from qiskit_metal.analyses.quantization import LOManalysis
from qiskit_metal.designs.design_base import QDesign

from qdesignoptimizer.estimation.classical_model_decay_into_charge_line import (
    calculate_t1_limit_floating_lumped_mode_decay_into_chargeline,
    calculate_t1_limit_grounded_lumped_mode_decay_into_chargeline,
)
from qdesignoptimizer.utils.names_parameters import CHARGE_LINE_LIMITED_T1, KAPPA


class CapacitanceMatrixStudy:
    """
    Capacitance matrix study for DesignAnalysis.

    When several components are rendered in the capacitance matrix analysis,
    the union of all connected components will be represented by a unique name,
    which is a column in the capacitance matrix returned from LOManalysis e.g.
    by running analyse_capacitance_matrix. We refer to this unique name as the capacitance name.
    Note that the capacitance name typically change if you change which components are included in the analysis.

    Args:
        qiskit_component_names (list):         list of Qiskit component names to be included in the capacitance simulation
        freq_GHz (float):               Sets the frequency in GHz of the capacitance matrix. Or (not supported yet): If tuple, the simulation will use the frequency from corresponding mode in the EPR analysis.n
                                        Example1: 5e9, Example2: ('BRANCH_1', 'qubit_freq')
        open_pins (list):               pins to be left open (called open_terminations in the capacitance matrix simulation),
                                        Example: [('comp_name', 'pin_name')]
        x_buffer_width_mm (float):      x buffer width added around the rendered components
        y_buffer_width_mm (float):      y buffer width added around the rendered components

        render_qiskit_metal (Callable): if provided, the design will be rerendered using this function before the capacitance simulation
                                        If a CapacitanceMatrixStudy is used in the DesignAnalysis optimization and self.render_qiskit_metal==None,
                                        render_qiskit_metal from DesignAnalysisState will be used instead (i.e. it doesn't need to be provided).
                                        Format: render_qiskit_metal(design, `**kw_args`)
        render_qiskit_metal_kwargs (dict): kwargs for render_qiskit_metal

        percent_error (float):          percent error in capacitance simulation
        nbr_passes (int):               group of passes in capacitance simulation
    """

    def __init__(
        self,
        qiskit_component_names: list,
        mode_freq_GHz: Union[float],
        open_pins: Optional[list] = None,
        x_buffer_width_mm: float = 2,
        y_buffer_width_mm: float = 2,
        render_qiskit_metal: Optional[Callable] = None,
        render_qiskit_metal_kwargs: Optional[dict] = None,
        percent_error: Optional[float] = 0.5,
        nbr_passes: Optional[int] = 10,
    ):
        self.qiskit_component_names = qiskit_component_names
        self.mode_freq_GHz = mode_freq_GHz
        self.open_pins: list = open_pins or []
        self.x_buffer_width_mm = x_buffer_width_mm
        self.y_buffer_width_mm = y_buffer_width_mm

        self.render_qiskit_metal = render_qiskit_metal
        self.render_qiskit_metal_kwargs: dict = render_qiskit_metal_kwargs or {}

        self.percent_error = percent_error
        self.nbr_passes = nbr_passes

        self.capacitance_matrix_fF = (
            None  # (fF) gets populated by simulate_capacitance_matrix (pd.DataFrame)
        )
        self.mode_capacitances_matrix_fF = (
            None  # (fF) gets populated by simulate_capacitance_matrix (dict)
        )
        # Example: {('branch_name', 'freq_name'): 100}

    def set_render_qiskit_metal(self, render_qiskit_metal: Callable):
        """
        Set the render_qiskit_metal function to be used before the capacitance simulation.

        Args:
            render_qiskit_metal (Callable): The render_qiskit_metal function. Format: render_qiskit_metal(design, `**kw_args`)
        """
        self.render_qiskit_metal = render_qiskit_metal

    def simulate_capacitance_matrix(
        self,
        design: QDesign,
    ):
        """Simualte the capacitance matrix for the given design and study parameters
        and populates the capacitance_matrix_fF attribute.

        Args:
            design (QDesign): The design to simulate.

        Returns:
            pandas.DataFrame: The capacitance matrix.
        """
        if self.render_qiskit_metal is not None:
            self.render_qiskit_metal(design, **self.render_qiskit_metal_kwargs)

        lom_analysis = LOManalysis(design, "q3d")
        lom_analysis.sim.setup.max_passes = self.nbr_passes
        # lom_analysis.sim.setup.freq_ghz = self.freq_GHz
        lom_analysis.sim.setup.percent_error = self.percent_error
        lom_analysis.sim.renderer.options["x_buffer_width_mm"] = self.x_buffer_width_mm
        lom_analysis.sim.renderer.options["y_buffer_width_mm"] = self.y_buffer_width_mm

        lom_analysis.sim.run(
            components=self.qiskit_component_names, open_terminations=self.open_pins
        )
        self.capacitance_matrix_fF = lom_analysis.sim.capacitance_matrix
        return self.capacitance_matrix_fF


class ModeDecayStudy(ABC, CapacitanceMatrixStudy):
    """Base class for mode decay studies using capacitance matrix simulation.

    Since the capacitance should be evaluated at the frequency of the mode,
    each decay analysis should be done in a separate ModeDecayStudy.

    Args:
        mode (str): The mode name
        mode_freq_GHz (float): The mode frequency in GHz
    """

    _decay_parameter_type = None  # To be defined by subclasses

    def __init__(
        self,
        mode: str,
        mode_freq_GHz: float,
        qiskit_component_names: list,
        open_pins: Optional[list] = None,
        x_buffer_width_mm: float = 2,
        y_buffer_width_mm: float = 2,
        percent_error: float = 0.5,
        nbr_passes: int = 10,
    ):
        super().__init__(
            qiskit_component_names=qiskit_component_names,
            mode_freq_GHz=mode_freq_GHz,
            open_pins=open_pins,
            x_buffer_width_mm=x_buffer_width_mm,
            y_buffer_width_mm=y_buffer_width_mm,
            percent_error=percent_error,
            nbr_passes=nbr_passes,
        )
        self.mode = mode
        self.freq_GHz = mode_freq_GHz

    @abstractmethod
    def get_decay_parameter_value(self):
        """Get the simulated parameter value.
        This method should be implemented by subclasses.

        Returns:
            float: The simulated parameter value.
        """
        pass

    def get_decay_parameter_type(self):
        """Get the type of parameter this study optimizes.

        Returns:
            str: The parameter type.
        """
        return self._decay_parameter_type


class ModeDecayIntoChargeLineStudy(ModeDecayStudy):
    """Mode decay into charge line study by capacitance matrix simulation.

    Since the capacitance should be evaluated at the frequency of the mode,
    each decay analysis should be done in a separate ModeDecayIntoChargeLineStudy.

    Args:
        mode (str): The mode name
        mode_freq_GHz (float): The mode frequency in GHz
        mode_capacitance_name (str or List[str]): capacitance name of mode, if grounded: 1 string of island name,
                                                if floating: list of 2 strings of respective island name
        charge_line_capacitance_name (str): capacitance name of charge line
        charge_line_impedance_Ohm (float): charge line impedance in Ohm
        ground_plane_capacitance_name (str, optional): capacitance name of ground plane
    """

    _decay_parameter_type = CHARGE_LINE_LIMITED_T1  # type: ignore

    def __init__(
        self,
        mode: str,
        mode_freq_GHz: float,
        mode_capacitance_name: Union[str, List[str]],
        charge_line_capacitance_name: str,
        charge_line_impedance_Ohm: float,
        qiskit_component_names: list,
        open_pins: Optional[list] = None,
        ground_plane_capacitance_name: Optional[str] = None,
        x_buffer_width_mm: float = 2,
        y_buffer_width_mm: float = 2,
        percent_error: float = 0.5,
        nbr_passes: int = 10,
    ):
        super().__init__(
            mode=mode,
            mode_freq_GHz=mode_freq_GHz,
            qiskit_component_names=qiskit_component_names,
            open_pins=open_pins,
            x_buffer_width_mm=x_buffer_width_mm,
            y_buffer_width_mm=y_buffer_width_mm,
            percent_error=percent_error,
            nbr_passes=nbr_passes,
        )
        self.mode_capacitance_name = mode_capacitance_name
        self.ground_plane_capacitance_name = ground_plane_capacitance_name
        self.charge_line_capacitance_name = charge_line_capacitance_name
        self.charge_line_impedance_Ohm = charge_line_impedance_Ohm
        self.t1_limit_due_to_decay_into_charge_line = None

    def get_t1_limit_due_to_decay_into_charge_line(self) -> float:
        """Get the T1 limit due to decay into charge line decay
        and populates the t1_limit_due_to_decay_into_charge_line attribute.

        Returns:
            float: The T1 limit due to decay into charge line decay (s).

        Raises:
            AssertionError: If capacitance_matrix_fF is not set when trying to access T1 limit.
        """
        assert (
            self.capacitance_matrix_fF is not None
        ), "capacitance_matrix_fF is not set, you need to run .simulate_capacitance_matrix()."

        # get the charge line name
        charge_line_name = self.charge_line_capacitance_name

        # get ground name
        name_ground = self.ground_plane_capacitance_name

        # get name of islands depending on floating or grounded design
        if isinstance(
            self.mode_capacitance_name, str
        ):  # single island, grounded design
            name_island = self.mode_capacitance_name

            Csum = np.abs(self.capacitance_matrix_fF.loc[name_island, name_island])
            Ccoupling = np.abs(
                self.capacitance_matrix_fF.loc[name_island, charge_line_name]
            )

            self.t1_limit_due_to_decay_into_charge_line = (
                calculate_t1_limit_grounded_lumped_mode_decay_into_chargeline(
                    mode_capacitance_fF=Csum,
                    mode_capacitance_to_charge_line_fF=Ccoupling,
                    mode_freq_GHz=self.freq_GHz,
                    charge_line_impedance=self.charge_line_impedance_Ohm,
                )
            )

        elif (
            isinstance(self.mode_capacitance_name, list)
            and len(self.mode_capacitance_name) == 2
            and all(isinstance(item, str) for item in self.mode_capacitance_name)
        ):  # split transmon, floating
            name_island_A = self.mode_capacitance_name[0]
            name_island_B = self.mode_capacitance_name[1]

            # deconstruct capacitance matrix to compute Ca, Cb, Csum for a split transmon
            Ca0 = np.abs(self.capacitance_matrix_fF.loc[name_island_A, name_ground])
            Cb0 = np.abs(self.capacitance_matrix_fF.loc[name_island_B, name_ground])
            Ca1 = np.abs(
                self.capacitance_matrix_fF.loc[name_island_A, charge_line_name]
            )
            Cb1 = np.abs(
                self.capacitance_matrix_fF.loc[name_island_B, charge_line_name]
            )
            CJ = np.abs(self.capacitance_matrix_fF.loc[name_island_A, name_island_B])

            self.t1_limit_due_to_decay_into_charge_line = (
                calculate_t1_limit_floating_lumped_mode_decay_into_chargeline(
                    mode_freq_GHz=self.freq_GHz,
                    cap_island_a_island_b_fF=CJ,
                    cap_island_a_ground_fF=Ca0,
                    cap_island_a_line_fF=Ca1,
                    cap_island_b_ground_fF=Cb0,
                    cap_island_b_line_fF=Cb1,
                    charge_line_impedance=self.charge_line_impedance_Ohm,
                )
            )

        else:
            raise NotImplementedError(
                "The mode capacitance name must be a string or a list of string matching the name of the island(s)."
            )

        return self.t1_limit_due_to_decay_into_charge_line

    def get_decay_parameter_value(self) -> float:
        """Get the optimized parameter value (T1 limit).

        Returns:
            float: The T1 limit due to decay into charge line.
        """
        return self.get_t1_limit_due_to_decay_into_charge_line()


class ResonatorDecayIntoWaveguideStudy(ModeDecayStudy):
    """Resonator decay into waveguide study by capacitance matrix simulation.

    Since the capacitance should be evaluated at the frequency of the mode,
    each decay analysis should be done in a separate ResonatorDecayIntoWaveguideStudy.

    Args:
        mode (str): The mode name
        mode_freq_GHz (float): The mode frequency in GHz
        resonator_name (str): capacitance name of resonator
        waveguide_name (str): capacitance name of waveguide
        waveguide_impedance_Ohm (float): waveguide impedance in Ohm
        resonator_type (Literal["lambda_4", "lambda_2"]): specifies the type of resonator
    """

    _decay_parameter_type = KAPPA  # type: ignore

    def __init__(
        self,
        mode: str,
        mode_freq_GHz: float,
        resonator_name: str,
        waveguide_name: str,
        waveguide_impedance_Ohm: float,
        qiskit_component_names: list,
        resonator_type: Literal["lambda_4", "lambda_2"],
        open_pins: Optional[list] = None,
        x_buffer_width_mm: float = 2,
        y_buffer_width_mm: float = 2,
        percent_error: float = 0.5,
        nbr_passes: int = 10,
    ):
        super().__init__(
            mode=mode,
            mode_freq_GHz=mode_freq_GHz,
            qiskit_component_names=qiskit_component_names,
            open_pins=open_pins,
            x_buffer_width_mm=x_buffer_width_mm,
            y_buffer_width_mm=y_buffer_width_mm,
            percent_error=percent_error,
            nbr_passes=nbr_passes,
        )
        self.resonator_name = resonator_name
        self.waveguide_name = waveguide_name
        self.waveguide_impedance_Ohm = waveguide_impedance_Ohm
        self.resonator_type = resonator_type
        self.kappa = None

    def get_kappa_estimate(self) -> float:
        """Get the estimated kappa (decay rate) of the resonator into the waveguide.

        Currently uses a simplified model, assuming that the resonator and waveguide have the same impedance.

        Returns:
            float: The estimated kappa (Hz).

        Raises:
            AssertionError: If capacitance_matrix_fF is not set when trying to access kappa.
        """
        assert (
            self.capacitance_matrix_fF is not None
        ), "capacitance_matrix_fF is not set, you need to run .simulate_capacitance_matrix()."

        omega = self.freq_GHz * 2 * np.pi

        Ccoupling = np.abs(
            self.capacitance_matrix_fF.loc[self.resonator_name, self.waveguide_name]
        )

        Z0 = self.waveguide_impedance_Ohm
        unit_conversion = 1e-3  # GHz^3 * fF^2
        kappa = Z0**2 * omega**3 * Ccoupling**2 / np.pi / (2 * np.pi) * unit_conversion

        if self.resonator_type == "lambda_4":
            kappa *= 2

        self.kappa = kappa
        return kappa

    def get_decay_parameter_value(self) -> float:
        """Get the optimized parameter value (kappa).

        Returns:
            float: The estimated kappa.
        """
        return self.get_kappa_estimate()
