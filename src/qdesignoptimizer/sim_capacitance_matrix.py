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
    """Base class for capacitance matrix simulations of quantum circuit components in DesignAnalysis.

    When multiple components are rendered in the capacitance matrix analysis, the union of all
    connected components will be represented by a unique name, which becomes a column in the
    resulting capacitance matrix. These capacitance names typically change if you modify which
    components are included in the analysis.

    This class can be used directly for basic capacitance simulations or extended by specialized
    subclasses to calculate specific decay parameters like T1 or kappa.

    Args:
        qiskit_component_names (list): Names of Qiskit Metal components to include in the
            capacitance simulation.
        open_pins (list, optional): Pin connections to leave open (not grounded or connected),
            specified as tuples of (component_name, pin_name). Defaults to an empty list.
        x_buffer_width_mm (float, optional): Width of simulation buffer space in x-direction
            in millimeters. Defaults to 2mm.
        y_buffer_width_mm (float, optional): Width of simulation buffer space in y-direction
            in millimeters. Defaults to 2mm.
        render_qiskit_metal (Callable, optional): Function for rendering the design before
            simulation. If None, the function from DesignAnalysisState will be used when this
            study is part of a DesignAnalysis optimization. Takes the form
            ``render_qiskit_metal(design, **kwargs)``.
        render_qiskit_metal_kwargs (dict, optional): Keyword arguments for the render_qiskit_metal
            function. Defaults to an empty dict.
        percent_error (float, optional): Target percentage error for simulation convergence.
            Defaults to 0.5%.
        nbr_passes (int, optional): Maximum number of mesh refinement passes to perform.
            Defaults to 10.
    """

    def __init__(
        self,
        qiskit_component_names: list,
        open_pins: Optional[list] = None,
        x_buffer_width_mm: float = 2,
        y_buffer_width_mm: float = 2,
        render_qiskit_metal: Optional[Callable] = None,
        render_qiskit_metal_kwargs: Optional[dict] = None,
        percent_error: Optional[float] = 0.5,
        nbr_passes: Optional[int] = 10,
    ):
        self.qiskit_component_names = qiskit_component_names
        self.open_pins: list = open_pins or []
        self.x_buffer_width_mm = x_buffer_width_mm
        self.y_buffer_width_mm = y_buffer_width_mm

        self.render_qiskit_metal = render_qiskit_metal
        self.render_qiskit_metal_kwargs: dict = render_qiskit_metal_kwargs or {}

        self.percent_error = percent_error
        self.nbr_passes = nbr_passes

        self.capacitance_matrix_fF = None
        """pandas.DataFrame: Capacitance matrix results from simulation in femtofarads (fF).
        Populated by calling simulate_capacitance_matrix().
        """

        self.mode_capacitances_matrix_fF = None
        """dict: Extracted mode capacitances from the capacitance matrix.
        Populated by calling simulate_capacitance_matrix().
        """

    def set_render_qiskit_metal(self, render_qiskit_metal: Callable):
        """Set the rendering function to use before capacitance simulation.

        This method allows updating the rendering function after initialization,
        particularly useful when a CapacitanceMatrixStudy is being used within
        a DesignAnalysis context.

        Args:
            render_qiskit_metal (Callable): Function that renders the design components,
                with signature ``render_qiskit_metal(design, **kwargs)``.

        """
        self.render_qiskit_metal = render_qiskit_metal

    def simulate_capacitance_matrix(
        self,
        design: QDesign,
    ):
        """Run the capacitance matrix simulation for the specified design.

        Args:
            design (QDesign): The Qiskit Metal design to simulate.

        Returns:
            pandas.DataFrame: The simulated capacitance matrix in femtofarads (fF).
            Rows and columns correspond to component/island names.

        Note:
            The simulation results are also stored in the capacitance_matrix_fF
            attribute for later use by derived classes.
        """
        if self.render_qiskit_metal is not None:
            self.render_qiskit_metal(design, **self.render_qiskit_metal_kwargs)

        lom_analysis = LOManalysis(design, "q3d")
        lom_analysis.sim.setup.max_passes = self.nbr_passes
        lom_analysis.sim.setup.percent_error = self.percent_error
        lom_analysis.sim.renderer.options["x_buffer_width_mm"] = self.x_buffer_width_mm
        lom_analysis.sim.renderer.options["y_buffer_width_mm"] = self.y_buffer_width_mm

        lom_analysis.sim.run(
            components=self.qiskit_component_names, open_terminations=self.open_pins
        )
        self.capacitance_matrix_fF = lom_analysis.sim.capacitance_matrix
        return self.capacitance_matrix_fF


class ModeDecayStudy(ABC, CapacitanceMatrixStudy):
    """Abstract base class for studies that analyze mode decay using capacitance simulations.

    This class extends CapacitanceMatrixStudy to provide a common interface for
    different types of decay studies. Specific decay mechanisms (such as decay into
    charge lines or waveguides) are implemented by subclasses.

    Since capacitance values should be evaluated at the frequency of the specific mode,
    each decay analysis should be performed in a separate ModeDecayStudy instance.

    Args:
        mode (str): Name of the mode being analyzed (e.g., "qubit_1").
        mode_freq_GHz (float): Frequency of the mode in GHz.
        qiskit_component_names (list): Components to include in simulation.
        open_pins (list, optional): Pins to leave open. Defaults to [].
        x_buffer_width_mm (float, optional): X buffer width in mm. Defaults to 2.
        y_buffer_width_mm (float, optional): Y buffer width in mm. Defaults to 2.
        render_qiskit_metal (Callable, optional): Function for rendering the design before
            simulation. If None, the function from DesignAnalysisState will be used when this
            study is part of a DesignAnalysis optimization. Takes the form
            ``render_qiskit_metal(design, **kwargs)``.
        render_qiskit_metal_kwargs (dict, optional): Keyword arguments for the render_qiskit_metal
            function. Defaults to an empty dict.
        percent_error (float, optional): Target simulation error. Defaults to 0.5.
        nbr_passes (int, optional): Maximum simulation passes. Defaults to 10.

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
        render_qiskit_metal: Optional[Callable] = None,
        render_qiskit_metal_kwargs: Optional[dict] = None,
        percent_error: float = 0.5,
        nbr_passes: int = 10,
    ):
        super().__init__(
            qiskit_component_names=qiskit_component_names,
            open_pins=open_pins,
            x_buffer_width_mm=x_buffer_width_mm,
            y_buffer_width_mm=y_buffer_width_mm,
            render_qiskit_metal=render_qiskit_metal,
            render_qiskit_metal_kwargs=render_qiskit_metal_kwargs,
            percent_error=percent_error,
            nbr_passes=nbr_passes,
        )
        self.mode = mode
        self.mode_freq_GHz = mode_freq_GHz

    @abstractmethod
    def get_decay_parameter_value(self):
        """Calculate and return the decay parameter value for this mode.

        This abstract method must be implemented by subclasses to compute
        specific decay parameters (T1, kappa, etc.) from the capacitance matrix.

        Returns:
            float: The calculated decay parameter value in appropriate units.
        """
        pass

    def get_decay_parameter_type(self):
        """Get the type of decay parameter this study calculates.

        Returns:
            str: Parameter type identifier (e.g., "charge_line_limited_t1", "kappa").
        """
        return self._decay_parameter_type


class ModeDecayIntoChargeLineStudy(ModeDecayStudy):
    """Study for calculating T1 relaxation time due to decay into a charge line.

    This class calculates the T1 relaxation time limit for a mode (typically a qubit)
    due to its coupling to a charge line. The calculation accounts for the capacitance
    between the mode's islands and the charge line, using either a grounded or floating
    island model depending on the physical configuration.

    Args:
        mode (str): Name of the mode being analyzed (e.g., "qubit_1").
        mode_freq_GHz (float): Frequency of the mode in GHz.
        mode_capacitance_name (str or List[str]): Capacitance name(s) of the mode's islands:

            - For a grounded design: provide a single string with the island name
            - For a floating design: provide a list of two strings with the island names

        charge_line_capacitance_name (str): Capacitance name of the charge line.
        charge_line_impedance_Ohm (float): Impedance of the charge line in Ohms.
        qiskit_component_names (list): Components to include in simulation.
        open_pins (list, optional): Pins to leave open. Defaults to [].
        ground_plane_capacitance_name (str, optional): Capacitance name of the ground plane.
            Required for floating island calculations.
        x_buffer_width_mm (float, optional): X buffer width in mm. Defaults to 2.
        y_buffer_width_mm (float, optional): Y buffer width in mm. Defaults to 2.
        render_qiskit_metal (Callable, optional): Function for rendering the design before
            simulation. If None, the function from DesignAnalysisState will be used when this
            study is part of a DesignAnalysis optimization. Takes the form
            ``render_qiskit_metal(design, **kwargs)``.
        render_qiskit_metal_kwargs (dict, optional): Keyword arguments for the render_qiskit_metal
            function. Defaults to an empty dict.
        percent_error (float, optional): Target simulation error. Defaults to 0.5.
        nbr_passes (int, optional): Maximum simulation passes. Defaults to 10.

    Note:
        The calculation method differs between grounded and floating designs:

        - Grounded design: Single island capacitively coupled to charge line
        - Floating design: Two islands (e.g., split transmon) coupled to charge line

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
        render_qiskit_metal: Optional[Callable] = None,
        render_qiskit_metal_kwargs: Optional[dict] = None,
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
            render_qiskit_metal=render_qiskit_metal,
            render_qiskit_metal_kwargs=render_qiskit_metal_kwargs,
            percent_error=percent_error,
            nbr_passes=nbr_passes,
        )
        self.mode_capacitance_name = mode_capacitance_name
        self.ground_plane_capacitance_name = ground_plane_capacitance_name
        self.charge_line_capacitance_name = charge_line_capacitance_name
        self.charge_line_impedance_Ohm = charge_line_impedance_Ohm
        self.t1_limit_due_to_decay_into_charge_line = None

    def get_t1_limit_due_to_decay_into_charge_line(self) -> float:
        """Calculate the T1 relaxation time limit due to charge line coupling..

        Returns:
            float: The T1 limit in seconds.

        Raises:
            AssertionError: If capacitance_matrix_fF has not been set (run simulate_capacitance_matrix first).
            NotImplementedError: If mode_capacitance_name has an invalid format.

        Note:
            For grounded designs (single island), the function calculates T1 using:

            - The total island self-capacitance
            - The coupling capacitance to the charge line

            For floating designs (two islands), the function uses:

            - The coupling capacitance between islands
            - The capacitances between each island and ground
            - The capacitances between each island and the charge line

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
                    mode_freq_GHz=self.mode_freq_GHz,
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
                    mode_freq_GHz=self.mode_freq_GHz,
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
        """Get the T1 limit due to charge line coupling.

        Returns:
            float: The T1 limit in seconds.
        """
        return self.get_t1_limit_due_to_decay_into_charge_line()


class ResonatorDecayIntoWaveguideStudy(ModeDecayStudy):
    """Study for calculating resonator decay rate (kappa) into a waveguide.

    This class calculates the decay rate (kappa) of a resonator mode into a
    coupled waveguide based on the capacitance between them. The calculation
    uses a simplified model assuming the resonator and waveguide have the same
    impedance.

    Args:
        mode (str): Name of the resonator mode being analyzed.
        mode_freq_GHz (float): Frequency of the resonator mode in GHz.
        resonator_name (str): Capacitance name of the resonator in the simulation.
        waveguide_name (str): Capacitance name of the waveguide in the simulation.
        impedance_ohm (float): Impedance of both the waveguide and resonator in Ohms.
        qiskit_component_names (list): Components to include in simulation.
        resonator_type (Literal["lambda_4", "lambda_2"]): Type of resonator:

            - "lambda_4": Quarter-wavelength resonator
            - "lambda_2": Half-wavelength resonator

        open_pins (list, optional): Pins to leave open. Defaults to [].
        x_buffer_width_mm (float, optional): X buffer width in mm. Defaults to 2.
        y_buffer_width_mm (float, optional): Y buffer width in mm. Defaults to 2.
        render_qiskit_metal (Callable, optional): Function for rendering the design before
            simulation. If None, the function from DesignAnalysisState will be used when this
            study is part of a DesignAnalysis optimization. Takes the form
            ``render_qiskit_metal(design, **kwargs)``.
        render_qiskit_metal_kwargs (dict, optional): Keyword arguments for the render_qiskit_metal
            function. Defaults to an empty dict.
        percent_error (float, optional): Target simulation error. Defaults to 0.5.
        nbr_passes (int, optional): Maximum simulation passes. Defaults to 10.

    Note:
        The calculation differs between quarter-wavelength and half-wavelength resonators:

        - Quarter-wavelength (lambda/4): kappa value is doubled compared to half-wavelength
        - Half-wavelength (lambda/2): standard calculation

    """

    _decay_parameter_type = KAPPA  # type: ignore

    def __init__(
        self,
        mode: str,
        mode_freq_GHz: float,
        resonator_name: str,
        waveguide_name: str,
        impedance_ohm: float,
        qiskit_component_names: list,
        resonator_type: Literal["lambda_4", "lambda_2"],
        open_pins: Optional[list] = None,
        x_buffer_width_mm: float = 2,
        y_buffer_width_mm: float = 2,
        render_qiskit_metal: Optional[Callable] = None,
        render_qiskit_metal_kwargs: Optional[dict] = None,
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
            render_qiskit_metal=render_qiskit_metal,
            render_qiskit_metal_kwargs=render_qiskit_metal_kwargs,
            percent_error=percent_error,
            nbr_passes=nbr_passes,
        )
        self.resonator_name = resonator_name
        self.waveguide_name = waveguide_name
        self.impedance_ohm = impedance_ohm
        self.resonator_type = resonator_type
        self.kappa = None

    def get_kappa_estimate(self) -> float:
        """Calculate the resonator decay rate (kappa) into the waveguide.

        This method extracts the coupling capacitance between the resonator and waveguide
        from the simulation results and calculates the decay rate using a simplified model.
        The calculation assumes the resonator and waveguide have the same impedance.

        Returns:
            float: The estimated kappa (decay rate) in Hz.

        Raises:
            AssertionError: If capacitance_matrix_fF has not been set (run simulate_capacitance_matrix first).

        Note:
            The formula used is: κ = Z₀²·ω³·C²/π/(2π)
            Where:

            - Z₀ is the impedance (ohms)
            - ω is the angular frequency (2π·f)
            - C is the coupling capacitance (fF)

            For quarter-wavelength resonators, the result is doubled.
        """
        assert (
            self.capacitance_matrix_fF is not None
        ), "capacitance_matrix_fF is not set, you need to run .simulate_capacitance_matrix()."

        omega = self.mode_freq_GHz * 2 * np.pi

        Ccoupling = np.abs(
            self.capacitance_matrix_fF.loc[self.resonator_name, self.waveguide_name]
        )

        Z0 = self.impedance_ohm
        unit_conversion = 1e-3  # GHz^3 * fF^2
        kappa = Z0**2 * omega**3 * Ccoupling**2 / np.pi / (2 * np.pi) * unit_conversion

        if self.resonator_type == "lambda_4":
            kappa *= 2

        self.kappa = kappa
        return kappa

    def get_decay_parameter_value(self) -> float:
        """Get the resonator decay rate (kappa) into the waveguide.

        Returns:
            float: The estimated kappa (decay rate) in Hz.
        """
        return self.get_kappa_estimate()
