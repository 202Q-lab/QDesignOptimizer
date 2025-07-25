"""Data structures for organizing quantum circuit design optimization workflows."""

from typing import Callable, Dict, List, Literal, Optional, Union
from dataclasses import dataclass, fields

from qiskit_metal.designs.design_base import QDesign

from qdesignoptimizer.sim_capacitance_matrix import CapacitanceMatrixStudy
from qdesignoptimizer.utils.names_parameters import Mode

@dataclass
class InterfaceProperties:
    eps_r: float
    th: float
    tan_delta_surf: float

@dataclass
class Interfaces:
    substrate_air: InterfaceProperties
    metal_substrate: InterfaceProperties
    underside_surface: InterfaceProperties
    metal_air: InterfaceProperties

    def keys(self):
        """Iterator that yields interface names like dict.keys()"""
        for field in fields(self):
            yield field.name

@dataclass
class SurfaceProperties:
    interfaces: Interfaces
    sheet_material: str = 'Aluminum'  
    sheet_thickness: float = 0.000150  # unit mm

class MeshingMap:
    """
    Maps a component class to a function that generates mesh names.

    The MeshingMap provides a way to specify how meshing should be applied to specific
    component types during electromagnetic simulations. Optimal meshing is crucial for achieving
    faster simulation results while mantaining accuracy.

    Args:
        component_class (type): The Qiskit Metal component class to be meshed.
            The class itself, not an instance (e.g., TransmonComponent, not transmon1).
        mesh_names (Callable[[List[str]], List[str]]): A function that takes a component
            name and returns a list of mesh region names for that component. These names
            should follow the naming conventions of the simulation tool being used.

    Example:
        .. code-block:: python

            def transmon_mesh_names(component_name):
                return [f"{component_name}_junction_region", f"{component_name}_pad_gap"]

            meshing_map = MeshingMap(TransmonComponent, transmon_mesh_names)

    """

    def __init__(self, component_class: type, mesh_names: Callable[[str], List[str]]):
        """Initialize a MeshingMap with a component class and mesh name generator function."""
        self.component_class = component_class
        self.mesh_names = mesh_names


class OptTarget:
    """
    Defines an optimization target relating a quantum parameter to a design variable.

    An OptTarget establishes the relationship between a physical parameter of interest
    (like frequency, coupling strength, or decay rate) and a design variable that can be
    adjusted to achieve the target value. It specifies the functional dependency and
    constraints on the design variable.

    Args:
        target_param_type (Literal): The type of system parameter to optimize.
            One of: "freq" (frequency), "kappa" (decay rate), "charge_line_limited_t1" (T1 relaxation time),
            "nonlinearity" (anharmonicity/cross-Kerr), or "capacitance" (inter-component capacitance).
        involved_modes (List[Mode] | List[str]): The quantum modes involved in this parameter:

            - For single-mode parameters (freq, kappa, T1): [mode_name]
            - For two-mode parameters (nonlinearity): [mode1_name, mode2_name]
            - For capacitance: [island1_name, island2_name] (capacitive island names). Note that the
              capacitances can correspond to two islands on a split transmon, a charge line etc.

        design_var (str): The design variable name that will be adjusted to achieve the target.
        design_var_constraint (dict[str, str]): Constraints on the design variable, with keys:

            - "larger_than": Minimum allowed value with unit (e.g., "10um")
            - "smaller_than": Maximum allowed value with unit (e.g., "100um"). The constraints are checked and enforced in each iteration of the optimization after all design variables have been updated by the algorithm.

        prop_to (Callable): Function defining how the parameter depends on system parameters
            and design variables. Must accept (system_params, design_vars) and return a value.
            IMPORTANT: If the expression can't be factorized as a chain of functions depending
            on a single variable each, all design variables must have consistent units. For example,
            ``func1(v[PARAM_X])*func2(PARAM_Y)`` can accept parameters with different units, while ``(v[PARAM_X] - v[PARAM_Y])`` requires
            ``PARAM_X`` and ``PARAM_Y`` to have the same units.
        independent_target (bool): If True, this target only depends on a single design variable
            and not on any system parameter.This allows the optimizer to solve this OptTarget
            independently, making it faster and more robust.

    Note:
        The `prop_to` function is crucial as it defines the physical relationship between
        the design variable and the target parameter. For example, for a resonator frequency:
        lambda p, v: 1/v["resonator_length"] encodes the inverse relationship between
        resonator length and frequency.

    Example:
        .. code-block:: python

            # Target for a qubit frequency via Josephson inductance
            qubit_freq_target = OptTarget(
                target_param_type="freq",
                involved_modes=["qubit_1"],
                design_var="inductance_Lj",
                design_var_constraint={"larger_than": "0.1nH", "smaller_than": "10nH"},
                prop_to=lambda p, v: 1/np.sqrt(v["inductance_Lj"]),
                independent_target=True
            )

    """

    def __init__(
        self,
        target_param_type: Literal[
            "freq",
            "kappa",
            "charge_line_limited_t1",
            "nonlinearity",
            "capacitance",
        ],
        involved_modes: List[Mode] | List[str],
        design_var: str,
        design_var_constraint: dict[str, str],
        prop_to: Optional[
            Callable[
                [Dict[str, Union[float, int]], Dict[str, Union[float, int]]],
                float | int,
            ]
        ] = None,
        independent_target: bool = False,
    ):
        """Initialize an optimization target."""
        self.target_param_type = target_param_type
        self.involved_modes = involved_modes
        self.design_var = design_var
        self.design_var_constraint = design_var_constraint
        self.prop_to = prop_to
        self.independent_target = independent_target


class MiniStudy:
    """
    Configures a specific electromagnetic simulation study for circuit optimization.

    A MiniStudy defines the scope and settings for electromagnetic simulations used during
    the optimization process. It specifies which components to simulate, which quantum modes
    to analyze, and how to configure the simulator. It can configure both eigenmode simulations
    and energy participation (EPR) analyses and additional capacitance matrix simulations.


    Args:
        qiskit_component_names (list): List of Qiskit Metal component names to include in the simulation.
        port_list (list): List of ports in the format ``[(comp_name, pin_name, impedance_ohms)]``.
            Example: ``[("resonator_1", "port1", 50)]`` for a 50 Ohm port on resonator_1's port1.
        open_pins (list): List of pins to leave open-circuited, format: ``[(comp_name, pin_name)]``.
        modes (List[Mode]): List of mode names to simulate, in increasing frequency order.
            The number of modes in this list determines how many eigenmodes will be solved for.
            If this list is not empty and run_capacitance_studies_only is False, eigenmode and EPR analysis will run automatically.
        nbr_passes (int): Number of adaptive mesh refinement passes for eigenmode simulation.
            Higher values give more accurate results but take longer to simulate.
        delta_f (float): Absolute frequency tolerance (%) determining convergence of eigenmode sims.
            Smaller values give more accurate results but may require more passes.
        jj_setup (dict): Junction setup configuration for energy participation analysis.
            Example: ``{'Lj_variable': 'Lj', 'rect': 'JJ_rect_Lj_Q1', 'line': 'JJ_Lj_Q1'}``
        design_name (str): Name of the design.
        project_name (str): Name of the HFSS/simulation project (default: dummy_project).
        x_buffer_width_mm (float): x buffer width in driven modal simulation.
        y_buffer_width_mm (float): y buffer width in driven modal simulation.
        max_mesh_length_port (str): Maximum mesh element size at ports (with unit).
        max_mesh_length_lines_to_ports (str): Maximum mesh size for transmission lines to ports to
        enhance accuracy of decay estimates.
        hfss_wire_bond_size (int): Size parameter for wire bonds in HFSS.
        hfss_wire_bond_offset (str): Offset parameter for wire bonds in HFSS (with unit).
        hfss_wire_bond_threshold (str): Threshold parameter for wire bonds in HFSS (with unit).
        build_fine_mesh (bool): If True, use default mesh to ports which give unreliable
        decay estimates in Eigenmode simulations.
        adjustment_rate (float): Rate at which design variables are adjusted during optimization
            with respect to the proposed optimal values. Values <1.0 slow down changes for more
            stable convergence but slower optimization.
        cos_trunc (int): Cosine truncation order in energy participation (EPR) analysis.
            You might need to lower this when simulating many modes.
        fock_trunc (int): Fock space truncation in EPR analysis.
            You might need to lower this when simulating many modes.
        render_qiskit_metal_eigenmode_kw_args (dict): Additional keyword arguments for the
            render_qiskit_metal function used during eigenmode simulation.
        run_capacitance_studies_only (bool): If True, skip eigenmode simulations and only
            run the capacitance matrix studies.
        capacitance_matrix_studies (List[CapacitanceMatrixStudy]): List of capacitance matrix
            studies to run.
        surface_properties (SurfaceProperties): Surface properties for the design, including
            interfaces, sheet material, and thickness.     

    Example:
        .. code-block:: python

            mini_study = MiniStudy(
                qiskit_component_names=["transmon_1", "resonator_1"],
                port_list=[("resonator_1", "port_1", 50)],  # 50 Ohm port
                open_pins=[("transmon_1", "junction_pin")],
                modes=["qubit_1", "resonator_1"],
                jj_setup={"Lj_variable": "Lj", "rect": "JJ_rect_Lj_Q1"},
                nbr_passes=15,
                delta_f=0.05,
            )

    """

    def __init__(
        self,
        qiskit_component_names: list,
        port_list: list,
        open_pins: list,
        modes: List[Mode],
        nbr_passes: int = 10,
        delta_f: float = 0.1,
        jj_setup: Optional[dict] = None,
        design_name: str = "mini_study",
        project_name: str = "dummy_project",
        x_buffer_width_mm=0.5,
        y_buffer_width_mm=0.5,
        max_mesh_length_port="3um",
        max_mesh_length_lines_to_ports="5um",
        hfss_wire_bond_size=3,
        hfss_wire_bond_offset="0um",
        hfss_wire_bond_threshold="300um",
        build_fine_mesh=False,
        adjustment_rate: float = 1.0,
        cos_trunc=8,
        fock_trunc=7,
        render_qiskit_metal_eigenmode_kw_args: Optional[dict] = None,
        run_capacitance_studies_only: bool = False,
        capacitance_matrix_studies: Optional[List[CapacitanceMatrixStudy]] = None,
        surface_properties: Optional[SurfaceProperties] = None,
    ):
        """Initialize a MiniStudy for electromagnetic simulation configuration."""
        self.qiskit_component_names = qiskit_component_names
        self.port_list = port_list
        self.open_pins = open_pins
        self.modes = modes
        self.nbr_passes = nbr_passes
        self.delta_f = delta_f
        self.jj_setup: dict = jj_setup or {}
        self.design_name = design_name
        self.project_name = project_name
        self.x_buffer_width_mm = x_buffer_width_mm
        self.y_buffer_width_mm = y_buffer_width_mm
        self.max_mesh_length_port = max_mesh_length_port
        self.max_mesh_length_lines_to_ports = max_mesh_length_lines_to_ports
        self.hfss_wire_bond_size = hfss_wire_bond_size
        self.hfss_wire_bond_offset = hfss_wire_bond_offset
        self.hfss_wire_bond_threshold = hfss_wire_bond_threshold
        self.build_fine_mesh = build_fine_mesh
        self.adjustment_rate = adjustment_rate
        self.cos_trunc = cos_trunc
        self.fock_trunc = fock_trunc
        self.render_qiskit_metal_eigenmode_kw_args: dict = (
            render_qiskit_metal_eigenmode_kw_args or {}
        )
        self.run_capacitance_studies_only = run_capacitance_studies_only
        self.capacitance_matrix_studies: List[CapacitanceMatrixStudy] = (
            capacitance_matrix_studies or []
        )
        self.surface_properties = surface_properties
        self.interfaces = surface_properties.interfaces if surface_properties else None
        self.sheet_thickness = surface_properties.sheet_thickness if surface_properties else 0.000150
        self.sheet_material = surface_properties.sheet_material if surface_properties else 'Aluminum'



class DesignAnalysisState:
    """
    Describes the state of a quantum circuit design for optimization analysis.

    The DesignAnalysisState class serves as a container for all information needed to
    perform optimizations on a quantum circuit design. It holds the Qiskit Metal design
    object, the rendering function, and the target and optimized parameter values.

    Args:
        design (QDesign): The Qiskit Metal design object containing the circuit components
            and their properties.
        render_qiskit_metal (Callable): Function used to render the design with updated
            parameters. Must accept a design object and keyword arguments.
            Format: ``render_qiskit_metal(design, **kwargs)``
        system_target_params (dict): Dictionary of target values for system parameters.
            Keys follow standard naming conventions and values are in Hz for frequencies, etc.
            Example: ``{´branch_1´: {´qubit_freq´: 5e9}}``
        system_optimized_params (Optional[dict]): Dictionary of current optimized values
            for system parameters. Initially can be None and will be populated during
            optimization. Should follow the same structure as system_target_params.

    Note:
        The system_target_params and system_optimized_params dictionaries should use
        standardized parameter names from the names_parameters module, such as:

        - 'qubit_1_freq': Qubit frequency
        - 'resonator_1_kappa': Resonator decay rate
        - 'qubit_1_to_resonator_1_nonlin': Qubit-resonator dispersive shift

    Example:

    .. code-block:: python

        from qdesignoptimizer.utils.names_parameters import (
            param,
            param_capacitance,
            param_nonlin,
        )
        # System target parameters (in Hz and s)
        target_params = {
            param(n.QUBIT_1, n.FREQ): 4e9,
            param(n.QUBIT_1, n.CHARGE_LINE_LIMITED_T1): 20e-3,
            param(n.RESONATOR_1, n.FREQ): 6e9,
        }
        state = DesignAnalysisState(
            design=my_qiskit_design,
            render_qiskit_metal=render_function,
            system_target_params=target_params
        )

    """

    def __init__(
        self,
        design: QDesign,
        render_qiskit_metal: Callable,
        system_target_params: dict,
        system_optimized_params: Optional[dict] = None,
    ):
        """Initialize a design analysis state."""
        self.design = design
        self.render_qiskit_metal = render_qiskit_metal
        self.system_target_params = system_target_params
        self.system_optimized_params = system_optimized_params
