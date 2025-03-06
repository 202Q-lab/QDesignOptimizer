from typing import Callable, Dict, List, Literal, Union

from qiskit_metal.designs.design_base import QDesign

from qdesignoptimizer.sim_capacitance_matrix import CapacitanceMatrixStudy
from qdesignoptimizer.utils.names_parameters import Mode


class MeshingMap:
    """
    A class to map a component class to a function that generates mesh names.

    Attributes:
        component_class: The class of the component being meshed.
        mesh_names: A callable function that generates mesh names from component names.
    """

    def __init__(
        self, component_class: type, mesh_names: Callable[[List[str]], List[str]]
    ):
        """
        Initializes the MeshingMap with a component class and a mesh name function.

        Args:
            component_class (type): The component class to be meshed.
            mesh_names (Callable[[List[str]], List[str]]): A function that takes a list
                of component names and returns a list of mesh names.
        """
        self.component_class = component_class
        self.mesh_names = mesh_names


class OptTarget:
    """Class for optimization target.

    Args:
        target_param_type: system target parameter to be optimized,
            "freq", "kappa", "nonlinearity", "capacitance"
        involved_modes (list): mode freqs involved in target except when target_param_type is "capacitance", involved_modes should be
            the names of the TWO capacitive islands as optained from capacitance matrix simulation.
            Note that the capacitances can correspond to two islands on a split transmon, a charge lines etc.
            Example: ['capacitance_name_1', 'capacitance_name_2']
        design_var (str): design variable to be varied
        design_var_constraint (object): design variable constraint, example {'larger_than': '10 um', 'smaller_than': '100 um'}. The constraints are checked and enforced in each iteration of the optimization after all design variables have been updated by the algorithm.
        prop_to (Callable): Callable which should accept the system_params (s) and the design_variables (v) dicts and return the proportionality factor.
            IMPORTANT!!! The units of the design variables MUST be consistent if the prop_to expression cannot be factorized into a chain of functions only depending on a single design variable each, such as func1(v[PARAM_X])*func2(PARAM_Y)... For example: (v[PARAM_X] - v[PARAM_Y]) requires PARAM_X and PARAM_Y to have the same units.
            Example: prop_to=lambda s, v: s[ac.QUBIT_FREQ] / np.sqrt(v[dv.DESIGN_VAR_LJ_ATS])
        independent_target: Mark independent_target=True if the target only depends on a single design variable and not on any system parameter. This allows the optimizer to solve this OptTarget independently, making it faster and more robust.
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
            Callable[[Dict[str, Union[float, int]], Dict[str, Union[float, int]]], None]
        ] = None,
        independent_target: bool = False,
    ):

        self.target_param_type = target_param_type
        self.involved_modes = involved_modes
        self.design_var = design_var
        self.design_var_constraint = design_var_constraint
        self.prop_to = prop_to
        self.independent_target = independent_target


class MiniStudy:
    """Mini_study for eigenmode simulation and energy participation (EPR) analysis in DesignAnalysis.

    Args:
        qiskit_component_names (list(str)): List of names
        port_list (list): component pins with ports, example with 50 Ohm: [(comp_name,'pin_name', 50)],
        open_pins (list): pins to be left open, example: [(comp_name, 'pin_name')],
        modes (list): list of modes to simulate in increasing frequency order, simulated group of modes = len(modes)
                           If the mode_freqs is empty, eigenmode and EPR analysis will be skipped.
                           Example: [qubit_1, resonator_1]
        nbr_passes (int): group of passes in eigenmode simulation
        delta_f (float): Convergence freq max delta percent diff
        jj_setup (object): junction setup, example: {'Lj_variable': 'Lj', 'rect': 'JJ_rect_Lj_Q1_rect_jj', 'line': 'JJ_Lj_Q1_rect_jj', 'Cj_variable': 'Cj'}
        design_name (str): name of design
        project_name (str): name of project (default: dummy_project
        x_buffer_width_mm (float): x buffer width in driven modal simulation
        y_buffer_width_mm (float): y buffer width in driven modal simulation
        max_mesh_length_port (str): max mesh length of port
        max_mesh_length_lines_to_ports (str): max mesh length of lines to ports to enhance accuracy of decay estiamtes
        build_fine_mesh (bool): if True: use default mesh to ports which gives unreliable decay estimates in Eigenmode sim
        cos_trunc (int): cosine truncation in the EPR analysis. You might have to lower cos_trunc if you simulate very many modes.
        fock_trunc (int): fock truncation in the EPR analysis. You might have to lower fock_trunc if you simulate very many modes.
        adjustment_rate (float): rate of adjustment of design variable w.r.t. to calculated optimal values. Example 0.7 is slower but might be more robust.
        render_qiskit_metal_eigenmode_kw_args (dict): kw_args for render_qiskit_metal used during eigenmode and EPR analysis,
                                                      Example: {'include_charge_line': True}
        capacitance_matrix_studies (List[CapacitanceMatrixStudy]): list of CapacitanceMatrixStudy objects
    """

    def __init__(
        self,
        qiskit_component_names: list,
        port_list: list,
        open_pins: list,
        modes: List[Mode],
        nbr_passes: int = 10,
        delta_f: float = 0.1,
        jj_setup: object = {},
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
        render_qiskit_metal_eigenmode_kw_args: dict = {},
        capacitance_matrix_studies: List[CapacitanceMatrixStudy] = [],
    ):
        self.qiskit_component_names = qiskit_component_names
        self.port_list = port_list
        self.open_pins = open_pins
        self.modes = modes
        self.nbr_passes = nbr_passes
        self.delta_f = delta_f
        self.jj_setup = jj_setup
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
        self.render_qiskit_metal_eigenmode_kw_args = (
            render_qiskit_metal_eigenmode_kw_args
        )
        self.capacitance_matrix_studies = capacitance_matrix_studies


class DesignAnalysisState:
    """Class for DesignAnalysis.

    Args:
        design (QDesign): QDesign object
        render_qiskit_metal (Callable): function which will be run to update design parameters,
                                        Format: render_qiskit_metal(design, `**kw_args`)
        system_target_params (dict): system target parameters in Hz, example: {'branch_1': {'qubit_freq': 5e9}}
        system_optimized_params (dict): system optimized parameters in Hz, example: {'branch_1': {{'qubit_freq': 5e9}}
    """

    def __init__(
        self,
        design: QDesign,
        render_qiskit_metal: Callable,
        system_target_params: dict,
        system_optimized_params: Optional[dict] = None,
    ):
        self.design = design
        self.render_qiskit_metal = render_qiskit_metal
        self.system_target_params = system_target_params
        self.system_optimized_params = system_optimized_params
