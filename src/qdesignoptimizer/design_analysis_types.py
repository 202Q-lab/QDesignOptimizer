from enum import Enum
from typing import Callable, Dict, List, Literal, Tuple, Union

from qiskit_metal.designs.design_base import QDesign

from qdesignoptimizer.sim_capacitance_matrix import CapacitanceMatrixStudy


class TargetType(Enum):
    FREQUENCY = "FREQUENCY"
    ANHARMONICITY = "ANHARMONICITY"
    KAPPA = "KAPPA"
    CHI = "CHI"

    INVERSE_SQUARED = "INVERSE_SQUARED"
    INVERSE_THREE_HALVES = "INVERSE_THREE_HALVES"
    INVERSE = "INVERSE"
    INVERSE_SQRT = "INVERSE_SQRT"
    SQRT = "SQRT"
    LINEAR = "LINEAR"
    THREE_HALVES = "THREE_HALVES"
    SQUARED = "SQUARED"


def convert_target_type_to_power(target_type: TargetType) -> float:
    """Convert TargetType to power p for the dependency of the target value on the design variable."""
    if target_type.value == TargetType.SQUARED.value:
        p = 2
    elif target_type.value == TargetType.THREE_HALVES.value:
        p = 1.5
    elif target_type.value == TargetType.LINEAR.value:
        p = 1
    elif target_type.value == TargetType.SQRT.value:
        p = 0.5
    elif target_type.value == TargetType.INVERSE_SQRT.value:
        p = -0.5
    elif target_type.value == TargetType.INVERSE.value:
        p = -1
    elif target_type.value == TargetType.INVERSE_THREE_HALVES.value:
        p = -1.5
    elif target_type.value == TargetType.INVERSE_SQUARED.value:
        p = -2
    else:
        raise ValueError(f"{target_type} cannot be identified")
    return p


BRANCH_PARAMETER = Tuple[str, str]
"""Example ("branch1", "qubit_freq")"""


class OptTarget:

    def __init__(
        self,
        system_target_param: Union[
            BRANCH_PARAMETER,
            Literal["CROSS_BRANCH_NONLIN", "CAPACITANCE_MATRIX_ELEMENTS"],
        ],
        involved_mode_freqs: List[Union[tuple, str]],
        design_var: str,
        design_var_constraint: object,
        prop_to: Callable[
            [Dict[str, Union[float, int]], Dict[str, Union[float, int]]], None
        ] = None,
        independent_target: bool = False,
    ):
        """Class for optimization target.

        Args:
            system_target_param: system target parameter to be optimized,
                (str, str) example ('branch1', qubit_freq')
                (str) CROSS_BRANCH_NONLIN is used when defining non-linear cross branch coupling
                (str) CAPACITANCE_MATRIX_ELEMENTS is used when defining capacitance matrix elements
            involved_mode_freqs (list): mode freqs involved in target,
                Example [('BRANCH_1, 'res_freq'), ('BRANCH_1, 'qubit_freq')]
                If system_target_param is CAPACITANCE_MATRIX_ELEMENTS, involved_mode_freqs should be
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
        self.system_target_param = system_target_param
        self.involved_mode_freqs = involved_mode_freqs
        self.design_var = design_var
        self.design_var_constraint = design_var_constraint
        self.prop_to = prop_to
        self.independent_target = independent_target


class ScatteringStudy:
    def __init__(
        self,
        nbr_passes: int = 18,
        max_delta_s: float = 0.005,
        basis_order=-1,  # Mixed order
    ):
        """Scattering study for DesignAnalysis.

        Args:
            mode_freqs (list): list of (branch, freq_name) of modes in component_names, simulated nbr of modes = len(mode_freqs), example: [('BRANCH_1, 'qubit_freq')]
            nbr_passes (int): max nbr of passes in driven modal simulation
            max_delta_s (float): max delta s in driven modal simulation
            basis_order (int): basis order in driven modal simulation

        """

        self.nbr_passes = nbr_passes
        self.max_delta_s = max_delta_s
        self.basis_order = basis_order


class MiniStudy:
    def __init__(
        self,
        component_names: list,
        port_list: list,
        open_pins: list,
        mode_freqs: List[tuple],
        nbr_passes: int = 10,
        delta_f: float = 0.1,
        jj_var: object = {},
        jj_setup: object = {},
        design_name: str = "mini_study",
        project_name: str = "dummy_project",
        x_buffer_width_mm=0.5,
        y_buffer_width_mm=0.5,
        max_mesh_length_port="3um",
        max_mesh_length_lines_to_ports="5um",
        allow_crude_decay_estimates=True,
        adjustment_rate: float = 1.0,
        render_qiskit_metal_eigenmode_kw_args: dict = {},
        scattering_studies: List[ScatteringStudy] = None,
        capacitance_matrix_studies: List[CapacitanceMatrixStudy] = [],
    ):
        """Mini_study for eigenmode simulation and energy participation (EPR) analysis in DesignAnalysis.

        Args:
            component_names (list(str)): List of names
            port_list (list): component pins with ports, example with 50 Ohm: [(comp_name,'pin_name', 50)],
            open_pins (list): pins to be left open, example: [(comp_name, 'pin_name')],
            mode_freqs (list): list of modes (branch, freq_name) to simulate in increasing frequency order, simulated nbr of modes = len(mode_freqs)
                               If the mode_freqs is empty, eigenmode and EPR analysis will be skipped.
                               Example: [('BRANCH_1, 'qb_freq'), ('BRANCH_1, 'res_freq')]
            nbr_passes (int): nbr of passes in eigenmode simulation
            delta_f (float): Convergence freq max delta percent diff
            jj_var (object):  junction variables, example: {'Lj': '10 nH', 'Cj': '0 fF'}
            jj_setup (object): junction setup, example: {'Lj_variable': 'Lj', 'rect': 'JJ_rect_Lj_Q1_rect_jj', 'line': 'JJ_Lj_Q1_rect_jj_', 'Cj_variable': 'Cj'}
            design_name (str): name of design
            project_name (str): name of project (default: dummy_project
            x_buffer_width_mm (float): x buffer width in driven modal simulation
            y_buffer_width_mm (float): y buffer width in driven modal simulation
            max_mesh_length_port (str): max mesh length of port
            max_mesh_length_lines_to_ports (str): max mesh length of lines to ports to enhance accuracy of decay estiamtes
            allow_crude_decay_estimates (bool): if True: use default mesh to ports which gives unreliable decay estimates in Eigenmode sim
            adjustment_rate (float): rate of adjustment of design variable w.r.t. to calculated optimal values. Example 0.7 is slower but might be more robust.
            render_qiskit_metal_eigenmode_kw_args (dict): kw_args for render_qiskit_metal used during eigenmode and EPR analysis,
                                                          Example: {'include_charge_line': True}
            scattering_studies (List[ScatteringStudy]): list of ScatteringStudy objects
            capacitance_matrix_studies (List[CapacitanceMatrixStudy]): list of CapacitanceMatrixStudy objects
        """
        self.component_names = component_names
        self.port_list = port_list
        self.open_pins = open_pins
        self.mode_freqs = mode_freqs
        self.nbr_passes = nbr_passes
        self.delta_f = delta_f
        self.jj_var = jj_var
        self.jj_setup = jj_setup
        self.design_name = design_name
        self.project_name = project_name
        self.x_buffer_width_mm = x_buffer_width_mm
        self.y_buffer_width_mm = y_buffer_width_mm
        self.max_mesh_length_port = max_mesh_length_port
        self.max_mesh_length_lines_to_ports = max_mesh_length_lines_to_ports
        self.allow_crude_decay_estimates = allow_crude_decay_estimates
        self.adjustment_rate = adjustment_rate
        self.render_qiskit_metal_eigenmode_kw_args = (
            render_qiskit_metal_eigenmode_kw_args
        )
        self.scattering_studies = scattering_studies
        self.capacitance_matrix_studies = capacitance_matrix_studies

    #     self._validate_scattering_studies()

    # def _validate_scattering_studies(self):
    #     """Validate scattering_studies."""
    #     if self.scattering_studies is None:
    #         return
    #     for scatteringStudy in self.scattering_studies:
    #         for scat_mode_freq in scatteringStudy.mode_freqs:
    #             assert (
    #                 scat_mode_freq in self.mode_freqs
    #             ), f"ScatteringStudy mode {scat_mode_freq} not found in MiniStudy mode_freqs {self.mode_freqs}"


class DesignAnalysisState:
    def __init__(
        self,
        design: QDesign,
        render_qiskit_metal: Callable,
        system_target_params: dict,
        system_optimized_params: dict = None,
    ):
        """Class for DesignAnalysis.

        Args:
            design (QDesign): QDesign object
            render_qiskit_metal (Callable): function which will be run to update design parameters,
                                            Format: render_qiskit_metal(design, **kw_args)
            system_target_params (dict): system target parameters in Hz, example: {'branch_1': {'qubit_freq': 5e9}}
            system_optimized_params (dict): system optimized parameters in Hz, example: {'branch_1': {{'qubit_freq': 5e9}}

        """
        self.design = design
        self.render_qiskit_metal = render_qiskit_metal
        self.system_target_params = system_target_params
        self.system_optimized_params = system_optimized_params
