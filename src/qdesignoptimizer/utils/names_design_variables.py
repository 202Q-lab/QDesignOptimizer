from typing import Literal, Union

from qiskit_metal.designs.design_planar import DesignPlanar

from qdesignoptimizer.utils.names_parameters import Mode
from qdesignoptimizer.utils.names_qiskit_components import name_mode


def add_design_variables_to_design(
    design: DesignPlanar, design_variables: dict[str, str]
):
    """Add design variables to a Qiskit Metal design so that the variables can be used in render.

    Args:
        design (DesignPlanar): A Qiskit Metal design.
        design_variables (dict[str:str]): Design variables to add to the design.
    """
    for key, val in {**design_variables}.items():
        design.variables[key] = val


# Design variables
def design_var_length(identifier: str):
    return f"design_var_length_{identifier}"


def design_var_width(identifier: str):
    return f"design_var_width_{identifier}"


def design_var_gap(identifier: str):
    return f"design_var_gap_{identifier}"


def design_var_coupl_length(identifier_1: str, identifier_2: str):
    identifier_first, identifier_second = sorted([identifier_1, identifier_2])
    return f"design_var_coupl_length_{identifier_first}_{identifier_second}"


def design_var_lj(identifier: str):
    return f"design_var_lj_{identifier}"


def design_var_cj(identifier: str):
    return f"design_var_cj_{identifier}"


def design_var_cl_pos_x(identifier: Union[str, int]):
    return f"design_var_cl_pos_x_{identifier}"


def design_var_cl_pos_y(identifier: Union[str, int]):
    return f"design_var_cl_pos_y_{identifier}"


def junction_setup(mode: Mode, type: Literal[None, "linear"] = None):
    """Generate jj setup for

    Args:
        component_name (str): component name
        type (str): type of JJ, e.g. 'linear' for a SNAIL/ATS tuned to the Kerr-free point. Default is None = ordinary jj.

    Returns:
        Dict: jj setup
    """
    jj_name = f"jj_{name_mode(mode)}"
    setup = {
        jj_name: dict(
            rect=f"JJ_rect_Lj_{name_mode(mode)}_rect_jj",
            line=f"JJ_Lj_{name_mode(mode)}_rect_jj_",
            Lj_variable=design_var_lj(mode),
            Cj_variable=design_var_cj(mode),
        )
    }
    if type is not None:
        setup[jj_name]["type"] = type
    return setup
