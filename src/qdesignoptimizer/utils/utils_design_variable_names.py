from typing import Literal, Union

from qiskit_metal.designs.design_planar import DesignPlanar

from qdesignoptimizer.utils.utils_parameter_names import Mode


def add_design_variables_to_design(
    design: DesignPlanar, design_variables: dict[str:str]
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
    return f"design_var_coupl_length_{identifier_1}_{identifier_2}"


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


# Component names


def name_(identifier: Union[str, int]):
    return f"name_{identifier}"


def name_mode(identifier: Mode):
    return f"name_{identifier}"


def name_mode_to_mode(identifier_1: Mode, identifier_2: Mode):
    return f"name_{identifier_1}_to_{identifier_2}"


def name_tee(identifier: Union[str, int]):
    return f"name_tee{identifier}"


def name_lp(identifier: Union[str, int]):
    return f"name_lp{identifier}"


def name_charge_line(identifier: Union[str, int]):
    return f"name_charge_line{identifier}"


def name_flux_line(identifier: Union[str, int]):
    return f"name_flux_line_{identifier}"


def name_lp_to_tee(lp_identifier: Union[str, int], tee_identifier: Union[str, int]):
    return f"name_lp{lp_identifier}_to_tee{tee_identifier}"


def name_tee_to_tee(tee_identifier1: Union[str, int], tee_identifier2: Union[str, int]):
    return f"name_tee{tee_identifier1}_to_tee{tee_identifier2}"


def name_lp_to_chargeline(
    lp_identifier: Union[str, int], chargeline_identifier: Union[str, int]
):
    return f"name_lp{lp_identifier}_to_chargeline{chargeline_identifier}"
