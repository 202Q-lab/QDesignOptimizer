from typing import Literal, Union

from qiskit_metal.designs.design_planar import DesignPlanar


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


# Junction design variables
def design_var_lj(component_name: str):
    assert component_name.startswith("NAME_")
    return f"design_var_lj_{component_name}"


def design_var_cj(component_name: str):
    assert component_name.startswith("NAME_")
    return f"design_var_cj_{component_name}"


def junction_setup(
    component_name: Union[str, int], type: Literal[None, "linear"] = None
):
    """Generate jj setup for

    Args:
        component_name (str): component name
        type (str): type of JJ, e.g. 'linear' for a SNAIL/ATS tuned to the Kerr-free point. Default is None = ordinary jj.

    Returns:
        Dict: jj setup
    """
    jj_name = f"jj_{component_name}"
    setup = {
        jj_name: dict(
            rect=f"JJ_rect_Lj_{component_name}_rect_jj",
            line=f"JJ_Lj_{component_name}_rect_jj_",
            Lj_variable=design_var_lj(component_name),
            Cj_variable=design_var_cj(component_name),
        )
    }
    if type is not None:
        setup[jj_name]["type"] = type
    return setup


# Design variables
def design_var_res_length(branch: int):
    return f"design_var_res_length_{branch}"


def design_var_res_coupl_length(branch: int):
    return f"design_var_ind_coupl_length_{branch}"


def design_var_qb_pad_width(branch: int):
    return f"design_var_qb_pad_width_{branch}"


def design_var_res_qb_coupl_length(branch: int):
    return f"design_var_res_qb_coupl_length_{branch}"


def design_var_res_coupl_length(branch: int):
    return f"design_var_res_coupl_length_{branch}"


def design_var_qb_coupl_gap(branch: int):
    return f"design_var_qb_coupl_gap_{branch}"


# Component names
def name_res(branch_number: int):
    return f"NAME_RES{branch_number}"


def name_qb(branch_number: int):
    return f"NAME_QB{branch_number}"


def name_cav(branch_number: int):
    return f"NAME_CAV{branch_number}"


def name_coupler(branch_number: int):
    return f"NAME_COUPLER{branch_number}"


def name_tee(branch_number: int):
    return f"NAME_TEE{branch_number}"


def name_lp(branch_number: int):
    return f"NAME_LP{branch_number}"


def name_charge_line(branch_number: Union[str, int]):
    return f"NAME_CHARGE_LINE_{branch_number}"


def name_flux_line(component_name: str):
    assert component_name.startswith("NAME_")
    component = component_name.strip("NAME_")
    return f"NAME_FLUX_LINE_{component}"


def name_cav_to_qb(branch_number: Union[str, int]):
    return f"NAME_CAV{branch_number}_TO_QB{branch_number}"


def name_cav_to_coupler(
    branch_number: int, coupler_number: Union[str, int, None] = None
):
    return f"NAME_CAV{branch_number}_TO_COUPLER{coupler_number}"


def name_lp_to_tee(
    lp_branch_number: Union[str, int], tee_branch_number: Union[str, int]
):
    return f"NAME_LP{lp_branch_number}_TO_TEE{tee_branch_number}"


def name_tee_to_tee(
    tee_branch_number1: Union[str, int], tee_branch_number2: Union[str, int]
):
    return f"NAME_TEE{tee_branch_number1}_TO_TEE{tee_branch_number2}"

# Extra design variables
def design_var_cl_pos_x(branch: int):
    return f"design_var_cl_pos_x{branch}"


def design_var_cl_pos_y(branch: int):
    return f"design_var_cl_pos_y{branch}"


# Extra component names
def name_lp_chargeline(branch_number: int):
    return f"NAME_LP_chargeline{branch_number}"

def design_var_res_length(branch: int):
    return f"design_var_res_length_{branch}"


def design_var_res_coupl_length(branch: int):
    return f"design_var_res_coupl_length_{branch}"


def design_var_qb_pad_width(branch: int):
    return f"design_var_qb_pad_width_{branch}"


def design_var_res_qb_coupl_length(branch: int):
    return f"design_var_res_qb_coupl_length_{branch}"