from qdesignoptimizer.utils.names_design_variables import *


# Design parameters for optimization
def design_var_res_length(branch: int):
    return f"design_var_res_length_{branch}"


def design_var_res_coupl_length(branch: int):
    return f"design_var_ind_coupl_length_{branch}"


def design_var_qb_pad_width(branch: int):
    return f"design_var_qb_pad_width_{branch}"


def design_var_res_qb_coupl_length(branch: int):
    return f"design_var_res_qb_coupl_length_{branch}"


def design_var_cl_pos_x(branch: int):
    """Distance between the end of the charge line and the transmon pocket."""
    return f"design_var_cl_pos_x{branch}"


# Extra component names
def name_lp_chargeline(branch_number: int):
    return f"NAME_LP_chargeline{branch_number}"


def name_otg_chargeline(branch: int):
    return f"NAME_OTG_chargeline_{branch}"
