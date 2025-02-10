from qdesignoptimizer.utils.utils_design_variables import *

# Design parameters for optimization
def design_var_res_length(branch: int):
    return f"design_var_res_length_{branch}"


def design_var_res_coupl_length(branch: int):
    return f"design_var_ind_coupl_length_{branch}"


def design_var_qb_pad_width(branch: int):
    return f"design_var_qb_pad_width_{branch}"


def design_var_res_qb_coupl_length(branch: int):
    return f"design_var_res_qb_coupl_length_{branch}"


