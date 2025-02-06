from qdesignoptimizer.utils.utils_design_variables import *


# Extra design variables
def design_var_cl_pos_x(branch: int):
    return f"design_var_cl_pos_x{branch}"


def design_var_cl_pos_y(branch: int):
    return f"design_var_cl_pos_y{branch}"


# Extra component names
def name_lp_chargeline(branch_number: int):
    return f"NAME_LP_chargeline{branch_number}"


