from qdesignoptimizer.utils.utils_design_variables import *

# Fixed design constants
LINE_50_OHM_WIDTH = "16.51um"
LINE_50_OHM_GAP = "10um"

RESONATOR_WIDTH = "20um"
RESONATOR_GAP = "20um"

BEND_RADIUS = "99um"


# Design parameters for optimization
def design_var_res_length(branch: int):
    return f"design_var_res_length_{branch}"


def design_var_res_coupl_length(branch: int):
    return f"design_var_ind_coupl_length_{branch}"


def design_var_qb_pad_width(branch: int):
    return f"design_var_qb_pad_width_{branch}"


def design_var_res_qb_coupl_length(branch: int):
    return f"design_var_res_qb_coupl_length_{branch}"


