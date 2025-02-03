from qiskit_metal.qt.utils.utils_design_variables import design_var_lj, design_var_cj
import qiskit_metal.qt.simulation.utils.utils_design_variables as u

# Fixed design constants
LINE_50_OHM_WIDTH = '16.51um'
LINE_50_OHM_GAP = '10um'

RESONATOR_WIDTH = '20um'
RESONATOR_GAP = '20um'

BEND_RADIUS = '99um'

# Design parameters for optimization
def design_var_res_length(branch: int):
    return f'design_var_res_length_{branch}'

def design_var_res_coupl_length(branch: int):
    return f'design_var_ind_coupl_length_{branch}'

def design_var_qb_pad_width(branch: int):
    return f'design_var_qb_pad_width_{branch}'

def design_var_res_qb_coupl_length(branch: int):
    return f'design_var_res_qb_coupl_length_{branch}'

JUNCTION_VARS = {
    # qubit junction
    design_var_lj(u.name_qb(0)): '12nH',
    design_var_lj(u.name_qb(1)): '12nH',
    design_var_lj(u.name_qb(2)): '12nH',
    design_var_lj(u.name_qb(3)): '12nH',
    design_var_lj(u.name_qb(4)): '12nH',

    design_var_cj(u.name_qb(0)): '0fF',
    design_var_cj(u.name_qb(1)): '0fF',
    design_var_cj(u.name_qb(2)): '0fF',
    design_var_cj(u.name_qb(3)): '0fF',
    design_var_cj(u.name_qb(4)): '0fF',
}

DESIGN_VARS = {
    **JUNCTION_VARS, 
    # resonator
    design_var_res_length(0): '5mm',
    design_var_res_length(1): '5mm',
    design_var_res_length(2): '5mm',
    design_var_res_length(3): '5mm',
    design_var_res_length(4): '5mm',

    design_var_res_coupl_length(0): '200um',
    design_var_res_coupl_length(1): '200um',
    design_var_res_coupl_length(2): '200um',
    design_var_res_coupl_length(3): '200um',
    design_var_res_coupl_length(4): '200um',

    # qubit
    design_var_qb_pad_width(0): '400um',
    design_var_qb_pad_width(1): '400um',
    design_var_qb_pad_width(2): '400um',
    design_var_qb_pad_width(3): '400um',
    design_var_qb_pad_width(4): '400um',

    # resonator - qubit
    design_var_res_qb_coupl_length(0): '120um',
    design_var_res_qb_coupl_length(1): '120um',
    design_var_res_qb_coupl_length(2): '120um',
    design_var_res_qb_coupl_length(3): '120um',
    design_var_res_qb_coupl_length(4): '120um',
    }