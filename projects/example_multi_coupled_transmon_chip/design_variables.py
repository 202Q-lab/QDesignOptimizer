from src.utils.utils_design_variables import *

# Fixed design constants
LINE_50_OHM_WIDTH = '16.51um'
LINE_50_OHM_GAP = '10um'

RESONATOR_WIDTH = '20um'
RESONATOR_GAP = '20um'

BEND_RADIUS = '99um'


# Extra design variables
def design_var_cl_pos_x(branch: int):
    return f'design_var_cl_pos_x{branch}'

def design_var_cl_pos_y(branch: int):
    return f'design_var_cl_pos_y{branch}'

# Extra component names
def name_lp_chargeline(branch_number: int):
    return f'NAME_LP_chargeline{branch_number}'


JUNCTION_VARS = {
    # qubit junction
    design_var_lj(name_qb(0)): '14.074377200737976 nH',
    design_var_lj(name_qb(1)): '12nH',
    design_var_lj(name_qb(2)): '9.469791842945993 nH',
    design_var_lj(name_qb(3)): '12nH',

    design_var_cj(name_qb(0)): '0fF',
    design_var_cj(name_qb(1)): '0fF',
    design_var_cj(name_qb(2)): '0fF',
    design_var_cj(name_qb(3)): '0fF',
}

DESIGN_VARS = {
    **JUNCTION_VARS, 
    # resonator
    design_var_res_length(0): '7.432076775195457 mm',
    design_var_res_length(1): '7.5mm',
    design_var_res_length(2): '6.643301157224769 mm',
    design_var_res_length(3): '7.5mm',

    design_var_res_coupl_length(0): '579.8134731520797um',
    design_var_res_coupl_length(1): '200um',
    design_var_res_coupl_length(2): '561.6630563898493 um',
    design_var_res_coupl_length(3): '200um',

    # qubit
    design_var_qb_pad_width(0): '747.6203104123206 um',
    design_var_qb_pad_width(1): '400um',
    design_var_qb_pad_width(2): '759.4518671253024 um',
    design_var_qb_pad_width(3): '400um',

    # resonator - qubit
    design_var_res_qb_coupl_length(0): '203.0766408416638 um',
    design_var_res_qb_coupl_length(1): '120um',
    design_var_res_qb_coupl_length(2): '136.41351496912102 um',
    design_var_res_qb_coupl_length(3): '120um',

    # resonator coupler
    design_var_res_coupl_length(5): '4mm',

    # qubit coupler
    design_var_qb_coupl_gap(6):'20um',

    # charge line
    design_var_cl_pos_x(0): '200um',
    design_var_cl_pos_x(1): '200um',
    design_var_cl_pos_x(2): '200um',
    design_var_cl_pos_x(3): '200um',

    design_var_cl_pos_y(0): '1um',
    design_var_cl_pos_y(1): '1um',
    design_var_cl_pos_y(2): '1um',
    design_var_cl_pos_y(3): '1um',
    }