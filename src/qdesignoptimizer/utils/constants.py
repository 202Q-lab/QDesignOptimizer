from typing import List

# mode names
RESONATOR = "res"
QUBIT = "qubit"
CAVITY = "cavity"
COUPLER = "coupler"

# target type
FREQ = "freq"
KAPPA = "kappa"
T1_DECAY = "charge_line_limited_t1"
NONLINEARITY = "nonlinearity"
CROSS_KERR = "CROSS_KERR"



def mode_freq(mode_name):
    return mode_name + "_" + FREQ


def mode_kappa(mode_name):
    return mode_name + "_" + KAPPA

def mode_t1_decay(mode_name):
    return mode_name + "_" + T1_DECAY

def mode_freq_to_mode_kappa(mode_freq: str)->str:
    assert mode_freq.endswith("_"+FREQ), f"mode frequency {mode_freq} must end with _{FREQ}"
    return (mode_freq[:-4] + KAPPA)

def mode_freq_to_mode(mode_freq: str)->str:
    assert mode_freq.endswith("_"+FREQ), f"mode frequency {mode_freq} must end with _{FREQ}"
    return (mode_freq[:-5])

def cross_kerr(branch_list: List[str],mode_list: List[str]):
    if len(branch_list)!=2 or len(mode_list)!=2:
        raise "branch_lis and mdoe_list both must contain two elements"
    mode_list,branch_list = zip(*sorted(zip(mode_list,branch_list))) # zip(*___) is the inverse of zip.
    return ((branch_list[0], mode_list[0]), (branch_list[1], mode_list[1]))

def mode_type(mode_name:str, target_type:str)->str:
    assert (target_type in [FREQ, KAPPA, T1_DECAY]), f"target_type {target_type} must be in {[FREQ, KAPPA, T1_DECAY]}"
    return mode_name+"_"+target_type



CROSS_BRANCH_NONLIN = "CROSS_BRANCH_NONLIN"

CAPACITANCE_MATRIX_ELEMENTS = "CAPACITANCE_MATRIX_ELEMENTS"
""" dict: Maps branch to capacitance matrix elements in capacitance matrix simulation.
    Capacitance matrix elements are in femto Farads (fF).

    Format: (capacitance_name, capacitance_name): value

    Example: {
        ('comb_NAME_QB1', 'comb_NAME_QB1'): 100,
        ('comb_NAME_QB1', 'comb_NAME_QB2'): 5,
        }
"""

BRANCH_1 = "BRANCH_1"
BRANCH_2 = "BRANCH_2"
BRANCH_3 = "BRANCH_3"
BRANCH_4 = "BRANCH_4"
BRANCH_5 = "BRANCH_5"
BRANCH_6 = "BRANCH_6"
BRANCH_7 = "BRANCH_7"
BRANCH_8 = "BRANCH_8"
BRANCH_9 = "BRANCH_9"
BRANCH_10 = "BRANCH_10"


def branch_str(branch):
    return f"BRANCH_{branch}"


def branch_str(branch):
    return f"BRANCH_{branch}"


# Resonator
RES_FREQ = "res_freq"
"""omega/2pi (Hz)"""
RES_KAPPA = "res_kappa"
"""kappa/2pi (Hz)"""
RES_KERR = "res_kerr"  # Hz
KERR = "kerr"  # Hz


# Qubit
QUBIT_FREQ = "qubit_freq"  # Hz - best estimate so far
QUBIT_FREQ_BARE = "qubit_freq_bare"  # Hz
QUBIT_PURCELL_DECAY = "qubit_purcell_decay"  # Hz
QUBIT_CHARGE_LINE_LIMITED_T1 = "qubit_charge_line_limited_t1"  # Hz

QUBIT_ANHARMONICITY = "qubit_anharmonicity"  # Hz
ANHARMONICITY = "anharmonicity"  # Hz


QUBIT_T1 = "qubit_T1"  # s
QUBIT_T2 = "qubit_T2"  # s

# Cavity
CAVITY_FREQ = "cavity_freq"  # Hz - Best estimate for the cavity frequency
CAVITY_KERR = "cavity_kerr"  # Hz
CAVITY_PURCELL_DECAY = "cavity_purcell_decay"  # Hz
CAVITY_RES_CROSS_KERR = "cavity_res_cross_kerr"  # Hz
CAVITY_KAPPA = "cavity_kappa"  # Hz
CAVITY_CHARGE_LINE_LIMITED_T1 = "cavity_charge_line_limited_t1"  # Hz

# Coupler
COUPLER_FREQ = "coupler_freq"
COUPLER_KAPPA = "coupler_kappa"


CAVITY_T1 = "cavity_T1"  # Hz

# %%  Coupling
# Resonator qubit interaction
RES_QUBIT_CHI = "res_qubit_chi"  # Hz
CHI = "chi"  # Hz
RES_QUBIT_CHI_PRIME = "res_qubit_chi_prime"  # Hz

# Cavity qubit interaction
QUBIT_CAVITY_G = "qubit_cavity_g"  # Hz
CAVITY_QUBIT_CHI = "cavity_qubit_chi"  # Hz
CAVITY_QUBIT_CHI_PRIME = "cavity_qubit_chi_prime"  # Hz

# Cavity coupler interaction
CAVITY_COUPLER_G = "cavity_coupler_g"  # Hz
CAVITY_COUPLER_CHI = "cavity_coupler_chi"  # Hz

ITERATION = "ITERATION"
