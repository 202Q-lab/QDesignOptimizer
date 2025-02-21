# mode names
RESONATOR = "resonator"
QUBIT = "qubit"
CAVITY = "cavity"
COUPLER = "coupler"

# target type
FREQ = "freq"
KAPPA = "kappa"
T1_DECAY = "charge_line_limited_t1"  # TODO this could be just purcell decay
NONLINEARITY = "nonlinearity"


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


ITERATION = "ITERATION"
