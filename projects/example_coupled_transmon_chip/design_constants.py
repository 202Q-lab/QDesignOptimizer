from qdesignoptimizer.utils.constants import *
from qdesignoptimizer.utils.utils_parameter_names import mode

# Fixed design constants
LINE_50_OHM_WIDTH = "16.51um"
LINE_50_OHM_GAP = "10um"

RESONATOR_WIDTH = "20um"
RESONATOR_GAP = "20um"

BEND_RADIUS = "99um"

# Modes
QUBIT_1 = mode(QUBIT, nbr=1)
RESONATOR_1 = mode(RESONATOR, nbr=1)
QUBIT_2 = mode(QUBIT, nbr=2)
RESONATOR_2 = mode(RESONATOR, nbr=2)
COUPLER_12 = mode(COUPLER, nbr="1to2")  # Coupler between qubit 1 and qubit 2
