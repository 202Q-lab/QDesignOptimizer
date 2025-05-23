from qdesignoptimizer.utils.names_design_variables import *
from qdesignoptimizer.utils.names_parameters import *
from qdesignoptimizer.utils.names_qiskit_components import *

"""These * imports allow you to use all generic function generators for design variables and qiskit components names."""


# Extra design variables, which you can define if the generic ones are not enough
def design_var_my_custom(identifier: int):
    return f"design_var_my_custom_{identifier}"


# Extra component names, which you can define if the generic ones are not enough
def name_my_custom_qiskit_component(identifier: int):
    return f"name_my_custom_qiskit_component{identifier}"

# Name of the chip
CHIP_NAME = "multi_transmon_chip"

# Modes names for all modes in the design, which you define according to your design needs
NBR_1 = 1
NBR_2 = 2


QUBIT_1 = mode(QUBIT, identifier=NBR_1)
RESONATOR_1 = mode(RESONATOR, identifier=NBR_1)
QUBIT_2 = mode(QUBIT, identifier=NBR_2)
RESONATOR_2 = mode(RESONATOR, identifier=NBR_2)
COUPLER_12 = mode(COUPLER, identifier="1to2")  # Coupler between qubit 1 and qubit 2
