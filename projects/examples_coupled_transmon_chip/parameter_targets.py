import names as n

from qdesignoptimizer.utils.names_parameters import (
    param,
    param_capacitance,
    param_nonlin,
)

PARAM_TARGETS = {
    param(n.QUBIT_1, n.FREQ): 4e9,
    param(n.QUBIT_1, n.CHARGE_LINE_LIMITED_T1): 20e-3,
    param(n.RESONATOR_1, n.FREQ): 6e9,
    param(n.RESONATOR_1, n.KAPPA): 1e6,
    param(n.QUBIT_2, n.FREQ): 5e9,
    param(n.RESONATOR_2, n.FREQ): 7e9,
    param(n.RESONATOR_2, n.KAPPA): 1e6,
    param(n.COUPLER_12, n.FREQ): 9e9,
    param_nonlin(n.QUBIT_1, n.QUBIT_1): 210e6,  # Qubit anharmonicity
    param_nonlin(n.QUBIT_1, n.RESONATOR_1): 1.1e6,  # Qubit resonator chi
    param_nonlin(n.QUBIT_2, n.QUBIT_2): 200e6,  # Qubit anharmonicity
    param_nonlin(n.QUBIT_2, n.RESONATOR_2): 1e6,  # Qubit resonator chi
    param_capacitance("prime_cpw_name_tee1", "second_cpw_name_tee1"): -3,  # fF
}
