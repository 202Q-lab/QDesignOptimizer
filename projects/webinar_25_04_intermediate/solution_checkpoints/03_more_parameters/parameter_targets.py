import names as n

from qdesignoptimizer.utils.names_parameters import param, param_nonlin

PARAM_TARGETS = {
    param(n.RESONATOR_1, n.FREQ): 6e9,
    param(n.RESONATOR_2, n.FREQ): 7e9,
    param(n.QUBIT_1, n.FREQ): 4e9,
    param(n.QUBIT_2, n.FREQ): 5e9,
    param_nonlin(n.QUBIT_1, n.QUBIT_1): 200e6,  # Qubit anharmonicity
    param_nonlin(n.QUBIT_1, n.RESONATOR_1): 1e6,  # Qubit resonator chi
    param_nonlin(n.QUBIT_2, n.QUBIT_2): 200e6,  # Qubit anharmonicity
    param_nonlin(n.QUBIT_2, n.RESONATOR_2): 1e6,  # Qubit resonator chi
}
