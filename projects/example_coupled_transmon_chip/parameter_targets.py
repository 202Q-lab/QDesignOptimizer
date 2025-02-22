import names as n

from qdesignoptimizer.utils.names_parameters import (
    FREQ,
    KAPPA,
    param,
    param_nonlin,
)

PARAM_TARGETS = {
    param(n.QUBIT_1, FREQ): 4e9,
    param(n.RESONATOR_1, FREQ): 6e9,
    param(n.RESONATOR_1, KAPPA): 1e6,
    param(n.QUBIT_2, FREQ): 5e9,
    param(n.RESONATOR_2, FREQ): 7e9,
    param(n.RESONATOR_2, KAPPA): 1e6,
    param(n.COUPLER_12, FREQ): 7.5e9,
    param_nonlin(n.QUBIT_1, n.QUBIT_1): 200e6,  # Qubit anharmonicity
    param_nonlin(n.QUBIT_1, n.RESONATOR_1): 1e6,  # Qubit resonaotr chi
    param_nonlin(n.QUBIT_2, n.QUBIT_2): 200e6,  # Qubit anharmonicity
    param_nonlin(n.QUBIT_2, n.RESONATOR_2): 1e6,  # Qubit resonaotr chi
}
