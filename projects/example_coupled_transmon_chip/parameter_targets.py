import design_constants as dc

from qdesignoptimizer.utils.utils_parameter_names import (
    FREQ,
    KAPPA,
    param,
    param_nonlin,
)

PARAM_TARGETS = {
    param(dc.QUBIT_1, FREQ): 4e9,
    param(dc.RESONATOR_1, FREQ): 6e9,
    param(dc.RESONATOR_1, KAPPA): 1e6,
    param(dc.QUBIT_2, FREQ): 5e9,
    param(dc.RESONATOR_2, FREQ): 7e9,
    param(dc.RESONATOR_2, KAPPA): 1e6,
    param(dc.COUPLER_12, FREQ): 7.5e9,
    param_nonlin(dc.QUBIT_1, dc.QUBIT_1): 200e6,  # Qubit anharmonicity
    param_nonlin(dc.QUBIT_1, dc.RESONATOR_1): 1e6,  # Qubit resonaotr chi
    param_nonlin(dc.QUBIT_2, dc.QUBIT_2): 200e6,  # Qubit anharmonicity
    param_nonlin(dc.QUBIT_2, dc.RESONATOR_2): 1e6,  # Qubit resonaotr chi
}
