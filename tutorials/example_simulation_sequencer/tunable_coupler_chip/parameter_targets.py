"""The target values the studies of this project tune towards: the
tunable-coupler tutorial chip's (``examples_coupled_transmons_tunable_coupler_chip``).
Edit a value and the next queued study uses it."""

import names as n

from qdesignoptimizer.utils.names_parameters import param, param_nonlin

PARAM_TARGETS = {
    param(n.QUBIT_1, n.FREQ): 4.16e9,
    param(n.QUBIT_2, n.FREQ): 4.0e9,
    param(n.COUPLER_12, n.FREQ): 5.45e9,
    param(n.RESONATOR_1, n.FREQ): 7.12e9,
    param(n.RESONATOR_2, n.FREQ): 7.07e9,
    param_nonlin(n.QUBIT_1, n.QUBIT_1): 220e6,  # qubit anharmonicity
    param_nonlin(n.QUBIT_2, n.QUBIT_2): 210e6,
    param_nonlin(n.COUPLER_12, n.COUPLER_12): 90e6,  # coupler anharmonicity
    param_nonlin(n.QUBIT_1, n.RESONATOR_1): 0.17e6,  # qubit-resonator chi
    param_nonlin(n.QUBIT_2, n.RESONATOR_2): 0.14e6,
    param_nonlin(n.COUPLER_12, n.QUBIT_1): 4.1e6,  # coupler-qubit chi
    param_nonlin(n.COUPLER_12, n.QUBIT_2): 3.5e6,
}
