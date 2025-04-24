import names as n

from qdesignoptimizer.sim_plot_progress import OptPltSet
from qdesignoptimizer.utils.names_parameters import (
    param,
    param_nonlin,
)


def get_plot_settings_resonator(group: int):
    resonator = [n.RESONATOR_1, n.RESONATOR_2][group - 1]
    qubit = [n.QUBIT_1, n.QUBIT_2][group - 1]
    return {
        "FREQUENCIES": [
            OptPltSet(
                n.ITERATION,
                param(resonator, n.FREQ),
                y_label="Resonator Frequency",
                unit="GHz",
            ),
            OptPltSet(
                n.ITERATION, param(qubit, n.FREQ), y_label="Qubit Frequency", unit="GHz"
            ),
        ]
    }


def get_plot_settings_resonator_qubit(group: int):
    resonator = [n.RESONATOR_1, n.RESONATOR_2][group - 1]
    qubit = [n.QUBIT_1, n.QUBIT_2][group - 1]
    return {
        "FREQUENCIES": [
            OptPltSet(
                n.ITERATION,
                param(resonator, n.FREQ),
                y_label="Resonator Frequency",
                unit="GHz",
            ),
            OptPltSet(
                n.ITERATION, param(qubit, n.FREQ), y_label="Qubit Frequency", unit="GHz"
            ),
        ],
        "NONLINEARITIES": [
            OptPltSet(
                n.ITERATION,
                param_nonlin(resonator, qubit),
                y_label="Resonator-Qubit Chi",
                unit="MHz",
            ),
            OptPltSet(
                n.ITERATION,
                param_nonlin(qubit, qubit),
                y_label="Qubit Anharmonicity",
                unit="MHz",
            ),
        ],
    }
