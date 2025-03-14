import names as n

from qdesignoptimizer.sim_plot_progress import OptPltSet
from qdesignoptimizer.utils.names_parameters import (
    param,
    param_capacitance,
    param_nonlin,
)

PLOT_SETTINGS = {
    "RES": [
        OptPltSet(
            n.ITERATION, param(n.RESONATOR_1, n.FREQ), y_label="RES Freq", unit="GHz"
        ),
        OptPltSet(
            n.ITERATION, param(n.RESONATOR_1, n.KAPPA), y_label="RES Kappa", unit="MHz"
        ),
        OptPltSet(
            n.design_var_length(n.RESONATOR_1),
            param(n.RESONATOR_1, n.KAPPA),
            y_label="RES Kappa",
            unit="MHz",
        ),  # As an example that design variables can also be plotted for the results
    ],
    "QUBIT": [
        OptPltSet(n.ITERATION, param(n.QUBIT_1, n.FREQ), y_label="QB Freq", unit="GHz"),
        OptPltSet(
            n.ITERATION,
            param_nonlin(n.QUBIT_1, n.QUBIT_1),
            y_label="QB Anharm.",
            unit="MHz",
        ),
    ],
    "COUPLINGS": [
        OptPltSet(
            n.ITERATION,
            param_nonlin(n.RESONATOR_1, n.QUBIT_1),
            y_label="RES-QB Chi",
            unit="kHz",
        ),
    ],
}

PLOT_SETTINGS_TWO_QB = {
    "RES": [
        OptPltSet(
            n.ITERATION,
            [param(n.RESONATOR_1, n.FREQ), param(n.RESONATOR_2, n.FREQ)],
            y_label="RES Freq",
            unit="Hz",
        ),
        OptPltSet(
            n.ITERATION,
            [param(n.RESONATOR_1, n.KAPPA), param(n.RESONATOR_2, n.KAPPA)],
            y_label="RES Kappa",
            unit="Hz",
        ),
        OptPltSet(
            n.ITERATION,
            [
                param_nonlin(n.RESONATOR_1, n.RESONATOR_1),
                param_nonlin(n.RESONATOR_2, n.RESONATOR_2),
            ],
            y_label="RES Kerr",
            unit="Hz",
        ),
    ],
    "QUBIT": [
        OptPltSet(
            n.ITERATION,
            [param(n.QUBIT_1, n.FREQ), param(n.QUBIT_2, n.FREQ)],
            y_label="QB Freq",
        ),
        OptPltSet(
            n.ITERATION,
            [param_nonlin(n.QUBIT_1, n.QUBIT_1), param_nonlin(n.QUBIT_2, n.QUBIT_2)],
            y_label="QB Anharm.",
        ),
    ],
    "COUPLINGS": [
        OptPltSet(
            n.ITERATION,
            [
                param_nonlin(n.RESONATOR_1, n.QUBIT_1),
                param_nonlin(n.RESONATOR_2, n.QUBIT_2),
            ],
            y_label="RES-QB Chi",
        ),
    ],
}

PLOT_SETTINGS_CHARGE_LINE_DECAY = {
    "QUBIT": [
        OptPltSet(
            n.ITERATION,
            param(n.QUBIT_1, n.CHARGE_LINE_LIMITED_T1),
            y_label="T1 limit",
            y_scale="log",
            unit="s",
        )
    ],
}

PLOT_SETTINGS_RESONATOR_KAPPA = {
    "RESONATOR": [
        OptPltSet(
            n.ITERATION,
            param(n.RESONATOR_1, n.KAPPA),
            y_label="RES Kappa",
            y_scale="log",
        )
    ],
}

PLOT_SETTINGS_CAPACITANCE = {
    "CAP": [
        OptPltSet(
            n.ITERATION,
            param_capacitance("prime_cpw_name_tee1", "second_cpw_name_tee1"),
            y_label="Capacitance",
            unit="fF",
        )
    ],
}
