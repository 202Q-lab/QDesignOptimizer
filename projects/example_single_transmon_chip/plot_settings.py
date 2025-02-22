import qdesignoptimizer.utils.constants as dc
from qdesignoptimizer.sim_plot_progress import OptPltSet

PLOT_SETTINGS = {
    "RES": [
        OptPltSet(dc.ITERATION, dc.RES_FREQ),
        OptPltSet(dc.ITERATION, dc.RES_KAPPA),
    ],
    "QUBIT": [
        OptPltSet(dc.ITERATION, dc.QUBIT_FREQ),
        OptPltSet(dc.ITERATION, dc.QUBIT_ANHARMONICITY),
        OptPltSet(dc.ITERATION, dc.RES_QUBIT_CHI),
    ],
}


PLOT_SETTINGS_CHARGE_LINE_DECAY = {
    "QUBIT": [
        OptPltSet(dc.ITERATION, dc.QUBIT_CHARGE_LINE_LIMITED_T1),
    ],
}
