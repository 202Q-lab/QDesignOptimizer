import qdesignoptimizer.utils.constants as dc
from qdesignoptimizer.utils.sim_plot_progress import OptPltSet

PLOT_SETTINGS = {
    "RES": [
        OptPltSet(dc.ITERATION, dc.RES_FREQ),
        OptPltSet(dc.ITERATION, dc.RES_KAPPA),
    ],
    "QUBIT": [
        OptPltSet(dc.ITERATION, dc.QUBIT_FREQ),
        OptPltSet(dc.ITERATION, dc.RES_QUBIT_CHI),
    ],
}
