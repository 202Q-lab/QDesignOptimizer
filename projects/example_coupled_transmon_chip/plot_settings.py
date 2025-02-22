import names as n

import qdesignoptimizer.utils.constants as c
from qdesignoptimizer.sim_plot_progress import OptPltSet
from qdesignoptimizer.utils.names_parameters import param, param_nonlin

PLOT_SETTINGS = {
    "RES": [
        OptPltSet(c.ITERATION, param(n.RESONATOR_1, c.FREQ)),
        OptPltSet(c.ITERATION, param(n.RESONATOR_1, c.KAPPA)),
        OptPltSet(c.ITERATION, param_nonlin(n.RESONATOR_1, n.RESONATOR_1)),
    ],
    "QUBIT": [
        OptPltSet(c.ITERATION, param(n.QUBIT_1, c.FREQ)),
        OptPltSet(c.ITERATION, param_nonlin(n.QUBIT_1, n.QUBIT_1)),
    ],
    "COUPLINGS": [
        OptPltSet(c.ITERATION, param_nonlin(n.RESONATOR_1, n.QUBIT_1)),
    ],
}
