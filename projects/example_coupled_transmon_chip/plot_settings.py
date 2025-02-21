import design_constants as dc

from qdesignoptimizer.utils.sim_plot_progress import OptPltSet
from qdesignoptimizer.utils.utils_parameter_names import param, param_nonlin

PLOT_SETTINGS = {
    "RES": [
        OptPltSet(dc.ITERATION, param(dc.RESONATOR_1, dc.FREQ)),
        OptPltSet(dc.ITERATION, param(dc.RESONATOR_1, dc.KAPPA)),
        OptPltSet(dc.ITERATION, param_nonlin(dc.RESONATOR_1, dc.RESONATOR_1)),
    ],
    "QUBIT": [
        OptPltSet(dc.ITERATION, param(dc.QUBIT_1, dc.FREQ)),
        OptPltSet(dc.ITERATION, param_nonlin(dc.QUBIT_1, dc.QUBIT_1)),
    ],
    "COUPLINGS": [
        OptPltSet(dc.ITERATION, param_nonlin(dc.RESONATOR_1, dc.QUBIT_1)),
    ],
}
