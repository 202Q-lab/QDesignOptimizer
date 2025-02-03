from qiskit_metal.qt.simulation.optimize.sim_plot_progress import OptPltSet
import qiskit_metal.qt.database.constants as dc

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
