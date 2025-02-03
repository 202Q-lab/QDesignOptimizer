from qiskit_metal.qt.simulation.optimize.sim_plot_progress import OptPltSet
import qiskit_metal.qt.database.constants as dc

PLOT_SETTINGS = { 
    "RES": [
        OptPltSet(dc.ITERATION, dc.RES_FREQ), 
        OptPltSet(dc.ITERATION, dc.RES_KAPPA), 
        OptPltSet(dc.ITERATION, dc.RES_KERR),
        ],
    "QUBIT": [
        OptPltSet(dc.ITERATION, dc.QUBIT_FREQ), 
        OptPltSet(dc.ITERATION, dc.QUBIT_ANHARMONICITY),
        OptPltSet(dc.ITERATION, dc.QUBIT_PURCELL_DECAY),
        ],
    "COUPLINGS": [
        OptPltSet(dc.ITERATION, dc.RES_QUBIT_CHI), 
        OptPltSet(dc.ITERATION, dc.RES_QUBIT_CHI), 
        OptPltSet(dc.ITERATION, dc.RES_QUBIT_CHI), 
        ],
}
