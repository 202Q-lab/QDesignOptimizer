from src.utils.sim_plot_progress import OptPltSet
import src.utils.constants as dc

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
