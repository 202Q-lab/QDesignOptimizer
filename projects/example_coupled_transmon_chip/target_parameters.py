import design_constants as dc

TARGET_PARAMS = {
    "0": {
        dc.mode_freq(dc.QUBIT): 4e9,
        dc.mode_freq(dc.RESONATOR): 7e9,
        dc.mode_kappa(dc.RESONATOR): 600e3,
    },
    "1": {
        dc.mode_freq(dc.QUBIT): 5e9,
        dc.mode_freq(dc.RESONATOR): 8e9,
        dc.mode_kappa(dc.RESONATOR): 600e3,
    },
    "2": {
        dc.mode_freq(dc.QUBIT): 5e9,
        dc.mode_freq(dc.RESONATOR): 8e9,
        dc.mode_kappa(dc.RESONATOR): 600e3,
    },
    "3": {
        dc.mode_freq(dc.QUBIT): 5e9,
        dc.mode_freq(dc.RESONATOR): 8e9,
        dc.mode_kappa(dc.RESONATOR): 600e3,
    },
    "4": {
        dc.mode_freq(dc.QUBIT): 5e9,
        dc.mode_freq(dc.RESONATOR): 8e9,
        dc.mode_kappa(dc.RESONATOR): 600e3,
    },
    "5": {
        dc.mode_freq(dc.RESONATOR): 10e9,

    },
    dc.CROSS_KERR: {
    dc.cross_kerr(["0","0"], [dc.QUBIT,dc.QUBIT]): 200e6,
    dc.cross_kerr(["0","0"], [dc.QUBIT,dc.RESONATOR]): 1e6,

    dc.cross_kerr(["1","1"], [dc.QUBIT,dc.QUBIT]): 200e6,
    dc.cross_kerr(["1","1"], [dc.QUBIT,dc.RESONATOR]): 1e6,

    dc.cross_kerr(["2","2"], [dc.QUBIT,dc.QUBIT]): 200e6,
    dc.cross_kerr(["2","2"], [dc.QUBIT,dc.RESONATOR]): 1e6,

    dc.cross_kerr(["3","3"], [dc.QUBIT,dc.QUBIT]): 200e6,
    dc.cross_kerr(["3","3"], [dc.QUBIT,dc.RESONATOR]): 1e6,
    
    dc.cross_kerr(["4","4"], [dc.QUBIT,dc.QUBIT]): 200e6,
    dc.cross_kerr(["4","4"], [dc.QUBIT,dc.RESONATOR]): 1e6,
    
    dc.cross_kerr(["5","5"], [dc.QUBIT,dc.RESONATOR]): 1e6,
    },
}
