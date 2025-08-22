from typing import Optional

import names as n
import numpy as np
import parameter_targets as pt

from qdesignoptimizer.design_analysis_types import MiniStudy, SurfaceProperties
from qdesignoptimizer.sim_capacitance_matrix import (
    CapacitanceMatrixStudy,
    ModeDecayIntoChargeLineStudy,
    ResonatorDecayIntoWaveguideStudy,
)
from qdesignoptimizer.utils.names_design_variables import junction_setup
from qdesignoptimizer.utils.names_parameters import FREQ, param

CONVERGENCE = dict(nbr_passes=4, delta_f=0.03)


def get_mini_study_2qb_resonator_coupler():
    all_comps = []
    all_ports = []
    all_modes = []
    all_jjs = {}
    for group in [n.NBR_1, n.NBR_2]:
        qubit = [n.QUBIT_1, n.QUBIT_2][group - 1]
        resonator = [n.RESONATOR_1, n.RESONATOR_2][group - 1]
        all_comps.extend(
            [
                n.name_mode(qubit),
                n.name_mode(resonator),
                n.name_tee(group),
            ]
        )
        all_ports.extend(
            [
                # (n.name_tee(group), "prime_end", 50),
                # (n.name_tee(group), "prime_start", 50),
            ]
        )
        all_modes.extend([qubit, resonator])
        all_jjs.update(junction_setup(qubit))
    all_jjs.update(junction_setup(n.COUPLER_12))

    all_comps.append(n.name_mode(n.COUPLER_12))
    all_modes.append(n.COUPLER_12)

    all_mode_freq = []
    for i in range(len(all_modes)):
        all_mode_freq.append(pt.PARAM_TARGETS[param(all_modes[i], FREQ)])
    all_modes_sorted = [all_modes[i] for i in np.argsort(all_mode_freq)]

    return MiniStudy(
        qiskit_component_names=all_comps,
        port_list=all_ports,
        open_pins=[],
        modes=all_modes_sorted,
        jj_setup=all_jjs,
        design_name="get_mini_study_2qb_resonator_coupler",
        adjustment_rate=1,
        cos_trunc=6,
        fock_trunc=5,
        build_fine_mesh=False,
        **CONVERGENCE,
    )
