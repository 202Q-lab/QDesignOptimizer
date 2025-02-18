import json
with open('design_variables.json') as in_file:
    dv = json.load(in_file)
    import design_variable_names as u
import numpy as np
import target_parameters as tp

import qdesignoptimizer.utils.constants as dc
from qdesignoptimizer.design_analysis_types import MiniStudy
from qdesignoptimizer.utils.utils_design_variables import junction_setup

CONVERGENCE = dict(nbr_passes=7, delta_f=0.03)


def get_mini_study_qb_res(branch: int):
    return MiniStudy(
        component_names=[u.name_qb(branch), u.name_res(branch), u.name_tee(branch)],
        port_list=[
            (u.name_tee(branch), "prime_end", 50),
            (u.name_tee(branch), "prime_start", 50),
        ],
        open_pins=[],
        mode_freqs=[
            (str(branch), dc.mode_freq(dc.QUBIT)),
            (str(branch), dc.mode_freq(dc.RESONATOR)),
        ],
        jj_var=dv,
        jj_setup={**junction_setup(u.name_qb(branch))},
        design_name="get_mini_study_qb_res",
        adjustment_rate=0.8,
        **CONVERGENCE
    )


def get_mini_study_2qb_resonator_coupler(branches: list, coupler: int):
    all_comps = []
    all_ports = []
    all_modes = []
    all_jjs = {}
    for branch in branches:
        all_comps.extend([u.name_qb(branch), u.name_res(branch), u.name_tee(branch)])
        all_ports.extend(
            [
                (u.name_tee(branch), "prime_end", 50),
                (u.name_tee(branch), "prime_start", 50),
            ]
        )
        all_modes.extend([(str(branch), dc.mode_freq(dc.QUBIT)), (str(branch), dc.mode_freq(dc.RESONATOR))])
        all_jjs.update(junction_setup(u.name_qb(branch)))

    all_comps.extend([u.name_res(coupler)])
    all_modes.extend([(str(coupler), dc.mode_freq(dc.RESONATOR))])

    all_mode_freq = []
    for i in range(len(all_modes)):
        all_mode_freq.append(tp.TARGET_PARAMS[all_modes[i][0]][all_modes[i][1]])
    all_modes_sorted = [all_modes[i] for i in np.argsort(all_mode_freq)]

    return MiniStudy(
        component_names=all_comps,
        port_list=all_ports,
        open_pins=[],
        mode_freqs=all_modes_sorted,
        jj_var=dv,
        jj_setup=all_jjs,
        design_name="get_mini_study_2qb_resonator_coupler",
        adjustment_rate=0.8,
        **CONVERGENCE
    )
