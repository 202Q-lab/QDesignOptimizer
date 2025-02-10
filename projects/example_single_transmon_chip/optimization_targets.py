from typing import List

import json
with open('design_variables.json') as in_file:
    dv = json.load(in_file)
import numpy as np

import design_constants as dc
import design_variable_names as u
from qdesignoptimizer.design_analysis_types import OptTarget


def get_opt_target_qubit_freq_via_lj(
    branch: int,
) -> OptTarget:

    return OptTarget(
        system_target_param=(str(branch), dc.QUBIT_FREQ),
        involved_mode_freqs=[(str(branch), dc.QUBIT_FREQ)],
        design_var=u.design_var_lj(u.name_qb(branch)),
        design_var_constraint={"larger_than": "0.1nH", "smaller_than": "400nH"},
        prop_to=lambda p, v: 1
        / np.sqrt(
            v[u.design_var_lj(u.name_qb(branch))]
            * v[u.design_var_qb_pad_width(branch)]
        ),
        independent_target=True,
    )


def get_opt_target_res_freq_via_length(
    branch: int,
) -> OptTarget:

    return OptTarget(
        system_target_param=(str(branch), dc.RES_FREQ),
        involved_mode_freqs=[(str(branch), dc.RES_FREQ)],
        design_var=u.design_var_res_length(branch),
        design_var_constraint={"larger_than": "1mm", "smaller_than": "12mm"},
        prop_to=lambda p, v: 1 / v[u.design_var_res_length(branch)],
        independent_target=False,
    )


def get_opt_target_res_kappa_via_coupl_length(
    branch: int,
) -> OptTarget:

    return OptTarget(
        system_target_param=(str(branch), dc.RES_KAPPA),
        involved_mode_freqs=[(str(branch), dc.RES_FREQ)],
        design_var=u.design_var_res_coupl_length(branch),
        design_var_constraint={"larger_than": "1um", "smaller_than": "12000um"},
        prop_to=lambda p, v: v[u.design_var_res_coupl_length(branch)] ** 2,
        independent_target=False,
    )


def get_opt_targets_qb_res(
    branch: int,
    qb_freq=True,
    res_freq=True,
    res_kappa=True,
) -> List[OptTarget]:
    opt_targets = []
    if qb_freq:
        opt_targets.append(get_opt_target_qubit_freq_via_lj(branch))
    if res_freq:
        opt_targets.append(get_opt_target_res_freq_via_length(branch))
    if res_kappa:
        opt_targets.append(get_opt_target_res_kappa_via_coupl_length(branch))
    return opt_targets
