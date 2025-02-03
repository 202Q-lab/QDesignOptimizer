from typing import List
from qiskit_metal.qt.simulation.design_analysis_types import OptTarget, TargetType
from qiskit_metal.qt.utils.utils_design_variables import design_var_lj
import design_variables as dv
import qiskit_metal.qt.database.constants as dc
import numpy as np

def get_opt_target_qubit_freq_via_lj(
        branch: int,
        ) -> OptTarget:

    return OptTarget(
        system_target_param=(str(branch), dc.QUBIT_FREQ),
        involved_mode_freqs= [(str(branch), dc.QUBIT_FREQ)],
        design_var= design_var_lj(dv.name_qb(branch)),
        design_var_constraint = {'larger_than': '0.1nH', 'smaller_than': '400nH'},
        prop_to =lambda p, v: 1 / np.sqrt( v[design_var_lj(dv.name_qb(branch))] * v[dv.design_var_qb_pad_width(branch)] ),
        independent_target=False,
    )

def get_opt_target_qubit_anharmonicity_via_pad_width(
        branch: int,
        ) -> OptTarget:

    return OptTarget(
        system_target_param=(str(branch), dc.QUBIT_ANHARMONICITY),
        involved_mode_freqs= [(str(branch), dc.QUBIT_FREQ)],
        design_var= dv.design_var_qb_pad_width(branch),
        design_var_constraint = {'larger_than': '5um', 'smaller_than': '1000um'},
        prop_to =lambda p, v: 1 / v[dv.design_var_qb_pad_width(branch)],
        independent_target=True,
    )

def get_opt_target_res_freq_via_length(
        branch: int,
        ) -> OptTarget:

    return OptTarget(
        system_target_param=(str(branch), dc.RES_FREQ),
        involved_mode_freqs= [(str(branch), dc.RES_FREQ)],
        design_var= dv.design_var_res_length(branch),
        design_var_constraint = {'larger_than': '1mm', 'smaller_than': '12mm'} ,
        prop_to=lambda p, v: 1 / v[dv.design_var_res_length(branch)],
        independent_target=True,
    )

def get_opt_target_res_kappa_via_coupl_length(
        branch: int,
        ) -> OptTarget:

    return OptTarget(
        system_target_param=(str(branch), dc.RES_KAPPA),
        involved_mode_freqs= [(str(branch), dc.RES_FREQ)],
        design_var= dv.design_var_res_coupl_length(branch),
        design_var_constraint = {'larger_than': '200um', 'smaller_than': '1000um'} ,
        prop_to=lambda p, v: v[dv.design_var_res_coupl_length(branch)]**2,
        independent_target=True,
    )

# def get_opt_target_res_qub_chi_via_pad_width(
#         branch: int,
#         ) -> OptTarget:

#     return OptTarget(
#         system_target_param=(str(branch), dc.RES_QUBIT_CHI),
#         involved_mode_freqs= [(str(branch), dc.RES_FREQ), (str(branch), dc.QUBIT_FREQ)],
#         design_var= dv.design_var_qb_pad_width(branch),
#         design_var_constraint = {'larger_than': '5um', 'smaller_than': '1000um'} ,
#         prop_to=lambda p, v: np.abs(
#         v[dv.design_var_qb_pad_width(branch)] 
#         * p[f'{branch}'][dc.QUBIT_ANHARMONICITY] 
#         / ( p[f'{branch}'][dc.QUBIT_FREQ] 
#            - p[f'{branch}'][dc.RES_FREQ] 
#            - p[f'{branch}'][dc.QUBIT_ANHARMONICITY] 
#            )
#         ),
#         independent_target=False,
#     )

def get_opt_target_res_qub_chi_via_res_qub_coupl_length(
        branch: int,
        ) -> OptTarget:

    return OptTarget(
        system_target_param=(str(branch), dc.RES_QUBIT_CHI),
        involved_mode_freqs= [(str(branch), dc.RES_FREQ), (str(branch), dc.QUBIT_FREQ)],
        design_var= dv.design_var_res_qb_coupl_length(branch),
        design_var_constraint = {'larger_than': '5um', 'smaller_than': '1000um'} ,
        prop_to=lambda p, v: v[dv.design_var_res_qb_coupl_length(branch)] ,
        independent_target=True,
    )

def get_opt_targets_qb_res(
        branch:int,
        qb_freq=True, 
        qb_anharmonicity=True,
        res_freq=True,
        res_kappa=True,
        res_qub_chi=True,

    ) -> List[OptTarget]:
    opt_targets = []
    if qb_freq:
        opt_targets.append(get_opt_target_qubit_freq_via_lj(branch))
    if qb_anharmonicity:
        opt_targets.append(get_opt_target_qubit_anharmonicity_via_pad_width(branch))
    if res_freq:
        opt_targets.append(get_opt_target_res_freq_via_length(branch))
    if res_kappa:
        opt_targets.append(get_opt_target_res_kappa_via_coupl_length(branch))
    if res_qub_chi:
        opt_targets.append(get_opt_target_res_qub_chi_via_res_qub_coupl_length(branch))
    return opt_targets

def get_opt_targets_2qb_resonator_coupler(
        branches:list,
        qb_freq=True, 
        qb_anharmonicity=True,
        res_freq=True,
        res_kappa=True,
        res_qub_chi=True,

    ) -> List[OptTarget]:
    opt_targets = []
    for branch in branches:
        if qb_freq:
            opt_targets.append(get_opt_target_qubit_freq_via_lj(branch))
        if qb_anharmonicity:
            opt_targets.append(get_opt_target_qubit_anharmonicity_via_pad_width(branch))
        if res_freq:
            opt_targets.append(get_opt_target_res_freq_via_length(branch))
        if res_kappa:
            opt_targets.append(get_opt_target_res_kappa_via_coupl_length(branch))
        if res_qub_chi:
            opt_targets.append(get_opt_target_res_qub_chi_via_res_qub_coupl_length(branch))
    return opt_targets