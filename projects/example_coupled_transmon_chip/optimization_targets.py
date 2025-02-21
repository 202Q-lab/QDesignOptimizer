from typing import List

# import qdesignoptimizer.utils.constants as dc
import design_constants as dc
import design_variable_names as u
import numpy as np

from qdesignoptimizer.design_analysis_types import OptTarget
from qdesignoptimizer.utils.utils_design_variable_names import design_var_lj
from qdesignoptimizer.utils.utils_parameter_names import Mode


def get_opt_target_qubit_freq_via_lj(
    qubit: Mode,
) -> OptTarget:

    return OptTarget(
        system_target_param=dc.FREQ,
        involved_modes=[qubit],
        design_var=u.design_var_lj(qubit),
        design_var_constraint={"larger_than": "0.1nH", "smaller_than": "400nH"},
        prop_to=lambda p, v: 1
        / np.sqrt(v[design_var_lj(qubit)] * v[u.design_var_width(qubit)]),
        independent_target=False,
    )


def get_opt_target_qubit_anharmonicity_via_pad_width(
    qubit: Mode,
) -> OptTarget:

    return OptTarget(
        system_target_param=dc.NONLINEARITY,
        involved_modes=[qubit, qubit],
        design_var=u.design_var_width(qubit),
        design_var_constraint={"larger_than": "5um", "smaller_than": "1000um"},
        prop_to=lambda p, v: 1 / v[u.design_var_width(qubit)],
        independent_target=True,
    )


def get_opt_target_res_freq_via_length(
    resonator: Mode,
) -> OptTarget:

    return OptTarget(
        system_target_param=dc.FREQ,
        involved_modes=[resonator],
        design_var=u.design_var_length(resonator),
        design_var_constraint={"larger_than": "1mm", "smaller_than": "12mm"},
        prop_to=lambda p, v: 1 / v[u.design_var_length(resonator)],
        independent_target=True,
    )


def get_opt_target_res_kappa_via_coupl_length(
    resonator: Mode, couples_to: str
) -> OptTarget:

    return OptTarget(
        system_target_param=dc.KAPPA,
        involved_modes=[resonator],
        design_var=u.design_var_coupl_length(resonator, couples_to),
        design_var_constraint={"larger_than": "200um", "smaller_than": "1000um"},
        prop_to=lambda p, v: v[u.design_var_coupl_length(resonator, couples_to)] ** 2,
        independent_target=True,
    )


# def get_opt_target_res_qub_chi_via_pad_width(
#         branch: int,
#         ) -> OptTarget:

#     return OptTarget(
#         system_target_param=(str(branch), dc.RES_QUBIT_CHI),
#         involved_mode_freqs= [(str(branch), dc.RES_FREQ), (str(branch), dc.QUBIT_FREQ)],
#         design_var= u.design_var_qb_pad_width(branch),
#         design_var_constraint = {'larger_than': '5um', 'smaller_than': '1000um'} ,
#         prop_to=lambda p, v: np.abs(
#         v[u.design_var_qb_pad_width(branch)]
#         * p[f'{branch}'][dc.QUBIT_ANHARMONICITY]
#         / ( p[f'{branch}'][dc.QUBIT_FREQ]
#            - p[f'{branch}'][dc.RES_FREQ]
#            - p[f'{branch}'][dc.QUBIT_ANHARMONICITY]
#            )
#         ),
#         independent_target=False,
#     )


def get_opt_target_res_qub_chi_via_coupl_length(
    resonator: Mode, qubit: Mode
) -> OptTarget:

    return OptTarget(
        system_target_param=dc.NONLINEARITY,
        involved_modes=[resonator, qubit],
        # involved_modes=[(str(branch), dc.RESONATOR), (str(branch), dc.QUBIT)],
        design_var=u.design_var_coupl_length(resonator, qubit),
        design_var_constraint={"larger_than": "5um", "smaller_than": "350um"},
        prop_to=lambda p, v: v[u.design_var_coupl_length(resonator, qubit)],
        independent_target=True,
    )


def get_opt_targets_qb_res(
    nbr: int,
    qb_freq=True,
    qb_anharmonicity=True,
    res_freq=True,
    res_kappa=True,
    res_qub_chi=True,
) -> List[OptTarget]:
    resonator = dc.RESONATOR_1 if nbr == 1 else dc.RESONATOR_2
    qubit = dc.QUBIT_1 if nbr == 1 else dc.QUBIT_2
    opt_targets = []
    if qb_freq:
        opt_targets.append(get_opt_target_qubit_freq_via_lj(qubit))
    if qb_anharmonicity:
        opt_targets.append(get_opt_target_qubit_anharmonicity_via_pad_width(qubit))
    if res_freq:
        opt_targets.append(get_opt_target_res_freq_via_length(resonator))
    if res_kappa:
        opt_targets.append(get_opt_target_res_kappa_via_coupl_length(resonator, "tee"))
    if res_qub_chi:
        opt_targets.append(
            get_opt_target_res_qub_chi_via_coupl_length(resonator, qubit)
        )
    return opt_targets


def get_opt_targets_2qb_resonator_coupler(
    branches: list,
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
            opt_targets.append(
                get_opt_target_res_kappa_via_coupl_length(resonator, "tee")
            )
        if res_qub_chi:
            opt_targets.append(get_opt_target_res_qub_chi_via_coupl_length(branch))
    return opt_targets
