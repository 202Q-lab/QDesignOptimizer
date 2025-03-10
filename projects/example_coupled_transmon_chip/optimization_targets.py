from typing import List

import names as n
import numpy as np

from qdesignoptimizer.design_analysis_types import OptTarget
from qdesignoptimizer.utils.optimization_targets import (
    get_opt_target_res_kappa_via_coupl_length,
    get_opt_targets_qb_res_transmission,
)


def get_opt_target_res_kappa_feedline(group: int) -> list[OptTarget]:
    resonator = [n.RESONATOR_1, n.RESONATOR_2][group - 1]
    target = get_opt_target_res_kappa_via_coupl_length(
        resonator=resonator, resonator_coupled_identifier="tee"
    )
    # target.involved_modes=[]
    return [target]


def get_opt_targets_2qubits_resonator_coupler(
    groups: List[int],
    opt_target_qubit_freq=False,
    opt_target_qubit_anharm=False,
    opt_target_resonator_freq=False,
    opt_target_resonator_kappa=False,
    opt_target_resonator_qubit_chi=False,
    opt_target_coupler_freq=False,
) -> List[OptTarget]:
    """Get the optimization targets for a 2 qubit-resonator system with a coupler.

    Args:
        groups (List[int]): The qubit-resonator pair numbers.
        opt_target_qubit_freq (bool, optional): Whether to add an optimization target for the qubit frequency.
        opt_target_qubit_anharm (bool, optional): Whether to add an optimization target for the qubit anharmonicity.
        opt_target_resonator_freq (bool, optional): Whether to add an optimization target for the resonator frequency.
        opt_target_resonator_kappa (bool, optional): Whether to add an optimization target for the resonator kappa.
        opt_target_resonator_qubit_chi (bool, optional): Whether to add an optimization target for the resonator-qubit chi.
        opt_target_coupler_freq (bool, optional): Whether to add an optimization target for the coupler frequency.
    """
    opt_targets = []

    # Example of how to add an optimization target.
    # IMPORTANT: the design variable name used MUST be specified in the design_variables.json and
    # should be used in the design.py to adjust a component's geometry.
    if opt_target_coupler_freq:
        opt_target_coupler = OptTarget(
            target_param_type=n.FREQ,
            involved_modes=[n.COUPLER_12],
            design_var=n.design_var_length(n.COUPLER_12),
            design_var_constraint={"larger_than": "50um", "smaller_than": "10000um"},
            prop_to=lambda p, v: 1 / v[n.design_var_length(n.COUPLER_12)],
            independent_target=True,
        )
        opt_targets.append(opt_target_coupler)

    # Add OptTargets for each specified qubit-resonator pair using a convenience function.
    for group in groups:
        qubit = [n.QUBIT_1, n.QUBIT_2][group - 1]
        resonator = [n.RESONATOR_1, n.RESONATOR_2][group - 1]

        opt_targets.extend(
            get_opt_targets_qb_res_transmission(
                qubit,
                resonator,
                resonator_coupled_identifier="tee",
                opt_target_qubit_freq=opt_target_qubit_freq,
                opt_target_qubit_anharm=opt_target_qubit_anharm,
                opt_target_resonator_freq=opt_target_resonator_freq,
                opt_target_resonator_kappa=opt_target_resonator_kappa,
                opt_target_resonator_qubit_chi=opt_target_resonator_qubit_chi,
            )
        )

    return opt_targets


def get_opt_target_qubit_T1_limit_via_charge_posx(
    group: int,
) -> OptTarget:
    qubit = [n.QUBIT_1, n.QUBIT_2][group - 1]
    return OptTarget(
        target_param_type=n.PURCELL_LIMIT_T1,
        involved_modes=[qubit],
        design_var=n.design_var_cl_pos_x(qubit),
        design_var_constraint={"larger_than": "-4000um", "smaller_than": "-2400um"},
        prop_to=lambda p, v: v[n.design_var_cl_pos_x(qubit)] ** 2,
        independent_target=True,
    )


def get_opt_targets_qb_charge_line(
    group: int, qb_T1_limit: bool = True
) -> List[OptTarget]:
    opt_targets = []
    if qb_T1_limit:
        opt_targets.append(get_opt_target_qubit_T1_limit_via_charge_posx(group))
    return opt_targets


def get_opt_target_capacitance(
    group: int,
) -> List[OptTarget]:
    resonator = [n.RESONATOR_1, n.RESONATOR_2][group - 1]
    return [
        OptTarget(
            target_param_type=n.CAPACITANCE,
            involved_modes=["prime_cpw_name_tee1", "second_cpw_name_tee1"],
            design_var=n.design_var_length(f"{resonator}_capacitance"),
            design_var_constraint={"larger_than": "1um", "smaller_than": "500um"},
            prop_to=lambda p, v: 1
            / np.sqrt(v[n.design_var_length(f"{resonator}_capacitance")]),
            independent_target=True,
        )
    ]
