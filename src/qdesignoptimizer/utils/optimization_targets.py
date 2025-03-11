"""Factory functions for creating common optimization targets in quantum circuit design."""

from typing import Callable, List

import numpy as np

import qdesignoptimizer.utils.names_design_variables as n
from qdesignoptimizer.design_analysis_types import OptTarget
from qdesignoptimizer.utils.names_parameters import (
    FREQ,
    KAPPA,
    NONLIN,
    Mode,
    param,
    param_nonlin,
)


def get_opt_target_qubit_freq_via_lj(
    qubit: Mode,
    design_var_qubit_lj: Callable = n.design_var_lj,
    design_var_qubit_width: Callable = n.design_var_width,
) -> OptTarget:
    """Get target for qubit frequency vs Josephson"""
    return OptTarget(
        target_param_type=FREQ,
        involved_modes=[qubit],
        design_var=design_var_qubit_lj(qubit),
        design_var_constraint={"larger_than": "0.1nH", "smaller_than": "400nH"},
        prop_to=lambda p, v: 1
        / np.sqrt(v[design_var_qubit_lj(qubit)] * v[design_var_qubit_width(qubit)]),
        independent_target=False,
    )


def get_opt_target_qubit_anharmonicity_via_capacitance_width(
    qubit: Mode,
    design_var_qubit_width: Callable = n.design_var_width,
) -> OptTarget:
    """Get target for qubit anharmonicity vs its capacitive pad(s) size."""
    return OptTarget(
        target_param_type=NONLIN,
        involved_modes=[qubit, qubit],
        design_var=design_var_qubit_width(qubit),
        design_var_constraint={"larger_than": "5um", "smaller_than": "1000um"},
        prop_to=lambda p, v: 1 / v[design_var_qubit_width(qubit)],
        independent_target=True,
    )


def get_opt_target_res_freq_via_length(
    resonator: Mode,
    design_var_res_length: Callable = n.design_var_length,
) -> OptTarget:
    """Get target for resonator frequency vs its length."""
    return OptTarget(
        target_param_type=FREQ,
        involved_modes=[resonator],
        design_var=design_var_res_length(resonator),
        design_var_constraint={"larger_than": "500um", "smaller_than": "15000um"},
        prop_to=lambda p, v: 1 / v[design_var_res_length(resonator)],
        independent_target=True,
    )


def get_opt_target_res_kappa_via_coupl_length(
    resonator: Mode,
    resonator_coupled_identifier: str,
    design_var_res_coupl_length: Callable = n.design_var_coupl_length,
) -> OptTarget:
    """Get target for resonator's kappa vs. length of the coupler to a feedline."""
    return OptTarget(
        target_param_type=KAPPA,
        involved_modes=[resonator],
        design_var=design_var_res_coupl_length(resonator, resonator_coupled_identifier),
        design_var_constraint={"larger_than": "20um", "smaller_than": "2000um"},
        prop_to=lambda p, v: v[
            design_var_res_coupl_length(resonator, resonator_coupled_identifier)
        ]
        ** 2,
        independent_target=True,
    )


def get_opt_target_res_qub_chi_via_coupl_length(
    qubit: Mode,
    resonator: Mode,
    design_var_res_qb_coupl_length: Callable = n.design_var_coupl_length,
    design_var_qubit_width: Callable = n.design_var_width,
) -> OptTarget:
    """Get optimization target for qubit-resonator dispersive shift."""
    return OptTarget(
        target_param_type=NONLIN,
        involved_modes=[qubit, resonator],
        design_var=design_var_res_qb_coupl_length(resonator, qubit),
        design_var_constraint={"larger_than": "5um", "smaller_than": "1000um"},
        prop_to=lambda p, v: np.abs(
            v[design_var_res_qb_coupl_length(resonator, qubit)]
            / v[design_var_qubit_width(qubit)]
            * p[param_nonlin(qubit, qubit)]
            / (
                p[param(qubit, FREQ)]
                - p[param(resonator, FREQ)]
                - p[param_nonlin(qubit, qubit)]
            )
        ),
        independent_target=False,
    )


def get_opt_target_res_qub_chi_via_coupl_length_simple(
    qubit: Mode,
    resonator: Mode,
    design_var_res_qb_coupl_length: Callable = n.design_var_coupl_length,
    design_var_qubit_width: Callable = n.design_var_width,
) -> OptTarget:
    """Get optimization target for qubit-resonator dispersive shift with simplified formula."""
    return OptTarget(
        target_param_type=NONLIN,
        involved_modes=[qubit, resonator],
        design_var=design_var_res_qb_coupl_length(resonator, qubit),
        design_var_constraint={"larger_than": "5um", "smaller_than": "1000um"},
        prop_to=lambda p, v: np.abs(
            v[design_var_res_qb_coupl_length(resonator, qubit)]
            / v[design_var_qubit_width(qubit)]
        ),
        independent_target=False,
    )


def get_opt_targets_qb_res_transmission(
    qubit: Mode,
    resonator: Mode,
    resonator_coupled_identifier: str,
    opt_target_qubit_freq=True,
    opt_target_qubit_anharm=True,
    opt_target_resonator_freq=True,
    opt_target_resonator_kappa=True,
    opt_target_resonator_qubit_chi=True,
    use_simple_resonator_qubit_chi=True,
    design_var_qubit_lj: Callable[[str], str] = n.design_var_lj,
    design_var_qubit_width: Callable[[str], str] = n.design_var_width,
    design_var_res_length: Callable[[str], str] = n.design_var_length,
    design_var_res_coupl_length: Callable[[str, str], str] = n.design_var_coupl_length,
) -> List[OptTarget]:
    """Get the optimization targets for a qubit-resonator system.

    Args:
        qubit (Mode): The qubit mode.
        resonator (Mode): The resonator mode.
        resonator_coupled_identifier (str): The identifier of the resonator coupled to the qubit.
        opt_target_qubit_freq (bool, optional): Whether to optimize the qubit frequency.
                                                Defaults to True.
        opt_target_qubit_anharm (bool, optional): Whether to optimize the qubit anharmonicity.
                                                  Defaults to True.
        opt_target_resonator_freq (bool, optional): Whether to optimize the resonator frequency.
                                                    Defaults to True.
        opt_target_resonator_kappa (bool, optional): Whether to optimize the resonator linewidth.
                                                     Defaults to True.
        opt_target_resonator_qubit_chi (bool, optional): Whether to optimize the qubit-resonator coupling strength.
                                                         Defaults to True.
        design_var_qubit_lj (Callable, optional): The function to get the qubit inductance.
                                                  Defaults to n.design_var_lj.
        design_var_qubit_width (Callable, optional): The function to get the qubit width.
                                                     Defaults to n.design_var_width.
        design_var_res_length (Callable, optional): The function to get the resonator length.
                                                    Defaults to n.design_var_length.
        design_var_res_coupl_length (Callable, optional): The function to get the resonator coupling length.
                                                          Defaults to n.design_var_coupl_length.

    Returns:
        List[OptTarget]: The optimization targets.
    """
    opt_targets = []

    if opt_target_qubit_freq:
        opt_targets.append(
            get_opt_target_qubit_freq_via_lj(
                qubit,
                design_var_qubit_lj=design_var_qubit_lj,
                design_var_qubit_width=design_var_qubit_width,
            )
        )
    if opt_target_qubit_anharm:
        opt_targets.append(
            get_opt_target_qubit_anharmonicity_via_capacitance_width(
                qubit, design_var_qubit_width=design_var_qubit_width
            )
        )
    if opt_target_resonator_freq:
        opt_targets.append(
            get_opt_target_res_freq_via_length(
                resonator,
                design_var_res_length=design_var_res_length,
            )
        )
    if opt_target_resonator_kappa:
        opt_targets.append(
            get_opt_target_res_kappa_via_coupl_length(
                resonator,
                resonator_coupled_identifier,
                design_var_res_coupl_length=design_var_res_coupl_length,
            )
        )
    if opt_target_resonator_qubit_chi:
        if use_simple_resonator_qubit_chi is True:
            opt_targets.append(
                get_opt_target_res_qub_chi_via_coupl_length_simple(
                    qubit,
                    resonator,
                    design_var_res_qb_coupl_length=design_var_res_coupl_length,
                    design_var_qubit_width=design_var_qubit_width,
                )
            )
        else:
            opt_targets.append(
                get_opt_target_res_qub_chi_via_coupl_length(
                    qubit,
                    resonator,
                    design_var_res_qb_coupl_length=design_var_res_coupl_length,
                    design_var_qubit_width=design_var_qubit_width,
                )
            )
    return opt_targets
