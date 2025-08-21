"""Definitions of common optimization targets for qubit/resonator/coupler systems.

This module provides standard optimization target definitions used to specify how
different physical parameters (frequency, coupling strength, anharmonicity, etc.)
respond to changes in design variables during the optimization process. Each function returns an
OptTarget instance that describes how a specific physical parameter depends on one or more design
variables and other parameters. The module implements common optimization strategies for:

- Qubit frequency via Josephson inductance
- Qubit anharmonicity via capacitance width
- Resonator frequency via length
- Resonator linewidth (kappa) via coupling length
- Qubit-resonator dispersive shift (chi) via coupling length
- Combined qubit-resonator system optimization

Each optimization target specifies a proportionality relationship between
physical parameters and design variables, along with valid ranges for the
design variables.
"""

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
    design_var_constraint: dict | None = None,
) -> OptTarget:
    """
    Create an optimization target for qubit frequency via Josephson inductance.

    This function creates an optimization target that models how a qubit's frequency
    depends on its Josephson inductance (Lj) and width. The relationship follows the
    standard LC oscillator model, where frequency is inversely proportional to the
    square root of inductance times capacitance (which scales with width).

    Args:
        qubit (Mode): The qubit mode identifier.
        design_var_qubit_lj (Callable, optional): Function to generate the design
            variable name for qubit Josephson inductance. Defaults to n.design_var_lj.
        design_var_qubit_width (Callable, optional): Function to generate the design
            variable name for qubit width. Defaults to n.design_var_width.
        design_var_constraint (dict | None, optional): Constraints for the design variable.

    Notes:
        - The target uses the relationship f ∝ 1/√(L·C), where C scales with width.
        - This is marked as a dependent target since frequency depends on multiple
          design variables and physical parameters.
    """
    if design_var_constraint is None:
        design_var_constraint = {"larger_than": "0.1nH", "smaller_than": "400nH"}    
    return OptTarget(
        target_param_type=FREQ,
        involved_modes=[qubit],
        design_var=design_var_qubit_lj(qubit),
        design_var_constraint=design_var_constraint,
        prop_to=lambda p, v: 1
        / np.sqrt(v[design_var_qubit_lj(qubit)] * v[design_var_qubit_width(qubit)]),
        independent_target=False,
    )


def get_opt_target_qubit_anharmonicity_via_capacitance_width(
    qubit: Mode,
    design_var_qubit_width: Callable = n.design_var_width,
    design_var_constraint: dict | None = None,
) -> OptTarget:
    """
    Create an optimization target for qubit anharmonicity via capacitive pad width.

    This function creates an optimization target that models how a qubit's anharmonicity
    (self-Kerr nonlinearity) depends on its capacitive pad width. Larger capacitance
    (wider pad) leads to smaller anharmonicity, following the relation where
    anharmonicity is inversely proportional to capacitance.

    Args:
        qubit (Mode): The qubit mode identifier.
        design_var_qubit_width (Callable, optional): Function to generate the design
            variable name for qubit width. Defaults to n.design_var_width.
        design_var_constraint (dict | None, optional): Constraints for the design variable.

    Notes:
        - The target uses the relationship α ∝ 1/C, where C scales with width.
        - This is marked as an independent target since anharmonicity depends
          only on the capacitance width.
    """
    if design_var_constraint is None:
        design_var_constraint = {"larger_than": "5um", "smaller_than": "1000um"}
    return OptTarget(
        target_param_type=NONLIN,
        involved_modes=[qubit, qubit],
        design_var=design_var_qubit_width(qubit),
        design_var_constraint=design_var_constraint,
        prop_to=lambda p, v: 1 / v[design_var_qubit_width(qubit)],
        independent_target=True,
    )


def get_opt_target_res_freq_via_length(
    resonator: Mode,
    design_var_res_length: Callable = n.design_var_length,
    design_var_constraint: dict | None = None,
) -> OptTarget:
    """
    Create an optimization target for resonator frequency via length.

    This function creates an optimization target that models how a resonator's frequency
    depends on its physical length. Longer resonators have lower frequencies, following
    the relation where frequency is inversely proportional to length.

    Args:
        resonator (Mode): The resonator mode identifier.
        design_var_res_length (Callable, optional): Function to generate the design
            variable name for resonator length. Defaults to n.design_var_length.
        design_var_constraint (dict | None, optional): Constraints for the design variable.

    Notes:
        - The target uses the relationship f ∝ 1/L, where L is the resonator length.
        - This is marked as an independent target since frequency depends only on
          the resonator's length.
    """
    if design_var_constraint is None:
        design_var_constraint = {"larger_than": "5um", "smaller_than": "15000um"}
    return OptTarget(
        target_param_type=FREQ,
        involved_modes=[resonator],
        design_var=design_var_res_length(resonator),
        design_var_constraint=design_var_constraint,
        prop_to=lambda p, v: 1 / v[design_var_res_length(resonator)],
        independent_target=True,
    )


def get_opt_target_res_kappa_via_coupl_length(
    resonator: Mode,
    resonator_coupled_identifier: str,
    design_var_res_coupl_length: Callable = n.design_var_coupl_length,
    design_var_constraint: dict | None = None,
) -> OptTarget:
    """
    Create an optimization target for resonator linewidth via coupling length.

    This function creates an optimization target that models how a resonator's linewidth
    (kappa) depends on its coupling length to a feedline or other component. Longer
    coupling sections lead to stronger coupling and thus higher linewidth, following
    the relation where kappa is proportional to the square of coupling length.

    Args:
        resonator (Mode): The resonator mode identifier.
        resonator_coupled_identifier (str): Identifier for the element to which the
            resonator is coupled (e.g., a feedline or transmission line).
        design_var_res_coupl_length (Callable, optional): Function to generate the design
            variable name for coupling length. Defaults to n.design_var_coupl_length.
        design_var_constraint (dict | None, optional): Constraints for the design variable.

    Notes:
        - The target uses the relationship κ ∝ L², where L is the coupling length.
        - This is marked as an independent target since kappa depends only on
          the coupling length.
    """
    if design_var_constraint is None:
        design_var_constraint = {"larger_than": "20um", "smaller_than": "1000um"}
    return OptTarget(
        target_param_type=KAPPA,
        involved_modes=[resonator],
        design_var=design_var_res_coupl_length(resonator, resonator_coupled_identifier),
        design_var_constraint=design_var_constraint,
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
    design_var_constraint: dict | None = None,
) -> OptTarget:
    """
    Create an optimization target for qubit-resonator dispersive shift.

    This function creates an optimization target that models how the dispersive shift
    (chi) between a qubit and resonator depends on their coupling length and the qubit's
    width. The dispersive shift follows from circuit QED theory, where chi depends on
    the coupling strength, qubit anharmonicity, and detuning between the qubit and resonator.

    Args:
        qubit (Mode): The qubit mode identifier.
        resonator (Mode): The resonator mode identifier.
        design_var_res_qb_coupl_length (Callable, optional): Function to generate the design
            variable name for qubit-resonator coupling length. Defaults to n.design_var_coupl_length.
        design_var_qubit_width (Callable, optional): Function to generate the design
            variable name for qubit width. Defaults to n.design_var_width.
        design_var_constraint (dict | None, optional): Constraints for the design variable.

    Notes:
        - The target uses the relationship χ ∝ g²α/(Δ·(Δ-α)), where:

          - g is the coupling strength (proportional to coupling length / qubit width)
          - α is the qubit anharmonicity
          - Δ is the detuning between qubit and resonator frequencies
        - This is marked as a dependent target since chi depends on multiple
          design variables and physical parameters.
    """
    if design_var_constraint is None:
        design_var_constraint = {"larger_than": "5um", "smaller_than": "1000um"}
    return OptTarget(
        target_param_type=NONLIN,
        involved_modes=[qubit, resonator],
        design_var=design_var_res_qb_coupl_length(resonator, qubit),
        design_var_constraint=design_var_constraint,
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
    design_var_constraint: dict | None = None,
) -> OptTarget:
    """
    Create a simplified optimization target for qubit-resonator dispersive shift.

    This function creates an optimization target that uses a simplified model for the
    dispersive shift (chi) between a qubit and resonator. In this model, chi is
    considered to be directly proportional to the coupling length between the qubit
    and resonator, ignoring other dependencies to provide a more straightforward
    optimization approach.

    Args:
        qubit (Mode): The qubit mode identifier.
        resonator (Mode): The resonator mode identifier.
        design_var_res_qb_coupl_length (Callable, optional): Function to generate the design
            variable name for qubit-resonator coupling length. Defaults to n.design_var_coupl_length.
        design_var_qubit_width (Callable, optional): Function to generate the design
            variable name for qubit width. Not used in this simplified model, but
            included for API compatibility.
        design_var_constraint (dict | None, optional): Constraints for the design variable.

    Notes:
        - This simplified model uses the relationship χ ∝ coupling_length, making
          the dispersive shift directly proportional to the coupling length.
        - This is marked as a dependent target to maintain consistency with the
          full model, although it has a simpler relationship.
        - Use this simplified model when a coarse approximation is sufficient or
          when the full parameter dependencies are not critical.
    """
    if design_var_constraint is None:
        design_var_constraint = {"larger_than": "5um", "smaller_than": "1000um"}
    return OptTarget(
        target_param_type=NONLIN,
        involved_modes=[qubit, resonator],
        design_var=design_var_res_qb_coupl_length(resonator, qubit),
        design_var_constraint=design_var_constraint,
        prop_to=lambda p, v: np.sqrt(v[design_var_res_qb_coupl_length(resonator, qubit)]) ,
        # np.abs(v[design_var_res_qb_coupl_length(resonator, qubit)])
        # v[design_var_res_qb_coupl_length(resonator, qubit)]**2
        # np.log(v[design_var_res_qb_coupl_length(resonator, qubit)])
        # np.sqrt(v[design_var_res_qb_coupl_length(resonator, qubit)])
        independent_target=False,
    )


def get_opt_targets_qb_res_transmission(
    qubit: Mode,
    resonator: Mode,
    resonator_coupled_identifier: str,
    opt_target_qubit_freq=False,
    opt_target_qubit_anharm=False,
    opt_target_resonator_freq=False,
    opt_target_resonator_kappa=False,
    opt_target_resonator_qubit_chi=False,
    use_simple_resonator_qubit_chi_relation=False,
    design_var_qubit_lj: Callable[[str], str] = n.design_var_lj,
    design_var_qubit_width: Callable[[str], str] = n.design_var_width,
    design_var_res_length: Callable[[str], str] = n.design_var_length,
    design_var_res_coupl_length: Callable[[str, str], str] = n.design_var_coupl_length,
    design_var_res_qb_coupl_length: Callable[[str, str], str] = n.design_var_coupl_length,
    design_var_constraint_qubit_lj: dict | None = None,
    design_var_constraint_qubit_width:  dict | None = None,
    design_var_constraint_res_length:  dict | None = None,
    design_var_constraint_res_coupl_length:  dict | None = None,
    design_var_constraint_res_qb_coupl_length:  dict | None = None
) -> List[OptTarget]:
    """
    Create a comprehensive set of optimization targets for a qubit-resonator system.

    This function combines multiple optimization targets to create a complete optimization
    strategy for a coupled qubit-resonator system. It allows for selectively including
    or excluding specific targets based on the design requirements.

    Args:
        qubit (Mode): The qubit mode identifier.
        resonator (Mode): The resonator mode identifier.
        resonator_coupled_identifier (str): Identifier for the element to which the
            resonator is coupled (e.g., a feedline or transmission line).
        opt_target_qubit_freq (bool, optional): Whether to include qubit frequency
            optimization. Defaults to True.
        opt_target_qubit_anharm (bool, optional): Whether to include qubit anharmonicity
            optimization. Defaults to True.
        opt_target_resonator_freq (bool, optional): Whether to include resonator
            frequency optimization. Defaults to True.
        opt_target_resonator_kappa (bool, optional): Whether to include resonator
            linewidth optimization. Defaults to True.
        opt_target_resonator_qubit_chi (bool, optional): Whether to include
            qubit-resonator dispersive shift optimization. Defaults to True.
        design_var_qubit_lj (Callable, optional): Function to generate the design
            variable name for qubit Josephson inductance. Defaults to n.design_var_lj.
        design_var_qubit_width (Callable, optional): Function to generate the design
            variable name for qubit width. Defaults to n.design_var_width.
        design_var_res_length (Callable, optional): Function to generate the design
            variable name for resonator length. Defaults to n.design_var_length.
        design_var_res_coupl_length (Callable, optional): Function to generate the design
            variable name for coupling length. Defaults to n.design_var_coupl_length.
        design_var_res_qb_coupl_length (Callable, optional): Function to generate the design
            variable name for qubit-resonator coupling length. Defaults to n.design_var_coupl_length.
        design_var_constraint_qubit_lj (dict | None, optional): Constraints for the qubit
            Josephson inductance design variable. Defaults to None.
        design_var_constraint_qubit_width (dict | None, optional): Constraints for the qubit
            width design variable. Defaults to None.
        design_var_constraint_res_length (dict | None, optional): Constraints for the resonator
            length design variable. Defaults to None.
        design_var_constraint_res_coupl_length (dict | None, optional): Constraints for the resonator
            coupling length design variable. Defaults to None.
        design_var_constraint_res_qb_coupl_length (dict | None, optional): Constraints for the qubit-resonator
            coupling length design variable. Defaults to None.

    Returns:
        List[OptTarget]: A list of optimization targets for the qubit-resonator system.

    Example:
        >>> targets = get_opt_targets_qb_res_transmission(
        ...     qubit="qubit_1",
        ...     resonator="resonator_1",
        ...     resonator_coupled_identifier="feedline",
        ...     opt_target_qubit_freq=True,
        ...     opt_target_qubit_anharm=True,
        ...     opt_target_resonator_freq=True,
        ...     opt_target_resonator_kappa=True,
        ...     opt_target_resonator_qubit_chi=True,
        ... )
        >>> len(targets)  # Returns 5 if all targets are enabled
        5
    """
    opt_targets = []

    if opt_target_qubit_freq:
        opt_targets.append(
            get_opt_target_qubit_freq_via_lj(
                qubit,
                design_var_qubit_lj=design_var_qubit_lj,
                design_var_qubit_width=design_var_qubit_width,
                design_var_constraint=design_var_constraint_qubit_lj
            )
        )
    if opt_target_qubit_anharm:
        opt_targets.append(
            get_opt_target_qubit_anharmonicity_via_capacitance_width(
                qubit, design_var_qubit_width=design_var_qubit_width,
                design_var_constraint=design_var_constraint_qubit_width
            )
        )
    if opt_target_resonator_freq:
        opt_targets.append(
            get_opt_target_res_freq_via_length(
                resonator,
                design_var_res_length=design_var_res_length,
                design_var_constraint=design_var_constraint_res_length
            )
        )
    if opt_target_resonator_kappa:
        opt_targets.append(
            get_opt_target_res_kappa_via_coupl_length(
                resonator,
                resonator_coupled_identifier,
                design_var_res_coupl_length=design_var_res_coupl_length,
                design_var_constraint=design_var_constraint_res_coupl_length
            )
        )
    if opt_target_resonator_qubit_chi:
        if use_simple_resonator_qubit_chi_relation is True:
            opt_targets.append(
                get_opt_target_res_qub_chi_via_coupl_length_simple(
                    qubit,
                    resonator,
                    design_var_res_qb_coupl_length=design_var_res_qb_coupl_length,
                    design_var_constraint=design_var_constraint_res_qb_coupl_length
                )
            )
        else:
            opt_targets.append(
                get_opt_target_res_qub_chi_via_coupl_length(
                    qubit,
                    resonator,
                    design_var_res_qb_coupl_length=design_var_res_qb_coupl_length,
                    design_var_qubit_width=design_var_qubit_width,
                    design_var_constraint=design_var_constraint_res_qb_coupl_length
                )
            )
    return opt_targets
