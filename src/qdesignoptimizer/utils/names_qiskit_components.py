"""Naming utilities for mapping between modes and Qiskit Metal component names."""

from typing import Union

from qdesignoptimizer.utils.names_parameters import Mode

QiskitComponentName = str
r""" A string representing the name of a Qiskit Metal component, by convention starts with 'name\_'

    Examples:
        name_qubit
        name_resonator_1
        name_coupler_1to2
"""


def name_id(identifier: Union[str, int]) -> QiskitComponentName:
    """Create a component name from a generic identifier."""
    return f"name_{identifier}"


def name_mode(identifier: Mode) -> QiskitComponentName:
    """Create a component name from a mode identifier."""
    return f"name_{identifier}"


def name_mode_to_mode(identifier_1: Mode, identifier_2: Mode) -> QiskitComponentName:
    """Create a component name representing a connection between two modes."""
    return f"name_{identifier_1}_to_{identifier_2}"


def name_tee(identifier: Union[str, int]) -> QiskitComponentName:
    """Create a component name for a tee junction."""
    return f"name_tee{identifier}"


def name_lp(identifier: Union[str, int]) -> QiskitComponentName:
    """Create a component name for a launch pad."""
    return f"name_lp{identifier}"


def name_charge_line(identifier: Union[str, int]) -> QiskitComponentName:
    """Create a component name for a charge line."""
    return f"name_charge_line{identifier}"


def name_flux_line(identifier: Union[str, int]) -> QiskitComponentName:
    """Create a component name for a flux line."""
    return f"name_flux_line_{identifier}"


def name_lp_to_tee(
    lp_identifier: Union[str, int], tee_identifier: Union[str, int]
) -> QiskitComponentName:
    """Create a component name for a connection between a launch pad and tee junction."""
    return f"name_lp{lp_identifier}_to_tee{tee_identifier}"


def name_tee_to_tee(
    tee_identifier1: Union[str, int], tee_identifier2: Union[str, int]
) -> QiskitComponentName:
    """Create a component name for a connection between two tee junctions."""
    return f"name_tee{tee_identifier1}_to_tee{tee_identifier2}"


def name_lp_to_chargeline(
    lp_identifier: Union[str, int], chargeline_identifier: Union[str, int]
) -> QiskitComponentName:
    """Create a component name for a connection between a launch pad and charge line."""
    return f"name_lp{lp_identifier}_to_chargeline{chargeline_identifier}"
