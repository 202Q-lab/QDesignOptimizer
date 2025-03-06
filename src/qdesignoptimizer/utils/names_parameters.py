from typing import Literal

# Standard mode types
RESONATOR = "resonator"
QUBIT = "qubit"
CAVITY = "cavity"
COUPLER = "coupler"

# Parameter types
FREQ: Literal["freq"] = "freq"
KAPPA: Literal["kappa"] = "kappa"
CHARGE_LINE_LIMITED_T1: Literal["charge_line_limited_t1"] = "charge_line_limited_t1"
NONLIN: Literal["nonlinearity"] = "nonlinearity"

CAPACITANCE = "capacitance"
""" dict: Maps branch to capacitance matrix elements in capacitance matrix simulation.
    Capacitance matrix elements are in femto Farads (fF).

    Format: (capacitance_name, capacitance_name): value

    Example: {
        ('comb_NAME_QB1', 'comb_NAME_QB1'): 100,
        ('comb_NAME_QB1', 'comb_NAME_QB2'): 5,
        }
"""

ITERATION = "ITERATION"


Mode = str
""""Mode name on the format  modetype_identifier, the complete Mode name must be unique.

Examples:
    qubit
    QUBIT_1
    qubit_xy3
"""
Parameter = str
""""Paramter name which is a unique mode name concatenated with a parameter type.

Examples:
    qubit_freq
    qubit_3_freq
"""


def mode(
    mode_type: str,
    identifier: int | str | None = None,
) -> Mode:
    """Construct a mode name from the mode type and identifier. Note that the complete mode name must be unique."""
    assert "_" not in mode_type, "mode_type cannot contain underscores"
    assert (
        "_to_" not in mode_type
    ), "mode_type cannot contain the string '_to_', since it is a keyword for non-linear parameters"

    assert identifier is None or "_" not in str(
        identifier
    ), "identifier cannot contain underscores"

    mode_name = mode_type
    if identifier is not None:
        assert (
            isinstance(identifier, int) or "_to_" not in identifier
        ), "identifier cannot contain the string '_to_', since it is a keyword for non-linear parameters"

        mode_name = f"{mode_name}_{identifier}"

    assert (
        ":" not in mode_name
    ), "Qiskit parsing does not allow mode_name to contain the character ':'"
    assert (
        "-" not in mode_name
    ), "Qiskit parsing does not allow mode_name to contain the character '-'"
    return mode_name


def param(
    mode: Mode,
    param_type: Literal["freq", "kappa", "charge_line_limited_t1", "capacitance"],
) -> Parameter:
    """Construct a parameter name from the mode and parameter type.

    Examples:
        param("QUBIT_1", "freq") -> "qubit_1_freq"
        param("QUBIT_1", "kappa") -> "qubit_1_kappa"
    """
    assert param_type in [
        "freq",
        "kappa",
        "charge_line_limited_t1",
    ], "param_type must be 'freq' or 'kappa' or 'charge_line_limited_t1"
    return f"{mode}_{param_type}"


def param_nonlin(mode_1: Mode, mode_2: Mode) -> Parameter:
    """Construct a non-linear parameter name from two modes.
    The modes are sorted alphabetically before constructing the parameter name.
    If mode_1 == mode_2, the parameter name is the anharmonicity/self-Kerr of the mode.

    Examples:
        param_nonlin("QUBIT_1", "QUBIT_2") -> "qubit_1_to_qubit_2_nonlin"
        param_nonlin("QUBIT_2", "QUBIT_1") -> "qubit_1_to_qubit_2_nonlin"
    """
    modes = [mode_1, mode_2]
    modes.sort()
    return f"{modes[0]}_to_{modes[1]}_{'nonlin'}"


def param_capacitance(capacitance_name_1: str, capacitance_name_2: str) -> Parameter:
    """Construct a parameter name for capacitance matrix elements (femto Farad) from two capacitance names.
    The capacitance names are sorted alphabetically before constructing the parameter name.

    Examples:
        param_capacitance("capacitance_name_1", "capacitance_name_2") -> "capacitance_name_1_to_capacitance_name_2_capacitance"
        param_capacitance("capacitance_name_2", "capacitance_name_1") -> "capacitance_name_1_to_capacitance_name_2_capacitance"
    """
    capacitance_names = [capacitance_name_1, capacitance_name_2]
    capacitance_names.sort()
    return f"{capacitance_names[0]}_to_{capacitance_names[1]}_capacitance"


def get_mode_from_param(param: Parameter) -> Mode:
    return "_".join(param.split("_")[:-1])


def get_modes_from_param_nonlin(param: Parameter) -> tuple[Mode, ...]:
    assert param.endswith("_nonlin"), "param must end with '_nonlin'"
    return tuple(param.split("_nonlin")[0].split("_to_")[:2])
