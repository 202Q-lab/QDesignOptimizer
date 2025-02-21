from typing import Literal

# Standard mode types
RESONATOR = "resonator"
QUBIT = "qubit"
CAVITY = "cavity"
COUPLER = "coupler"

# Parameter types
FREQ = "freq"
KAPPA = "kappa"
NONLIN = "nonlin"


Mode = str
""""Mode name on the format group_modetype_nbr where group and nbr are optional.

Examples:
    qubit
    qubit_1
    1_qubit_3
    gr1_qubit_3
"""
Parameter = str
""""Paramter name on the format group_modetype_nbr_paramtype where group and nbr are optional.

Examples:
    qubit_freq
    gr1_qubit_3_freq
"""


def mode(
    mode_type: str,
    nbr: int | str | None = None,
    group: int | None = None,
) -> Mode:
    """Construct a mode name from the mode type, group, and number."""
    assert "_" not in mode_type, "mode_type cannot contain underscores"
    assert (
        "_to_" not in mode_type
    ), "mode_type cannot contain the string '_to_', since it is a keyword for non-linear parameters"

    assert nbr is None or "_" not in str(nbr), "nbr cannot contain underscores"
    assert (
        isinstance(nbr, int) or "_to_" not in nbr
    ), "nbr cannot contain the string '_to_', since it is a keyword for non-linear parameters"

    assert group is None or isinstance(group, int), "group must be an integer or None"

    mode_name = mode_type
    if group is not None:
        mode_name = f"gr{group}_{mode_name}"
    if nbr is not None:
        mode_name = f"{mode_name}_{nbr}"

    assert (
        ":" not in mode_name
    ), "Qiskit parsing does not allow mode_name to contain the character ':'"
    assert (
        "-" not in mode_name
    ), "Qiskit parsing does not allow mode_name to contain the character '-'"
    return mode_name


def get_group_from_mode(mode: Mode) -> int | None:
    if "gr" in mode:
        return int(mode.split("_")[0][2:])
    else:
        return None


def param(mode: Mode, param_type: Literal["freq", "kappa"]) -> Parameter:
    """Construct a parameter name from the mode and parameter type.

    Examples:
        param("qubit_1", "freq") -> "qubit_1_freq"
        param("qubit_1", "kappa") -> "qubit_1_kappa"
    """
    assert param_type in ["freq", "kappa"], "param_type must be 'freq' or 'kappa'"
    return f"{mode}_{param_type}"


def param_nonlin(mode_1: Mode, mode_2: Mode) -> Parameter:
    """Construct a non-linear parameter name from two modes.
    The modes are sorted alphabetically before constructing the parameter name.
    If mode_1 == mode_2, the parameter name is the anharmonicity/self-Kerr of the mode.

    Examples:
        param_nonlin("qubit_1", "qubit_2") -> "qubit_1_to_qubit_2_nonlin"
        param_nonlin("qubit_2", "qubit_1") -> "qubit_1_to_qubit_2_nonlin"
    """
    modes = [mode_1, mode_2]
    modes.sort()
    return f"{modes[0]}_to_{modes[1]}_{'nonlin'}"


def param_capacitance(capacitance_name_1: str, capacitance_name_2: str) -> Parameter:
    """Construct a parameter name for capacitance matrix elements from two capacitance names.
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


def get_modes_from_param_nonlin(param: Parameter) -> tuple[Mode, Mode]:
    assert param.endswith("_nonlin"), "param must end with '_nonlin'"
    return tuple(param.split("_nonlin")[0].split("_to_")[:2])


def get_paramtype_from_param(param: Parameter) -> Literal["freq", "kappa", "nonlin"]:
    return param.split("_")[-1]
