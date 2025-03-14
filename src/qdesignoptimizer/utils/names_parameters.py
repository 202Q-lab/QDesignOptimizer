"""Standard naming conventions for physical parameters and modes in quantum circuit designs."""

from typing import Literal, Tuple, Union

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
"""Special parameter identifier used for tracking optimization iterations."""


Mode = str
"""Type representing a unique mode name in the format 'modetype_identifier'.

    The mode name must be unique within a design and follows a standardized format
    of lowercase mode type, optionally followed by an underscore and an identifier.

    Examples:
        >>> "qubit"       # A single qubit without an identifier
        >>> "qubit_1"     # First qubit in a multi-qubit system
        >>> "resonator_a" # A resonator with string identifier 'a'
"""

Parameter = str
"""Type representing a parameter name, formed by concatenating a unique mode name with a parameter type.

    The parameter name uniquely identifies a physical quantity associated with
    one or more modes in the system.

    Examples:
        >>> "qubit_freq"      # Frequency of a qubit
        >>> "qubit_1_kappa"   # Decay rate of qubit 1
        >>> "qubit_1_to_resonator_1_nonlin"  # Cross-Kerr interaction
"""


def mode(
    mode_type: str,
    identifier: Union[int, str, None] = None,
) -> Mode:
    """
    Construct a standardized mode name from a mode type and optional identifier.

    Creates a unique mode name by combining the mode type with an optional identifier. The resulting
    name follows the convention 'modetype_identifier' and must be unique within a design.

    Args:
        mode_type (str): The type of mode (e.g., "qubit", "resonator").
            Must not contain underscores or the string "_to_".
        identifier (int, str, None): Optional identifier for the mode.
            Must not contain underscores or the string "_to_". If None, only the mode_type is used.

    Returns:
        Mode: A standardized mode name in the format 'modetype_identifier'.

    Raises:
        AssertionError: If mode_type or identifier contains forbidden characters.

    Examples:
        >>> mode("qubit", 1)
        'qubit_1'
        >>> mode("resonator", "a")
        'resonator_a'
        >>> mode("cavity")
        'cavity'
    """
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
    mode_instance: Mode,
    param_type: Literal["freq", "kappa", "charge_line_limited_t1", "capacitance"],
) -> Parameter:
    """
    Construct a parameter name from a mode and parameter type.

    Creates a parameter name by concatenating the mode name with the parameter type.
    Used for parameters associated with a single mode, such as frequency or decay rate.

    Args:
        mode_instance (Mode): The mode name (e.g., "qubit_1").
        param_type (Literal): The type of parameter, must be one of:
            - "freq": Frequency
            - "kappa": Decay rate/linewidth
            - "charge_line_limited_t1": T1 limited by charge line

    Returns:
        Parameter: A parameter name in the format 'mode_paramtype'.

    Raises:
        AssertionError: If param_type is not one of the allowed values.

    Examples:
        >>> param("qubit_1", "freq")
        'qubit_1_freq'
        >>> param("resonator_a", "kappa")
        'resonator_a_kappa'
    """
    assert param_type in [
        "freq",
        "kappa",
        "charge_line_limited_t1",
        "capacitance",
    ], "param_type must be 'freq', 'kappa', 'charge_line_limited_t1', or 'capacitance'"
    return f"{mode_instance}_{param_type}"


def param_nonlin(mode_1: Mode, mode_2: Mode) -> Parameter:
    """
    Construct a non-linear parameter name from two modes.

    Creates a parameter name for nonlinear interactions between two modes,
    such as cross-Kerr coupling or anharmonicity (self-Kerr). The modes are
    sorted alphabetically to ensure consistency regardless of argument order.

    Args:
        mode_1 (Mode): First mode name.
        mode_2 (Mode): Second mode name.

    Returns:
        Parameter: A nonlinearity parameter name in the format 'mode1_to_mode2_nonlin'.

    Note:
        If mode_1 == mode_2, the returned parameter represents the anharmonicity
        (self-Kerr) of the mode.

    Examples:
        >>> param_nonlin("qubit_1", "qubit_2")
        'qubit_1_to_qubit_2_nonlin'
        >>> param_nonlin("qubit_2", "qubit_1")  # Order is normalized
        'qubit_1_to_qubit_2_nonlin'
        >>> param_nonlin("qubit_1", "qubit_1")  # Anharmonicity
        'qubit_1_to_qubit_1_nonlin'
    """
    modes = [mode_1, mode_2]
    modes.sort()
    return f"{modes[0]}_to_{modes[1]}_{'nonlin'}"


def param_capacitance(capacitance_name_1: str, capacitance_name_2: str) -> Parameter:
    """
    Construct a parameter name for capacitance matrix elements.

    Creates a parameter name for capacitance between two islands or components
    in the system. The capacitance names are sorted alphabetically to ensure
    consistency regardless of argument order.

    Args:
        capacitance_name_1 (str): First capacitance island/component name.
        capacitance_name_2 (str): Second capacitance island/component name.

    Returns:
        Parameter: A capacitance parameter name in the format
            'name1_to_name2_capacitance', with values measured in femto Farads (fF).

    Examples:
        >>> param_capacitance("island_1", "island_2")
        'island_1_to_island_2_capacitance'
        >>> param_capacitance("island_2", "island_1")  # Order is normalized
        'island_1_to_island_2_capacitance'
    """
    capacitance_names = [capacitance_name_1, capacitance_name_2]
    capacitance_names.sort()
    return f"{capacitance_names[0]}_to_{capacitance_names[1]}_capacitance"


def get_mode_from_param(parameter: Parameter) -> Mode:
    """
    Extract the mode identifier from a parameter name.

    Parses a parameter name to retrieve the original mode name by removing
    the parameter type suffix.

    Args:
        parameter (Parameter): The parameter name to parse.

    Returns:
        Mode: The extracted mode name.

    Examples:
        >>> get_mode_from_param("qubit_1_freq")
        'qubit_1'
        >>> get_mode_from_param("resonator_a_kappa")
        'resonator_a'
    """
    return "_".join(parameter.split("_")[:-1])


def get_modes_from_param_nonlin(parameter: Parameter) -> Tuple[Mode, ...]:
    """
    Extract mode identifiers from a non-linear parameter name.

    Parses a nonlinearity parameter name to retrieve the original mode names
    that are involved in the interaction.

    Args:
        parameter (Parameter): The nonlinearity parameter name to parse.
            Must end with "_nonlin".

    Returns:
        Tuple[Mode, ...]: A tuple containing the extracted mode names.

    Raises:
        AssertionError: If parameter does not end with "_nonlin".

    Examples:
        >>> get_modes_from_param_nonlin("qubit_1_to_qubit_2_nonlin")
        ('qubit_1', 'qubit_2')
        >>> get_modes_from_param_nonlin("qubit_1_to_qubit_1_nonlin")
        ('qubit_1', 'qubit_1')
    """
    assert parameter.endswith("_nonlin"), "parameter must end with '_nonlin'"
    return tuple(parameter.split("_nonlin")[0].split("_to_")[:2])
