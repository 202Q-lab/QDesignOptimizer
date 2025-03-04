import numpy as np


def mode_decay_rate_into_transmissionline(
    mode_freq_GHz: float,
    mode_capacitance_fF: float,
    coupling_capacitance_fF: float,
    impedance_env: float = 50,
):
    """Calculate the decay of a grounded mode into a charge line from a classical model. For a single decay channel.
    Reference: https://www.research-collection.ethz.ch/handle/20.500.11850/155858
    Appendix B Equation B.15

    Args:
        mode_freq_GHz (float): (GHz) eigenfrequency of the oscillator or qubit transistion frequency
        mode_capacitance_fF (float): (fF) The total capacitance of the mode.
        coupling_capacitance_fF (float): (fF) The capacitance between the mode and the charge line.
        impedance_env (float): (Ohm) The impedance of the charge line.

    Returns:
        float: (Hz) Decay rate to charge line.
    """
    omega = 2 * np.pi * mode_freq_GHz
    beta_i = coupling_capacitance_fF / mode_capacitance_fF
    unit_conversion = 1e3  # GHz_to_Hz^2 * fF_to_F, GHz_to_Hz = 1e9, fF_to_F = 1e-15
    gamma = omega**2 * mode_capacitance_fF * beta_i**2 * impedance_env * unit_conversion

    return gamma


def calculate_t1_limit_grounded_lumped_mode_decay_into_chargeline(
    mode_freq_GHz: float,
    mode_capacitance_fF: float,
    mode_capacitance_to_charge_line_fF: float,
    charge_line_impedance: float = 50.0,
):
    """Wrapper around mode_decay_rate_into_transmissionline for the case of a grounded mode.
    In this case the coupling capacitance is just the capacitance between the single island and the charge line.

    Args:
        mode_freq_GHz (float): (GHz) The frequency of the mode omega/2pi.
        mode_capacitance_fF (float): (fF) The total capacitance of the mode. The mode should be grounded such that its capacitance is constituted by a single capacitive pad.
        mode_capacitance_to_charge_line_fF (float): (fF) The capacitance between the mode and the charge line.
        charge_line_impedance (float): (Ohm) The impedance of the charge line.

    Returns:
        float: (s) The T1 limit due to decay into charge line decay.
    """

    gamma = mode_decay_rate_into_transmissionline(
        mode_freq_GHz,
        mode_capacitance_fF,
        mode_capacitance_to_charge_line_fF,
        charge_line_impedance,
    )

    return 1 / gamma


def calculate_t1_limit_floating_lumped_mode_decay_into_chargeline(
    mode_freq_GHz: float,
    cap_island_a_island_b_fF: float,
    cap_island_a_ground_fF: float,
    cap_island_a_line_fF: float,
    cap_island_b_ground_fF: float,
    cap_island_b_line_fF: float,
    charge_line_impedance: float = 50.0,
):
    """Wrapper around mode_decay_rate_into_transmissionline for the case of a floating mode.
    In this case the coupling capacitance is an effective capacitance including different capacitance networks, which we need to compute here first.

    Args:
        mode_freq_GHz (float): (GHz) The frequency of the mode omega/2pi.
        cap_island_a_island_b_fF (float): (fF) The capacitance across the junction from island B to island A.
        cap_island_a_ground_fF (float): (fF) The capacitance of island A to ground.
        cap_island_b_ground_fF (float): (fF) The capacitance of island B to ground.
        cap_island_a_line_fF (float): (fF) The coupling capacitance of the charge line to island A.
        cap_island_b_line_fF (float): (fF) The coupling capacitance of the charge line to island B.
        charge_line_impedance (float): (Ohm) The impedance of the charge line.

    Returns:
        float: (s) The T1 limit due to decay into charge line decay.
    """

    cap_a = cap_island_a_ground_fF + cap_island_a_line_fF
    cap_b = cap_island_b_ground_fF + cap_island_b_line_fF
    Csum = (cap_a * cap_b) / (cap_a + cap_b) + cap_island_a_island_b_fF
    coupling_capacitance = np.abs(
        (cap_a * cap_island_b_line_fF - cap_b * cap_island_a_line_fF) / (cap_a + cap_b)
    )

    gamma = mode_decay_rate_into_transmissionline(
        mode_freq_GHz, Csum, coupling_capacitance, charge_line_impedance
    )

    return 1 / gamma
