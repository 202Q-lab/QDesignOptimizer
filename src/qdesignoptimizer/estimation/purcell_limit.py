import numpy as np

def calculate_purcell_t1(chi, kappa, alpha, f_q, f_r):
    """
    Calculates Purcell T1 limit for a transmon-resonator system in the dispersive regime.

    Derives the effective coupling (g/delta)^2 from the transmon dispersive shift chi:

        chi = g^2 * alpha / (delta * (delta + alpha))
        => (g/delta)^2 = chi * (delta + alpha) / (alpha * delta)

    Then computes the Purcell decay rate and corresponding T1:

        Gamma_P [rad/s] = kappa [rad/s] * (g/delta)^2
        T1 = 1 / Gamma_P = 1 / (2*pi * kappa_Hz * (g/delta)^2)

    All input frequencies must be in consistent units (Hz, i.e. cycles/s, not rad/s).

    Args:
        chi (float): Dispersive shift (Hz).
        kappa (float): Resonator linewidth (Hz).
        alpha (float): Qubit anharmonicity (Hz); negative for a transmon.
        f_q (float): Qubit frequency (Hz).
        f_r (float): Resonator frequency (Hz).

    Returns:
        float: Purcell-limited T1 in seconds.

    References:
        - Koch et al., Phys. Rev. A 76, 042319 (2007) [transmon dispersive shift]
        - Blais et al., Rev. Mod. Phys. 93, 025005 (2021) [circuit QED review, Sec. V]
    """
    delta = f_q - f_r

    # Purcell weight (g/delta)^2 inverted from the transmon chi formula
    weight = np.abs((chi / alpha) * ((delta + alpha) / delta))

    # Purcell decay rate in Hz (cycles/s); T1 = 1 / (2*pi * gamma_Hz)
    gamma_p_hz = kappa * weight
    t1_s = 1 / (2 * np.pi * gamma_p_hz)

    return t1_s