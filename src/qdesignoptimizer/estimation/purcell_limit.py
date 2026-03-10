import numpy as np


def purcell_t1_transmon_resonator(chi, kappa, alpha, f_q, f_r):
    """
    Calculates Purcell T1 limit for a transmon-resonator system in the dispersive regime.

    Valid only for a two-level system approximation of the transmon (ground and first excited
    state), as discussed in "Controlling the Spontaneous Emission of a Superconducting
    Transmon Qubit".

    The transmon has negative anharmonicity in reality; here alpha is passed as its positive
    absolute value. Using the bosonic chi convention (H = hbar*chi*a†a*b†b) and expressing
    chi in terms of the physical coupling g and detuning delta = f_q - f_r:

        chi = -2 * g^2 * alpha / (delta * (delta - alpha))
        => (g/delta)^2 = -chi * (delta - alpha) / (2 * alpha * delta)

    Then computes the Purcell decay rate and corresponding T1:

        Gamma_P [rad/s] = kappa [rad/s] * (g/delta)^2
        T1 = 1 / Gamma_P = 1 / (2*pi * kappa_Hz * (g/delta)^2)

    NOTE: The formula diverges near the straddling regime (delta ≈ alpha, i.e. f_r between
    f_q and f_q + alpha) where the dispersive approximation breaks down.

    All input frequencies must be in consistent units (Hz, i.e. cycles/s, not rad/s).

    Args:
        chi (float): Absolute dispersive shift in the bosonic convention (Hz).
        kappa (float): Resonator linewidth (Hz).
        alpha (float): Qubit anharmonicity as a absolute value (Hz); the transmon
            anharmonicity is physically negative, but is passed here as its magnitude.
        f_q (float): Qubit frequency (Hz).
        f_r (float): Resonator frequency (Hz).

    Returns:
        float: Purcell-limited T1 in seconds.

    References:
        - Koch et al., Phys. Rev. A 76, 042319 (2007) [transmon dispersive shift]
        - Blais et al., Rev. Mod. Phys. 93, 025005 (2021) [circuit QED review, Sec. V]
        - Houck et al., PRL 101, 080502 (2008) [Controlling the Spontaneous Emission of a
          Superconducting Transmon Qubit]
    """
    delta = f_q - f_r

    # Purcell weight (g/delta)^2 inverted from the bosonic-convention transmon chi formula:
    # chi = -2 * g^2 * alpha / (delta * (delta - alpha))
    # => (g/delta)^2 = -chi * (delta - alpha) / (2 * alpha * delta)
    weight = np.abs(chi * (delta - alpha) / (2 * alpha * delta))

    # Purcell decay rate in Hz (cycles/s); T1 = 1 / (2*pi * gamma_Hz)
    gamma_p_hz = kappa * weight
    t1_s = 1 / (2 * np.pi * gamma_p_hz)

    return t1_s
