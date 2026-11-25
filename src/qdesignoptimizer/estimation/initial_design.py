"""First guesses for a design: how long a resonator, how large a junction
inductance and qubit capacitance for the frequencies one wants.

These are textbook estimates meant for the initial design variables, not
for the final values: they leave out the geometry (meanders, couplers,
kinetic inductance, the junction's own capacitance) that the optimizer then
corrects for. All values are in SI units (Hz, m, H, F).
"""

from __future__ import annotations

import math

from scipy.constants import c as SPEED_OF_LIGHT
from scipy.constants import e as ELEMENTARY_CHARGE
from scipy.constants import h as PLANCK
from scipy.constants import physical_constants

#: Relative permittivity of silicon at millikelvin temperatures.
EPS_R_SILICON = 11.45
#: A transmon's anharmonicity when none is given (Hz, as a magnitude).
DEFAULT_ANHARMONICITY = 200e6

_REDUCED_FLUX_QUANTUM = physical_constants["mag. flux quantum"][0] / (2 * math.pi)
_WAVELENGTHS = {"lambda_4": 0.25, "lambda_2": 0.5}


def cpw_effective_permittivity(eps_r: float = EPS_R_SILICON) -> float:
    """The effective permittivity of a coplanar waveguide on a substrate
    of ``eps_r``: half the field is in the substrate, half in vacuum."""
    return (eps_r + 1) / 2


def resonator_length_for_frequency(
    freq: float,
    resonator_type: str = "lambda_4",
    eps_r: float = EPS_R_SILICON,
) -> float:
    """The length (m) of a coplanar waveguide resonator with fundamental
    frequency ``freq`` (Hz).

    ``L = v / (4 f)`` for a quarter-wave resonator (one end shorted),
    ``L = v / (2 f)`` for a half-wave one, with ``v = c / sqrt(eps_eff)``.
    A real resonator comes out somewhat shorter: its open end, its
    coupling sections and kinetic inductance add electrical length.

    Args:
        freq: The frequency wanted (Hz).
        resonator_type: ``"lambda_4"`` or ``"lambda_2"``.
        eps_r: The substrate's relative permittivity (silicon by default).

    Raises:
        ValueError: For a frequency that is not positive or an unknown
            resonator type.
    """
    if freq <= 0:
        raise ValueError(f"the frequency must be positive, not {freq}")
    if resonator_type not in _WAVELENGTHS:
        raise ValueError(
            f"resonator_type must be one of {sorted(_WAVELENGTHS)}, "
            f"not {resonator_type!r}"
        )
    speed = SPEED_OF_LIGHT / math.sqrt(cpw_effective_permittivity(eps_r))
    return _WAVELENGTHS[resonator_type] * speed / freq


def resonator_frequency_for_length(
    length: float,
    resonator_type: str = "lambda_4",
    eps_r: float = EPS_R_SILICON,
) -> float:
    """The fundamental frequency (Hz) of a resonator of ``length`` (m):
    the inverse of ``resonator_length_for_frequency``."""
    if length <= 0:
        raise ValueError(f"the length must be positive, not {length}")
    return resonator_length_for_frequency(length, resonator_type, eps_r)


def transmon_capacitance_for_anharmonicity(
    anharmonicity: float = DEFAULT_ANHARMONICITY,
) -> float:
    """The total capacitance (F) of a transmon with ``anharmonicity`` (Hz,
    its magnitude): in the transmon regime the anharmonicity is about the
    charging energy, ``Ec = e^2 / 2C`` (Koch et al. 2007)."""
    if anharmonicity <= 0:
        raise ValueError(
            f"the anharmonicity is given as a positive magnitude, not {anharmonicity}"
        )
    return ELEMENTARY_CHARGE**2 / (2 * PLANCK * anharmonicity)


def transmon_inductance_for_frequency(
    freq: float, anharmonicity: float = DEFAULT_ANHARMONICITY
) -> float:
    """The junction inductance Lj (H) that gives a transmon the qubit
    frequency ``freq`` (Hz, the 0-1 transition) with ``anharmonicity``
    (Hz, its magnitude; 200 MHz unless given).

    From ``f01 = sqrt(8 Ej Ec) - Ec`` with ``Ec`` the anharmonicity:
    ``Ej = (f01 + Ec)^2 / (8 Ec)`` and ``Lj = (Phi0 / 2 pi)^2 / Ej``.

    Raises:
        ValueError: For values that are not positive, or that give
            ``Ej / Ec`` below 20, outside the transmon regime where these
            relations hold.
    """
    if freq <= 0:
        raise ValueError(f"the frequency must be positive, not {freq}")
    if anharmonicity <= 0:
        raise ValueError(
            f"the anharmonicity is given as a positive magnitude, not {anharmonicity}"
        )
    ej = (freq + anharmonicity) ** 2 / (8 * anharmonicity)  # Hz
    if ej / anharmonicity < 20:
        raise ValueError(
            f"Ej/Ec = {ej / anharmonicity:.1f} for {freq / 1e9:g} GHz with "
            f"{anharmonicity / 1e6:g} MHz anharmonicity: not a transmon "
            f"(Ej/Ec >= 20), the estimate does not hold"
        )
    return _REDUCED_FLUX_QUANTUM**2 / (PLANCK * ej)


def transmon_frequency(
    inductance: float, anharmonicity: float = DEFAULT_ANHARMONICITY
) -> float:
    """The qubit frequency (Hz) of a transmon with junction inductance
    ``inductance`` (H) and ``anharmonicity`` (Hz): the inverse of
    ``transmon_inductance_for_frequency``."""
    if inductance <= 0:
        raise ValueError(f"the inductance must be positive, not {inductance}")
    ej = _REDUCED_FLUX_QUANTUM**2 / (PLANCK * inductance)
    return math.sqrt(8 * ej * anharmonicity) - anharmonicity
