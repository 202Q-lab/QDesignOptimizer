"""A Josephson junction's energy from what fabrication measures, and the
transmon frequency that follows.

A junction is designed as an inductance Lj; fabrication gives its normal
resistance Rn (measured at room temperature on a probe station) or, once
cooled, its Josephson energy Ej. These convert between the three and
re-estimate a simulated transmon's frequency for the junction it actually
got. All values are in SI units (H, Ohm, Hz for energies E/h).
"""

from __future__ import annotations

import math
import re

from scipy.constants import e as ELEMENTARY_CHARGE
from scipy.constants import h as PLANCK
from scipy.constants import physical_constants

#: The superconducting gap of thin-film aluminium (eV).
ALUMINIUM_GAP_EV = 180e-6

_REDUCED_FLUX_QUANTUM = physical_constants["mag. flux quantum"][0] / (2 * math.pi)


def josephson_energy_from_inductance(inductance: float) -> float:
    """Ej (Hz) of a junction of inductance ``inductance`` (H):
    ``Ej = (Phi0 / 2 pi)^2 / Lj``."""
    _positive(inductance, "inductance")
    return _REDUCED_FLUX_QUANTUM**2 / inductance / PLANCK


def inductance_from_josephson_energy(ej: float) -> float:
    """Lj (H) of a junction with Josephson energy ``ej`` (Hz)."""
    _positive(ej, "Josephson energy")
    return _REDUCED_FLUX_QUANTUM**2 / (ej * PLANCK)


def josephson_energy_from_resistance(
    resistance: float, gap_ev: float = ALUMINIUM_GAP_EV
) -> float:
    """Ej (Hz) of a junction with normal resistance ``resistance`` (Ohm),
    by Ambegaokar-Baratoff at zero temperature: ``Ic = pi Delta / (2 e
    Rn)``, ``Ej = hbar Ic / 2e``, so ``Ej / h = Delta / (8 e Rn)`` with
    Delta in eV.

    A resistance measured at room temperature is typically some 10 %
    below the one at millikelvin; correct it before, if known.
    """
    _positive(resistance, "resistance")
    _positive(gap_ev, "gap")
    return gap_ev / (8 * ELEMENTARY_CHARGE * resistance)


def transmon_frequency_for_josephson_energy(
    simulated_freq: float,
    anharmonicity: float,
    designed_ej: float,
    fabricated_ej: float,
) -> float:
    """A simulated transmon's frequency (Hz) with the junction it got.

    The simulation found ``simulated_freq`` and ``anharmonicity`` (Hz, its
    magnitude, about the charging energy Ec) with a junction of
    ``designed_ej``. With ``f01 + Ec = sqrt(8 Ej Ec)`` and Ec unchanged
    (the capacitances are as drawn): ``f01' = (f01 + Ec) sqrt(Ej' / Ej) -
    Ec``. Scaling the simulated value keeps what the simulation knew
    beyond the formula (the resonator's pull, the participation).
    """
    for value, name in (
        (simulated_freq, "simulated frequency"),
        (anharmonicity, "anharmonicity"),
        (designed_ej, "designed Josephson energy"),
        (fabricated_ej, "fabricated Josephson energy"),
    ):
        _positive(value, name)
    return (simulated_freq + anharmonicity) * math.sqrt(
        fabricated_ej / designed_ej
    ) - anharmonicity


def squid_frequency_range(
    max_freq: float, anharmonicity: float, asymmetry: float
) -> tuple[float, float]:
    """The frequencies (Hz) a SQUID transmon can be tuned over.

    At zero flux the SQUID's Ej is its junctions' sum and the transmon is
    at ``max_freq``; at half a flux quantum Ej drops to ``asymmetry``
    (``d = |Ej1 - Ej2| / (Ej1 + Ej2)``) times that. The low end is the
    same rescaling as ``transmon_frequency_for_josephson_energy``, never
    below 0 (a symmetric SQUID tunes all the way down; the transmon
    formula itself no longer holds once Ej is near Ec there, so the low
    end is an estimate).
    """
    _positive(max_freq, "highest frequency")
    _positive(anharmonicity, "anharmonicity")
    if not 0 <= asymmetry <= 1:
        raise ValueError(f"a SQUID's asymmetry is from 0 to 1, not {asymmetry}")
    low = (max_freq + anharmonicity) * math.sqrt(asymmetry) - anharmonicity
    return max(low, 0.0), max_freq


_MEASURED = re.compile(r"^\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*(\S*)\s*$")
_ENERGY = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}
_RESISTANCE = {"ohm": 1.0, "kohm": 1e3, "mohm": 1e6, "ω": 1.0, "kω": 1e3}
_INDUCTANCE = {"h": 1.0, "nh": 1e-9, "ph": 1e-12, "uh": 1e-6}


def josephson_energy_from_measurement(
    text: str | float, gap_ev: float = ALUMINIUM_GAP_EV
) -> float:
    """Ej (Hz) from what fabrication reports, by its unit: an energy
    (``"12.3 GHz"``), a normal resistance (``"8.1 kOhm"``, see
    ``josephson_energy_from_resistance``) or an inductance (``"14 nH"``).

    Raises:
        ValueError: For a value without one of these units, or not positive.
    """
    match = _MEASURED.match(str(text))
    if not match:
        raise ValueError(f"{text!r} is not a number with a unit")
    number, unit = float(match.group(1)), match.group(2).lower()
    if unit in _ENERGY:
        ej = number * _ENERGY[unit]
        _positive(ej, "Josephson energy")
        return ej
    if unit in _RESISTANCE:
        return josephson_energy_from_resistance(number * _RESISTANCE[unit], gap_ev)
    if unit in _INDUCTANCE:
        return josephson_energy_from_inductance(number * _INDUCTANCE[unit])
    raise ValueError(
        f"{text!r}: give Ej in GHz, Rn in kOhm or Lj in nH (the unit says which)"
    )


def _positive(value: float, name: str) -> None:
    if not value > 0:
        raise ValueError(f"the {name} must be positive, not {value}")
