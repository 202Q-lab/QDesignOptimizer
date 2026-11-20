"""The initial-design estimates. Reference numbers were computed by hand
from the formulas (CODATA constants) and are hard-coded, so a change of a
formula or unit is noticed."""

import pytest

from qdesignoptimizer.estimation.initial_design import (
    DEFAULT_ANHARMONICITY,
    resonator_frequency_for_length,
    resonator_length_for_frequency,
    transmon_capacitance_for_anharmonicity,
    transmon_frequency,
    transmon_inductance_for_frequency,
)


def test_a_quarter_wave_resonator_on_silicon():
    # v = c / sqrt((11.45 + 1) / 2); L = v / (4 * 6 GHz)
    assert resonator_length_for_frequency(6e9) == pytest.approx(5.0065641e-3, rel=1e-7)


def test_a_half_wave_resonator_is_twice_as_long():
    assert resonator_length_for_frequency(7e9, "lambda_2") == pytest.approx(
        8.5826814e-3, rel=1e-7
    )


def test_resonator_length_and_frequency_are_inverses():
    for kind in ("lambda_4", "lambda_2"):
        length = resonator_length_for_frequency(7.3e9, kind, eps_r=10.0)
        assert resonator_frequency_for_length(length, kind, eps_r=10.0) == (
            pytest.approx(7.3e9, rel=1e-12)
        )


def test_a_200_mhz_transmon_has_about_97_ff():
    assert DEFAULT_ANHARMONICITY == 200e6
    # C = e^2 / (2 h * 200 MHz)
    assert transmon_capacitance_for_anharmonicity() == pytest.approx(
        96.851147e-15, rel=1e-7
    )


def test_a_4_ghz_transmon_needs_about_15_nh():
    # Ej = (4.2 GHz)^2 / (8 * 200 MHz) = 11.025 GHz; Lj = (Phi0 / 2 pi)^2 / (h Ej)
    lj = transmon_inductance_for_frequency(4e9)
    assert lj == pytest.approx(14.826441e-9, rel=1e-6)
    assert transmon_frequency(lj) == pytest.approx(4e9, rel=1e-12)


def test_a_larger_anharmonicity_needs_a_larger_inductance():
    # Less capacitance (larger Ec): a smaller Ej keeps the frequency
    assert transmon_inductance_for_frequency(
        5e9, 300e6
    ) > transmon_inductance_for_frequency(5e9)


@pytest.mark.parametrize(
    "call",
    [
        lambda: resonator_length_for_frequency(0),
        lambda: resonator_length_for_frequency(6e9, "lambda_8"),
        lambda: resonator_frequency_for_length(-1e-3),
        lambda: transmon_capacitance_for_anharmonicity(-200e6),
        lambda: transmon_inductance_for_frequency(4e9, 0),
        lambda: transmon_frequency(0),
    ],
)
def test_values_that_make_no_sense_are_refused(call):
    with pytest.raises(ValueError):
        call()


def test_outside_the_transmon_regime_is_refused():
    # 1 GHz with 400 MHz: Ej / Ec = 6.1, a Cooper-pair box, not a transmon
    with pytest.raises(ValueError, match="not a transmon"):
        transmon_inductance_for_frequency(1e9, 400e6)
