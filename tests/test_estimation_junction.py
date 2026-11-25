"""Junction conversions and the transmon frequency for a fabricated
junction. Reference numbers computed by hand (CODATA constants)."""

import pytest

from qdesignoptimizer.estimation.junction import (
    inductance_from_josephson_energy,
    josephson_energy_from_inductance,
    josephson_energy_from_measurement,
    josephson_energy_from_resistance,
    transmon_frequency_for_josephson_energy,
)


def test_ambegaokar_baratoff_for_aluminium():
    # Ej / h = 180 ueV / (8 e * 10 kOhm)
    assert josephson_energy_from_resistance(10e3) == pytest.approx(14.0430e9, rel=1e-4)
    # a smaller gap, a smaller Ej
    assert josephson_energy_from_resistance(10e3, 170e-6) < 14.0430e9


def test_inductance_and_energy_are_inverses():
    # Lj = 11.6 nH for Rn = 10 kOhm: the rule of thumb hbar Rn / (pi Delta)
    lj = inductance_from_josephson_energy(josephson_energy_from_resistance(10e3))
    assert lj == pytest.approx(11.638e-9, rel=1e-3)
    assert josephson_energy_from_inductance(lj) == pytest.approx(
        josephson_energy_from_resistance(10e3), rel=1e-12
    )


@pytest.mark.parametrize(
    "text, expected",
    [
        ("12.5 GHz", 12.5e9),
        ("12500MHz", 12.5e9),
        ("10 kOhm", 14.0430e9),
        ("10kΩ", 14.0430e9),
        ("11.638 nH", 14.0430e9),
    ],
)
def test_what_fabrication_reports_is_read_by_its_unit(text, expected):
    assert josephson_energy_from_measurement(text) == pytest.approx(expected, rel=1e-3)


@pytest.mark.parametrize("text", ["12.5", "12.5 V", "-3 GHz", "a lot"])
def test_a_measurement_without_a_known_unit_is_refused(text):
    with pytest.raises(ValueError):
        josephson_energy_from_measurement(text)


def test_the_designed_junction_keeps_the_simulated_frequency():
    assert transmon_frequency_for_josephson_energy(
        4.055e9, 214e6, 11.46e9, 11.46e9
    ) == pytest.approx(4.055e9, rel=1e-12)


def test_four_times_the_ej_doubles_f01_plus_ec():
    # (f + Ec) scales with sqrt(Ej)
    f = transmon_frequency_for_josephson_energy(4e9, 200e6, 10e9, 40e9)
    assert f == pytest.approx(2 * 4.2e9 - 200e6, rel=1e-12)
