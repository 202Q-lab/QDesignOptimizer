"""Reference values for the analytical decay models.

The expected numbers were computed once by hand from the formulas in the
docstrings (SI units) and are hard-coded so that a change of the library's
formula or unit conversion is detected.
"""

import pandas as pd
import pytest

from qdesignoptimizer.estimation.classical_model_decay_into_charge_line import (
    calculate_t1_limit_floating_lumped_mode_decay_into_chargeline,
    calculate_t1_limit_grounded_lumped_mode_decay_into_chargeline,
    mode_decay_rate_into_transmissionline,
)
from qdesignoptimizer.sim_capacitance_matrix import ResonatorDecayIntoWaveguideStudy


def test_mode_decay_rate_into_transmission_line():
    # gamma = (2 pi 4 GHz)^2 * 80 fF * (0.1 fF / 80 fF)^2 * 50 Ohm
    gamma = mode_decay_rate_into_transmissionline(
        mode_freq_GHz=4.0,
        mode_capacitance_fF=80.0,
        coupling_capacitance_fF=0.1,
        impedance_env=50,
    )
    assert gamma == pytest.approx(3947.8417604357433, rel=1e-12)


def test_t1_limit_grounded_mode():
    t1 = calculate_t1_limit_grounded_lumped_mode_decay_into_chargeline(
        mode_freq_GHz=4.0,
        mode_capacitance_fF=80.0,
        mode_capacitance_to_charge_line_fF=0.1,
        charge_line_impedance=50.0,
    )
    assert t1 == pytest.approx(2.5330295910584445e-04, rel=1e-12)


def test_t1_limit_floating_mode():
    # C_a = 60 + 0.5, C_b = 50 + 0.2, C_sum = C_a C_b / (C_a + C_b) + 5,
    # C_c = |C_a 0.2 - C_b 0.5| / (C_a + C_b)
    t1 = calculate_t1_limit_floating_lumped_mode_decay_into_chargeline(
        mode_freq_GHz=4.0,
        cap_island_a_island_b_fF=5.0,
        cap_island_a_ground_fF=60.0,
        cap_island_a_line_fF=0.5,
        cap_island_b_ground_fF=50.0,
        cap_island_b_line_fF=0.2,
        charge_line_impedance=50.0,
    )
    assert t1 == pytest.approx(7.446946247757012e-05, rel=1e-12)


@pytest.mark.parametrize(
    "resonator_type, expected_kappa",
    [("lambda_2", 2693.9157004532476), ("lambda_4", 5387.831400906495)],
)
def test_resonator_kappa_estimate(resonator_type, expected_kappa):
    # kappa = Z0^2 omega^3 C^2 / pi / (2 pi), doubled for a quarter-wave resonator
    study = ResonatorDecayIntoWaveguideStudy(
        mode="resonator_1",
        mode_freq_GHz=7.0,
        resonator_name="res",
        waveguide_name="feed",
        impedance_ohm=50,
        qiskit_component_names=[],
        resonator_type=resonator_type,
    )
    study.capacitance_matrix_fF = pd.DataFrame(
        [[60.0, -0.5], [-0.5, 70.0]], index=["res", "feed"], columns=["res", "feed"]
    )
    assert study.get_decay_parameter_value() == pytest.approx(expected_kappa, rel=1e-12)
    assert study.get_decay_parameter_type() == "kappa"
