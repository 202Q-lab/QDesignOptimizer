"""The optimizer against a readout resonator with its own physics.

The other integration tests answer with :class:`FakePhysics`, which follows the
targets' own ``prop_to`` relations, so they check the plumbing and the loop but
not whether a relation is right. Here the faked HFSS answers from a model of a
quarter-wave CPW resonator capacitively side-coupled to a feedline, written
from the textbook formulas and nothing in ``qdesignoptimizer``:

* frequency ``f = v / (4 (L + dL))`` with ``v = c / sqrt(eps_eff)`` and an
  open-end correction ``dL``, so ``f`` is close to, but not exactly, ``1 / L``;
* linewidth from the coupling capacitance ``Cc = Cc0 + c_m * Lc`` of the
  coupling section: near resonance the resonator is a parallel LC with
  ``C_r = pi / (4 w Z_r)``, ``Cc`` in series with the feedline (``Z0 / 2``, it
  leaks both ways) loads it with ``G = w^2 Cc^2 Z0 / 2``, so
  ``kappa = G / C_r = (2 / pi) w^3 Cc^2 Z0 Z_r`` (rad/s). ``kappa`` therefore
  also depends on the resonator length, which the target ignores.

The library's targets (``f ~ 1 / L``, ``kappa ~ Lc^2``) only approximate this,
so the optimizer needs a few iterations; with a relation of the wrong sign it
must not get there, which shows the test can tell a wrong relation from a right
one.
"""

from __future__ import annotations

import math

import pytest

from qdesignoptimizer.design_analysis_types import MiniStudy, OptTarget
from qdesignoptimizer.utils.names_parameters import FREQ, KAPPA, param
from qdesignoptimizer.utils.optimization_targets import (
    get_opt_target_res_freq_via_length,
    get_opt_target_res_kappa_via_coupl_length,
)

from .conftest import COUPLED_TRANSMON
from .example_harness import prepare_example, relative_errors_to_targets
from .fake_ansys import EIGENMODE_FREQ_FACTOR

SPEED_OF_LIGHT = 299_792_458.0
EPS_EFF = (11.45 + 1) / 2  # CPW on silicon
OPEN_END_UM = 60.0  # fringing at the open end, as extra length
CC0_FF = 1.0  # coupling capacitance of the section's ends
CM_FF_PER_UM = 0.025  # mutual capacitance per length of the coupled section
Z0_OHM = 50.0  # feedline
ZR_OHM = 50.0  # resonator line

ITERATIONS = 8


def _um(value: str) -> float:
    number, unit = float(value[:-2]), value[-2:]
    assert unit == "um", value
    return number


class QuarterWaveResonatorPhysics:
    """A resonator side-coupled to a feedline; the interface of ``FakePhysics``."""

    def __init__(self, resonator: str, length_var: str, coupling_var: str):
        self.resonator = resonator
        self.length_var = length_var
        self.coupling_var = coupling_var
        self.targets: dict = {}  # no capacitance studies here

    def freq(self, mode: str, design_variables: dict) -> float:
        assert mode == self.resonator, mode
        length_m = (_um(design_variables[self.length_var]) + OPEN_END_UM) * 1e-6
        return SPEED_OF_LIGHT / math.sqrt(EPS_EFF) / (4 * length_m)

    def kappa(self, mode: str, design_variables: dict) -> float:
        """Linewidth kappa / 2 pi in Hz."""
        omega = 2 * math.pi * self.freq(mode, design_variables)
        coupling_um = _um(design_variables[self.coupling_var])
        cc = (CC0_FF + CM_FF_PER_UM * coupling_um) * 1e-15
        kappa_rad = 2 / math.pi * omega**3 * cc**2 * Z0_OHM * ZR_OHM
        return kappa_rad / (2 * math.pi)

    def nonlin(self, mode_i, mode_j, idx_i, idx_j, design_variables) -> float:
        raise AssertionError("a linear resonator has no junction to analyse")

    def value(self, name, design_variables):
        raise AssertionError(f"{name} is not simulated here")


def _optimize(tmp_path, monkeypatch, targets_for):
    """Run the resonator optimization; returns the analysis."""
    # The example's modules (names, design, ...) are dropped again at
    # teardown by prepare_example's monkeypatch.
    session = prepare_example(COUPLED_TRANSMON, tmp_path, monkeypatch)
    import design as d
    import names as n
    import parameter_targets as pt

    from qdesignoptimizer.design_analysis import (
        DesignAnalysis,
        DesignAnalysisState,
    )
    from qdesignoptimizer.utils.chip_generation import create_chip_base

    resonator = n.RESONATOR_1
    length_var = n.design_var_length(resonator)
    coupling_var = n.design_var_coupl_length(resonator, "tee")
    session.physics = QuarterWaveResonatorPhysics(resonator, length_var, coupling_var)

    design, gui = create_chip_base(n.CHIP_NAME, d.chip_type, open_gui=True)
    d.render_qiskit_metal_design(design, gui)
    mini_study = MiniStudy(
        qiskit_component_names=[n.name_mode(resonator), n.name_tee(n.NBR_1)],
        port_list=[
            (n.name_tee(n.NBR_1), "prime_end", 50),
            (n.name_tee(n.NBR_1), "prime_start", 50),
        ],
        open_pins=[],
        modes=[resonator],
        jj_setup={},
        design_name="readout_resonator_physics",
        nbr_passes=7,
        delta_f=0.03,
    )
    analysis = DesignAnalysis(
        DesignAnalysisState(
            design,
            lambda design, **kwargs: d.render_qiskit_metal_design(
                design, gui, **kwargs
            ),
            pt.PARAM_TARGETS,
        ),
        mini_study=mini_study,
        opt_targets=targets_for(resonator, length_var, coupling_var),
        save_path=None,
        update_design_variables=False,
    )
    for _ in range(ITERATIONS):
        analysis.optimize_target({}, {})
    return analysis, session


def _library_targets(resonator, length_var, coupling_var):
    return [
        get_opt_target_res_freq_via_length(resonator),
        get_opt_target_res_kappa_via_coupl_length(resonator, "tee"),
    ]


def test_the_resonator_reaches_its_frequency_and_linewidth(tmp_path, monkeypatch):
    analysis, session = _optimize(tmp_path, monkeypatch, _library_targets)
    resonator = session.physics.resonator
    wanted_f = analysis.system_target_params[param(resonator, FREQ)]
    wanted_kappa = analysis.system_target_params[param(resonator, KAPPA)]

    # The first design is far off: this is not a test that starts at the answer.
    first = analysis.optimization_results[0]["system_optimized_params"]
    assert abs(first[param(resonator, FREQ)] / wanted_f - 1) > 0.2

    errors = relative_errors_to_targets(analysis)
    assert max(errors.values()) < 1e-3, errors

    # Checked against the model itself too, not only against what the
    # optimizer read back: the final design really has these values.
    final = analysis.optimization_results[-1]["design_variables"]
    true_f = session.physics.freq(resonator, final)
    true_kappa = session.physics.kappa(resonator, final)
    # HFSS's eigenmode frequency is reported slightly high, see fake_ansys.
    assert true_f == pytest.approx(wanted_f / EIGENMODE_FREQ_FACTOR, rel=1e-3)
    assert true_kappa == pytest.approx(wanted_kappa, rel=2e-2)
    # ... and is physically sensible: a few mm long, a few hundred um coupled.
    assert 3000 < _um(final[session.physics.length_var]) < 7000
    assert 100 < _um(final[session.physics.coupling_var]) < 600


def _flipped(target: OptTarget, prop_to) -> OptTarget:
    return OptTarget(
        target_param_type=target.target_param_type,
        involved_modes=target.involved_modes,
        design_var=target.design_var,
        design_var_constraint=target.design_var_constraint,
        prop_to=prop_to,
        independent_target=True,
    )


@pytest.mark.parametrize("wrong", [FREQ, KAPPA])
def test_a_relation_of_the_wrong_sign_does_not_converge(wrong, tmp_path, monkeypatch):
    def targets(resonator, length_var, coupling_var):
        freq, kappa = _library_targets(resonator, length_var, coupling_var)
        if wrong == FREQ:  # f ~ L: longer is higher
            return [_flipped(freq, lambda p, v: v[length_var]), kappa]
        # kappa ~ 1 / Lc^2: more coupling, less leakage
        return [freq, _flipped(kappa, lambda p, v: v[coupling_var] ** -2)]

    analysis, session = _optimize(tmp_path, monkeypatch, targets)
    errors = relative_errors_to_targets(analysis)
    assert errors[f"{wrong} {[session.physics.resonator]}"] > 0.1, errors
