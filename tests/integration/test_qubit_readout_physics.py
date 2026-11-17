"""A full optimization of a transmon and its readout resonator, against
their own physics.

The tutorial's qubit-resonator study (``get_mini_study_qb_res``, with its
junction, and the five targets of ``main_eigenmode_single_qubit_resonator``:
qubit frequency and anharmonicity, resonator frequency and linewidth, and the
dispersive shift) is run against a model written from the textbook formulas
and nothing in ``qdesignoptimizer``:

* transmon: ``C = C_pads(width) + Cg``, ``Ec = e^2 / 2C``, ``Ej = (Phi0 / 2pi)^2
  / Lj``, ``f01 = sqrt(8 Ej Ec) - Ec``, anharmonicity ``Ec``;
* resonator: the quarter-wave model of ``test_readout_resonator_physics``;
* coupling: ``Cg`` grows with the coupling length, ``g = Cg / (2 sqrt(C C_r))
  sqrt(wq wr)`` with ``C_r = pi / (4 wr Z_r)``; the dispersive shift (the
  cross-Kerr) is ``2 g^2 a / (D (D - a))`` (Koch et al. 2007, ``a`` the
  anharmonicity, ``D = fq - fr``), and both modes are pushed apart by
  ``g^2 / D``.

Every quantity depends on several design variables (``Cg`` loads the qubit,
the resonator's length moves ``g``, ``D`` and ``kappa``), while each target
follows one simple relation, so the optimizer has to iterate its way there,
as against HFSS.
"""

from __future__ import annotations

import json
import math
import re

import pytest

from qdesignoptimizer.utils.names_parameters import (
    FREQ,
    KAPPA,
    NONLIN,
    param,
    param_nonlin,
)

from .conftest import COUPLED_TRANSMON
from .example_harness import prepare_example, relative_errors_to_targets
from .test_readout_resonator_physics import ZR_OHM, QuarterWaveResonatorPhysics

E = 1.602176634e-19
H = 6.62607015e-34
PHI0_REDUCED = 2.067833848e-15 / (2 * math.pi)

C_PADS0_FF = 20.0  # the qubit's pads: fixed part ...
C_PADS_FF_PER_UM = 0.15  # ... and per um of pad width
CG0_FF = 0.5  # qubit-resonator coupling: the claw's ends ...
CG_FF_PER_UM = 0.05  # ... and per um of coupling length

ITERATIONS = 10
# The tutorial's resonator (7500 um) is below its qubit in this model; the
# optimizer assigns eigenmodes by target frequency, so the two may not
# cross. Shortened to start 12 % below its target, the qubit 22 % above.
START = {"design_var_length_resonator_1": "5600um"}

_SI = {"um": 1e-6, "mm": 1e-3, "nH": 1e-9, "pH": 1e-12, "fF": 1e-15, "": 1.0}


def _si(value) -> float:
    match = re.fullmatch(r"\s*([-+0-9.eE]+)\s*([a-zA-Z]*)\s*", str(value))
    assert match and match.group(2) in _SI, value
    return float(match.group(1)) * _SI[match.group(2)]


class TransmonReadoutPhysics:
    """A transmon coupled to a quarter-wave resonator on a feedline."""

    def __init__(self, qubit: str, resonator: str, design_var: dict):
        self.qubit, self.resonator = qubit, resonator
        self.var = design_var  # {(param type, modes): design variable}
        self.bare_resonator = QuarterWaveResonatorPhysics(
            resonator,
            design_var[(FREQ, (resonator,))],
            design_var[(KAPPA, (resonator,))],
        )
        self.targets: dict = {}  # no capacitance studies here

    def _bare(self, v: dict) -> dict:
        width = _si(v[self.var[(NONLIN, (self.qubit, self.qubit))]]) / 1e-6
        coupling = _si(v[self.var[(NONLIN, (self.qubit, self.resonator))]]) / 1e-6
        lj = _si(v[self.var[(FREQ, (self.qubit,))]])
        cg = (CG0_FF + CG_FF_PER_UM * coupling) * 1e-15
        c_q = (C_PADS0_FF + C_PADS_FF_PER_UM * width) * 1e-15 + cg
        ec = E**2 / (2 * c_q) / H  # Hz
        ej = PHI0_REDUCED**2 / lj / H
        f_q = math.sqrt(8 * ej * ec) - ec
        f_r = self.bare_resonator.freq(self.resonator, v)
        c_r = math.pi / (4 * 2 * math.pi * f_r * ZR_OHM)
        g = cg / (2 * math.sqrt(c_q * c_r)) * math.sqrt(f_q * f_r)
        return {"f_q": f_q, "f_r": f_r, "ec": ec, "g": g, "d": f_q - f_r}

    def freq(self, mode: str, v: dict) -> float:
        b = self._bare(v)
        lamb = b["g"] ** 2 / b["d"]  # pushes the two apart
        return b["f_q"] + lamb if mode == self.qubit else b["f_r"] - lamb

    def kappa(self, mode: str, v: dict) -> float:
        kappa_r = self.bare_resonator.kappa(self.resonator, v)
        if mode == self.resonator:
            return kappa_r
        b = self._bare(v)
        return (b["g"] / b["d"]) ** 2 * kappa_r  # Purcell

    def chi(self, v: dict) -> float:
        b = self._bare(v)
        alpha = -b["ec"]
        return abs(2 * b["g"] ** 2 * alpha / (b["d"] * (b["d"] - alpha)))

    def nonlin(self, mode_i, mode_j, idx_i, idx_j, v) -> float:
        del idx_i, idx_j
        modes = {mode_i, mode_j}
        if modes == {self.qubit}:
            return self._bare(v)["ec"]
        if modes == {self.qubit, self.resonator}:
            return self.chi(v)
        return self.chi(v) ** 2 / (4 * self._bare(v)["ec"])  # resonator Kerr

    def value(self, name, design_variables):
        raise AssertionError(f"{name} is not simulated here")


def optimize_qubit_readout(tmp_path, monkeypatch):
    """Run the optimization; returns the analysis and the model."""
    session = prepare_example(COUPLED_TRANSMON, tmp_path, monkeypatch)
    stored = json.loads((tmp_path / "design_variables.json").read_text("utf-8"))
    (tmp_path / "design_variables.json").write_text(
        json.dumps({**stored, **START}), "utf-8"
    )
    import design as d
    import mini_studies as ms
    import names as n
    import optimization_targets as ot
    import parameter_targets as pt

    from qdesignoptimizer.design_analysis import DesignAnalysis, DesignAnalysisState
    from qdesignoptimizer.utils.chip_generation import create_chip_base

    qubit, resonator = n.QUBIT_1, n.RESONATOR_1
    targets = ot.get_opt_targets_2qubits_resonator_coupler(
        groups=[n.NBR_1],
        opt_target_qubit_freq=True,
        opt_target_qubit_anharm=True,
        opt_target_resonator_freq=True,
        opt_target_resonator_kappa=True,
        opt_target_resonator_qubit_chi=True,
        use_simple_resonator_qubit_chi_relation=False,
    )
    session.physics = physics = TransmonReadoutPhysics(
        qubit,
        resonator,
        {(t.target_param_type, tuple(t.involved_modes)): t.design_var for t in targets},
    )
    design, gui = create_chip_base(n.CHIP_NAME, d.chip_type, open_gui=True)
    d.render_qiskit_metal_design(design, gui)
    analysis = DesignAnalysis(
        DesignAnalysisState(
            design,
            lambda design, **kwargs: d.render_qiskit_metal_design(
                design, gui, **kwargs
            ),
            pt.PARAM_TARGETS,
        ),
        mini_study=ms.get_mini_study_qb_res(group=n.NBR_1),
        opt_targets=targets,
        save_path=None,
        update_design_variables=False,
    )
    for _ in range(ITERATIONS):
        analysis.optimize_target({}, {})
    return analysis, physics


def test_the_qubit_and_its_readout_reach_all_five_targets(tmp_path, monkeypatch):
    analysis, physics = optimize_qubit_readout(tmp_path, monkeypatch)
    qubit, resonator = physics.qubit, physics.resonator
    wanted = analysis.system_target_params
    first = analysis.optimization_results[0]["system_optimized_params"]
    assert abs(first[param(qubit, FREQ)] / wanted[param(qubit, FREQ)] - 1) > 0.1
    assert abs(first[param(resonator, FREQ)] / wanted[param(resonator, FREQ)] - 1) > 0.1

    errors = relative_errors_to_targets(analysis)
    assert len(errors) == 5 and max(errors.values()) < 1e-2, errors

    # The final design really has these values in the model.
    final = analysis.optimization_results[-1]["design_variables"]
    assert physics.freq(qubit, final) == pytest.approx(
        wanted[param(qubit, FREQ)], rel=1e-2
    )
    assert physics.chi(final) == pytest.approx(
        wanted[param_nonlin(qubit, resonator)], rel=1e-2
    )
