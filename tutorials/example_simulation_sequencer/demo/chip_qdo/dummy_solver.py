"""A dummy in place of Ansys: a toy model, a QDO-style update, 5 s.

``solve`` takes what ``QDOBackend.run`` hands a real project's solve --
the MiniStudy, its OptTargets, the design variables and the parameter
targets -- and returns what it reads back from QDO's DesignAnalysis:
``optimization_results`` (one entry per iteration, the measured
parameters under ``system_optimized_params``) and a ``save_path`` whose
``_design_variables.json`` holds the proposed values.

The "measurement" is a toy model of a quarter-wave resonator on a tee:

    f     = 5.0 GHz * (4000 um / length) ** 0.9
    kappa = 0.5 MHz * (coupling / 500 um) ** 2.2 * (f / 5 GHz) ** 1.5

and each iteration moves every target's design variable the way QDO
does: so that the target's ``prop_to`` ratio equals target / measured.
The model's exponents (0.9, 2.2) are deliberately not the helpers'
(1, 2), so a study takes a few iterations to converge -- as a real one
does. A study's ``seconds`` setting is split over its iterations.

Whether a target was hit is not judged here: each opt target carries
its own ``tolerance`` (``opt_targets/resonators.yaml``).
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from pathlib import Path
from types import SimpleNamespace

log = logging.getLogger("qdesignoptimizer.sequencer.demo")

_VALUE = re.compile(r"^\s*([-+0-9.eE]+)\s*([A-Za-z]*)\s*$")


def _number(value):
    """``"4000.0 um"`` -> ``(4000.0, "um")``."""
    if isinstance(value, (int, float)):
        return float(value), ""
    match = _VALUE.match(str(value))
    if not match:
        raise ValueError(f"cannot read {value!r} as a number")
    return float(match.group(1)), match.group(2)


def measure(study, values: dict) -> dict:
    """What the toy model gives for a study's modes, in Hz."""
    out = {}
    for resonator in study.modes:
        length = values[f"design_var_length_{resonator}"]
        coupling = values[f"design_var_coupl_length_{resonator}_tee"]
        freq = 5.0e9 * (4000.0 / length) ** 0.9
        if study.run_capacitance_studies_only:
            out[f"{resonator}_kappa"] = (
                0.5e6 * (coupling / 500.0) ** 2.2 * (freq / 5.0e9) ** 1.5
            )
        else:
            out[f"{resonator}_freq"] = freq
    return out


def _param(target) -> str:
    return f"{target.involved_modes[0]}_{target.target_param_type}"


def _bounds(target, unit: str):
    lo, hi = 0.0, math.inf
    for key, value in (target.design_var_constraint or {}).items():
        number, _ = _number(value)
        if key == "larger_than":
            lo = number
        elif key == "smaller_than":
            hi = number
    return lo, hi


def _step(target, values: dict, measured: dict, wanted: float) -> float:
    """The design variable's next value: where ``prop_to`` has moved by
    ``wanted / measured`` (bisection on a log scale, as the ratio is
    monotonic in one variable)."""
    name = target.design_var
    old = values[name]
    ratio = wanted / measured[_param(target)]
    base = target.prop_to(measured, values)

    def miss(x):
        trial = {**values, name: x}
        return target.prop_to(measured, trial) / base - ratio

    lo, hi = old / 20.0, old * 20.0
    rising = miss(hi) > miss(lo)
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        if (miss(mid) > 0) == rising:
            hi = mid
        else:
            lo = mid
    return math.sqrt(lo * hi)


def solve(study, targets, folder, settings, variables, params):
    """Iterate the toy model; see the module docstring."""
    iterations = int(settings.get("iterations") or 3)
    seconds = float(settings.get("seconds", 5))
    units = {}
    values = {}
    for name, value in variables.items():
        values[name], units[name] = _number(value)
    results = []
    for n in range(1, iterations + 1):
        time.sleep(seconds / iterations)
        measured = measure(study, values)
        # As QDO records it: the values this iteration solved with.
        solved_with = {
            name: f"{values[name]:.4f} {units[name]}".strip() for name in variables
        }
        for target in targets:
            wanted = params[_param(target)]
            moved = _step(target, values, measured, wanted)
            lo, hi = _bounds(target, units[target.design_var])
            values[target.design_var] = min(max(moved, lo), hi)
        results.append(
            {
                "system_optimized_params": dict(measured),
                "design_variables": solved_with,
            }
        )
        log.info(
            "iteration %d/%d: %s",
            n,
            iterations,
            ", ".join(
                (
                    f"{k} {v / 1e9:.4f} GHz (target {params[k] / 1e9:.4f})"
                    if k.endswith("_freq")
                    else f"{k} {v / 1e6:.3f} MHz (target {params[k] / 1e6:.3f})"
                )
                for k, v in measured.items()
            ),
        )
    save_path = Path(folder) / "dummy_solve"
    # As QDO does: the geometry the last iteration measured, so that the
    # "converged" check (judged on that measurement) is about what is
    # written back -- not the step after it, which nothing measured.
    proposed = dict(results[-1]["design_variables"]) if results else {}
    Path(str(save_path) + "_design_variables.json").write_text(
        json.dumps(proposed, indent=4) + "\n", encoding="utf-8"
    )
    return SimpleNamespace(
        optimization_results=results,
        save_path=str(save_path),
        targets=list(targets),
        params=dict(params),
    )
