"""What a QDO study adds to a report (``engine.report``): its targets per
iteration, its values with units, and a table of the qubits it tuned with
columns for what fabrication gave and the frequency that follows from it."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import yaml

from .qdo_record import CONVERGENCE_JSON

_SUFFIXES = (
    ("_freq", 1e9, "GHz", 4),
    ("_kappa", 1e6, "MHz", 4),
    ("_nonlin", 1e6, "MHz", 4),
    ("_charge_line_limited_t1", 1e-6, "us", 4),
)

#: The table's columns (Notion takes them as the header row).
COLUMNS = [
    "Qubit",
    "Study",
    "f01 sim (GHz)",
    "f01 target (GHz)",
    "α (MHz)",
    "Readout",
    "f_r sim (GHz)",
    "κ (MHz)",
    "χ (MHz)",
    "Lj design (nH)",
    "Ej design (GHz)",
    "Fabricated",
    "Ej fab (GHz)",
    "f01 from fab (GHz)",
    "Δf01 (MHz)",
]


def targets(folder: Path) -> list[dict]:
    """The study's targets per iteration (its ``convergence.json``)."""
    path = Path(folder) / CONVERGENCE_JSON
    if not path.exists():
        return []
    return list(json.loads(path.read_text("utf-8")).get("targets") or [])


def value(parameter: str, number: Any) -> str:
    """A QDO parameter's value as read: ``4.0550 GHz``, ``1.103 MHz``."""
    if not isinstance(number, (int, float)) or isinstance(number, bool):
        return str(number)
    for suffix, scale, unit, digits in _SUFFIXES:
        if parameter.endswith(suffix):
            return f"{number / scale:.{digits}g} {unit}"
    return f"{number:.6g}"


def table(runs: list[dict], fabrication: dict[str, str]) -> dict:
    """A row per qubit a run tuned: what it was simulated as, and --
    when ``fabrication`` names it -- its junction as made and the
    frequency that follows (the simulated one rescaled, see
    ``transmon_frequency_for_josephson_energy``).

    A run that tunes some modes reports only those: the others are in its
    model to load the ones it tunes, and their values are the model's,
    not the chip's (a partition solves the coupler beside qubit 1 alone).
    A run with no target on any mode's own parameters reports them all."""
    from qdesignoptimizer.estimation.junction import (
        ALUMINIUM_GAP_EV,
        josephson_energy_from_inductance,
        josephson_energy_from_measurement,
        transmon_frequency_for_josephson_energy,
    )
    from qdesignoptimizer.utils.names_design_variables import design_var_lj
    from qdesignoptimizer.utils.names_parameters import param, param_nonlin

    gap = ALUMINIUM_GAP_EV
    if fabrication.get("gap_ueV"):
        gap = float(fabrication["gap_ueV"]) * 1e-6
    rows, notes = [], []
    for run in runs:
        outputs = run["outputs"]
        modes = sorted(k[: -len("_freq")] for k in outputs if k.endswith("_freq"))
        started = ran_with(run["folder"])
        params = _parameter_targets(run["folder"])

        def lj_of(mode, run=run, started=started):
            return _last_design(run, design_var_lj(mode)) or started.get(
                design_var_lj(mode)
            )

        linear = [m for m in modes if lj_of(m) is None]
        for qubit in tuned_modes(run, modes):
            alpha = outputs.get(param_nonlin(qubit, qubit))
            lj_text = lj_of(qubit)
            if alpha is None or lj_text is None:
                continue  # not a transmon this study measured
            f01 = outputs.get(param(qubit, "freq"))
            readout = _readout(qubit, linear, outputs, param_nonlin)
            lj = _henry(lj_text)
            ej = josephson_energy_from_inductance(lj) if lj else None
            measured = fabrication.get(qubit, "")
            ej_fab = f01_fab = None
            if measured:
                try:
                    ej_fab = josephson_energy_from_measurement(measured, gap)
                    if f01 and ej:
                        f01_fab = transmon_frequency_for_josephson_energy(
                            f01, abs(alpha), ej, ej_fab
                        )
                except ValueError as exc:
                    notes.append(f"{qubit}: {exc}")
            rows.append(
                [
                    qubit,
                    study_label(run),
                    _fmt(f01, 1e9, 4),
                    _fmt(params.get(param(qubit, "freq")), 1e9, 4),
                    _fmt(abs(alpha), 1e6, 1),
                    readout or "",
                    _fmt(
                        outputs.get(param(readout, "freq")) if readout else None, 1e9, 4
                    ),
                    _fmt(
                        outputs.get(param(readout, "kappa")) if readout else None,
                        1e6,
                        3,
                    ),
                    _fmt(
                        (
                            abs(outputs[param_nonlin(qubit, readout)])
                            if readout and param_nonlin(qubit, readout) in outputs
                            else None
                        ),
                        1e6,
                        3,
                    ),
                    _fmt(lj, 1e-9, 2),
                    _fmt(ej, 1e9, 3),
                    measured,
                    _fmt(ej_fab, 1e9, 3),
                    _fmt(f01_fab, 1e9, 4),
                    _fmt((f01_fab - f01) if f01_fab and f01 else None, 1e6, 0),
                ]
            )
    if rows:
        notes.append(
            "f01 from fab = (f01 sim + α) · sqrt(Ej fab / Ej design) − α: "
            "the simulated transmon with the junction it got, its "
            "capacitances as drawn (α ≈ Ec)."
        )
        notes.append(
            "Fabricated: Ej (GHz), a normal resistance Rn (kOhm, by "
            f"Ambegaokar-Baratoff, gap {gap * 1e6:g} µeV) or Lj (nH); "
            "set in the project's fabrication.yaml or in the GUI."
        )
    return {
        "title": "Qubits: simulated and fabricated",
        "columns": COLUMNS,
        "rows": rows,
        "notes": notes,
    }


def tuned_modes(run: dict, modes: list[str]) -> list[str]:
    """The modes whose own parameters (frequency, decay, anharmonicity,
    charge-line T1) the run has a target on; all of them if none."""
    from qdesignoptimizer.utils.names_parameters import param, param_nonlin

    aimed = {t.get("parameter") for t in run.get("targets") or []}
    tuned = [
        m
        for m in modes
        if aimed
        & {
            param(m, "freq"),
            param(m, "kappa"),
            param(m, "charge_line_limited_t1"),
            param_nonlin(m, m),
        }
    ]
    return tuned or modes


def study_label(run: dict) -> str:
    """The study and, in a sequence, its step: ``Q1.readout · s001 step 4``
    (a sequence runs one study several times; the id says so less
    plainly), and ``rerun 2`` for a requeued copy (its id's ``_r2``)."""
    label = run.get("task") or run["id"]
    sequence = run.get("sequence") or {}
    if sequence.get("id") and sequence.get("entry"):
        label = f"{label} · {sequence['id']} step {sequence['entry']}"
    rerun = re.search(r"_r(\d+)$", run["id"]) if run.get("requeued_from") else None
    if rerun:
        label = f"{label}, rerun {rerun.group(1)}"
    return label


def _readout(qubit, linear, outputs, param_nonlin) -> str:
    """The mode the qubit is read out through: the linear mode (no
    junction) it has the largest cross-Kerr with; none if the model has
    no linear mode coupled to it."""
    coupled = [m for m in linear if param_nonlin(qubit, m) in outputs]
    if not coupled:
        return ""
    return max(coupled, key=lambda m: abs(outputs[param_nonlin(qubit, m)] or 0))


def _last_design(run: dict, var: str) -> str | None:
    """The design variable as the last iteration simulated it."""
    for target in run["targets"]:
        if target.get("design_var") == var:
            found = [v for v in target.get("design_values") or [] if v is not None]
            if found:
                return f"{found[-1]} {target.get('unit', '')}"
    return None


def ran_with(folder: Path) -> dict:
    """The design variables a study started from (its folder's copy of
    the store as the study began)."""
    try:
        return json.loads((Path(folder) / "design_variables.json").read_text("utf-8"))
    except (OSError, ValueError):
        return {}


def _parameter_targets(folder: Path) -> dict:
    try:
        data = yaml.safe_load(
            (Path(folder) / "parameter_targets.yaml").read_text("utf-8")
        )
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


_HENRY = {"h": 1.0, "nh": 1e-9, "ph": 1e-12, "uh": 1e-6}


def _henry(text: Any) -> float | None:
    match = re.match(r"\s*([-+0-9.eE]+)\s*([A-Za-z]*)", str(text))
    if not match:
        return None
    unit = match.group(2).lower() or "h"
    try:
        return float(match.group(1)) * _HENRY[unit] if unit in _HENRY else None
    except ValueError:
        return None


def _fmt(number: Any, scale: float, decimals: int) -> str:
    if number is None or not isinstance(number, (int, float)) or math.isnan(number):
        return ""
    return f"{number / scale:.{decimals}f}"
