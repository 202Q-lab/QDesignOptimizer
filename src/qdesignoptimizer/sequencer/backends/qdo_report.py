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
    "Element",
    "Kind",
    "f01 sim (GHz)",
    "f01 target (GHz)",
    "α (MHz)",
    "Tuning range (GHz)",
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
    "Tuning range from fab (GHz)",
    "From study",
]
#: Columns a report shows as a side note (``engine.report``).
ASIDE = ["From study"]


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


def table(
    runs: list[dict], fabrication: dict[str, str], modes: dict | None = None
) -> dict:
    """The modes this report is about, simulated and as fabricated.

    Per element (a mode with a junction: a qubit, a coupler): what it was
    simulated as, its targets, its readout resonator, and -- when
    ``fabrication`` names it -- its junction as made and the frequency
    that follows (the simulated one rescaled, see
    ``transmon_frequency_for_josephson_energy``). ``modes`` (the
    project's ``modes.yaml``, see ``qdo_modes``) says what kind each is
    and the range a tunable one can take.

    ``rows`` are flat, a row per element (the paste for Notion);
    ``display`` is the report's table: a group of rows per element, each
    quantity numbered and the ones it follows from named (``← ①``).
    ``picture`` is where every mode's target sits (``qdo_modes.picture``).

    A run that tunes some modes reports only those: the others are in its
    model to load the ones it tunes, and their values are the model's,
    not the chip's (a partition solves the coupler beside qubit 1 alone).
    A run with no target on any mode's own parameters reports them all.

    Of the runs that tuned a mode, its values are from the latest that
    wrote its design back (as the design stands), else the latest; the
    study is a side note."""
    from qdesignoptimizer.estimation.junction import (
        ALUMINIUM_GAP_EV,
        josephson_energy_from_inductance,
        josephson_energy_from_measurement,
        transmon_frequency_for_josephson_energy,
    )
    from qdesignoptimizer.utils.names_design_variables import design_var_lj
    from qdesignoptimizer.utils.names_parameters import param, param_nonlin

    from . import qdo_modes

    modes = modes or {}
    gap = ALUMINIUM_GAP_EV
    if fabrication.get("gap_ueV"):
        gap = float(fabrication["gap_ueV"]) * 1e-6
    found, notes = [], []
    simulated: dict[str, tuple] = {}  # mode -> (rank, frequency)
    targets: dict[str, float] = {}
    for run in sorted(runs, key=lambda r: r.get("started", "")):
        outputs = run["outputs"]
        modes_run = sorted(k[: -len("_freq")] for k in outputs if k.endswith("_freq"))
        started = ran_with(run["folder"])
        params = _parameter_targets(run["folder"])
        targets.update(
            {
                k[: -len("_freq")]: number
                for k, v in params.items()
                if k.endswith("_freq") and (number := _number(v)) is not None
            }
        )
        rank = (bool(run.get("committed")), run.get("started", ""))
        for mode in tuned_modes(run, modes_run):
            f = outputs.get(param(mode, "freq"))
            if isinstance(f, (int, float)) and (
                mode not in simulated or rank > simulated[mode][0]
            ):
                simulated[mode] = (rank, float(f))

        def lj_of(mode, run=run, started=started):
            return _last_design(run, design_var_lj(mode)) or started.get(
                design_var_lj(mode)
            )

        linear = [m for m in modes_run if lj_of(m) is None]
        for qubit in tuned_modes(run, modes_run):
            alpha = outputs.get(param_nonlin(qubit, qubit))
            lj_text = lj_of(qubit)
            if alpha is None or lj_text is None:
                continue  # not a mode with a junction this study measured
            alpha = abs(alpha)
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
                            f01, alpha, ej, ej_fab
                        )
                except ValueError as exc:
                    notes.append(f"{qubit}: {exc}")
            chi = param_nonlin(qubit, readout) if readout else ""
            element = {
                "mode": qubit,
                "kind": qdo_modes.kind_of(modes, qubit),
                "f01": f01,
                "f01_target": params.get(param(qubit, "freq")),
                "alpha": alpha,
                "alpha_target": params.get(param_nonlin(qubit, qubit)),
                "readout": readout,
                "fr": outputs.get(param(readout, "freq")) if readout else None,
                "fr_target": params.get(param(readout, "freq")) if readout else None,
                "kappa": outputs.get(param(readout, "kappa")) if readout else None,
                "kappa_target": (
                    params.get(param(readout, "kappa")) if readout else None
                ),
                "chi": abs(outputs[chi]) if chi and chi in outputs else None,
                "chi_target": params.get(chi) if chi else None,
                "lj": lj,
                "ej": ej,
                "measured": measured,
                "ej_fab": ej_fab,
                "f01_fab": f01_fab,
                "band": qdo_modes.band(modes, qubit, f01, alpha),
                "band_fab": (
                    qdo_modes.band(modes, qubit, f01_fab, alpha)
                    if f01_fab and not (modes.get(qubit) or {}).get("range")
                    else None
                ),
                "study": study_label(run),
            }
            found.append((qubit, rank[0], rank[1], element))
    elements = _latest_per_qubit(found)
    rows = [_flat_row(e) for e in elements]
    if rows:
        notes.append(
            "f01 from fabrication = (f01 sim + α) · sqrt(Ej fab / Ej design) − α: "
            "the simulated transmon with the junction it got, its "
            "capacitances as drawn (α ≈ Ec). For a SQUID, f01 at zero flux "
            "and Ej the junctions' sum; for an ATS or a LINC, the same "
            "rescaling at its operating point (an estimate)."
        )
        notes.append(
            "Fabricated: Ej (GHz), a normal resistance Rn (kOhm, by "
            f"Ambegaokar-Baratoff, gap {gap * 1e6:g} µeV) or Lj (nH); "
            "set in the project's fabrication.yaml or in the GUI."
        )
        if any(e["band"] for e in elements):
            notes.append(
                "Tuning range: from the project's modes.yaml. A SQUID's runs "
                "from half a flux quantum (Ej times its asymmetry d) up to "
                "zero flux, (f + α) · sqrt(d) − α; an ATS's or a LINC's is "
                "as given there."
            )
    by_mode = {e["mode"]: e for e in elements}
    lanes = []
    for mode in sorted(set(targets) | set(simulated)):
        element = by_mode.get(mode) or {}
        kind = qdo_modes.kind_of(modes, mode, "transmon" if element else "resonator")
        lanes.append(
            {
                "mode": mode,
                "kind": kind,
                "target": targets.get(mode),
                "simulated": (simulated.get(mode) or (None, None))[1],
                "fabricated": element.get("f01_fab"),
                "band": element.get("band") or qdo_modes.band(modes, mode, None, None),
                "band_fab": element.get("band_fab"),
            }
        )
    try:
        picture = qdo_modes.picture(lanes)
    except Exception as exc:  # noqa: BLE001 -- the table stands without
        picture = None
        notes.append(f"No frequency picture: {exc}")
    return {
        "title": "Qubits and couplers: simulated and fabricated",
        "columns": COLUMNS,
        "aside": ASIDE,
        "rows": rows,
        "display": _display(elements),
        "notes": notes,
        "picture": picture,
        "picture_title": "Frequencies",
    }


def _flat_row(e: dict) -> list:
    return [
        e["mode"],
        e["kind"],
        _fmt(e["f01"], 1e9, 4),
        _fmt(e["f01_target"], 1e9, 4),
        _fmt(e["alpha"], 1e6, 1),
        _band_text(e["band"]),
        e["readout"] or "",
        _fmt(e["fr"], 1e9, 4),
        _fmt(e["kappa"], 1e6, 3),
        _fmt(e["chi"], 1e6, 3),
        _fmt(e["lj"], 1e-9, 2),
        _fmt(e["ej"], 1e9, 3),
        e["measured"],
        _fmt(e["ej_fab"], 1e9, 3),
        _fmt(e["f01_fab"], 1e9, 4),
        _fmt((e["f01_fab"] - e["f01"]) if e["f01_fab"] and e["f01"] else None, 1e6, 0),
        _band_text(e["band_fab"]),
        e["study"],
    ]


#: The report's table (``display``): a group of rows per element.
DISPLAY_COLUMNS = [
    "Element",
    "#",
    "Quantity",
    "Simulated",
    "Target",
    "From fabrication",
    "Δ fab − sim",
    "From study",
]
_CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"


def _display(elements: list[dict]) -> dict:
    """Each element's quantities as rows, numbered, each naming the
    numbers it follows from: the junction (①) sets Ej (② ← ①), Ej the
    frequency (③ ← ②), the frequency the range a tunable one can take."""
    rows = []
    for e in elements:
        delta_ej = (
            f"{(e['ej_fab'] / e['ej'] - 1) * 100:+.1f} %"
            if e["ej_fab"] and e["ej"]
            else ""
        )
        delta_f = (
            f"{(e['f01_fab'] - e['f01']) / 1e6:+.0f} MHz"
            if e["f01_fab"] and e["f01"]
            else ""
        )
        readout = e["readout"]
        quantities = [
            (
                "lj",
                "Junction Lj",
                _unit(e["lj"], 1e-9, 2, "nH"),
                "",
                e["measured"],
                "",
                [],
            ),
            (
                "ej",
                "Ej",
                _unit(e["ej"], 1e9, 3, "GHz"),
                "",
                _unit(e["ej_fab"], 1e9, 3, "GHz"),
                delta_ej,
                ["lj"],
            ),
            (
                "f01",
                "f01" + (" at zero flux" if e["kind"] == "squid" else ""),
                _unit(e["f01"], 1e9, 4, "GHz"),
                _unit(e["f01_target"], 1e9, 4, "GHz"),
                _unit(e["f01_fab"], 1e9, 4, "GHz"),
                delta_f,
                ["ej"],
            ),
            (
                "alpha",
                "Anharmonicity |α|",
                _unit(e["alpha"], 1e6, 1, "MHz"),
                _unit(e["alpha_target"], 1e6, 1, "MHz"),
                "",
                "",
                [],
            ),
            (
                "band",
                "Tuning range",
                _band_text(e["band"], " GHz"),
                "",
                _band_text(e["band_fab"], " GHz"),
                "",
                ["f01"] if e["kind"] == "squid" else [],
            ),
            (
                "fr",
                f"Readout {readout} f",
                _unit(e["fr"], 1e9, 4, "GHz"),
                _unit(e["fr_target"], 1e9, 4, "GHz"),
                "",
                "",
                [],
            ),
            (
                "kappa",
                f"Readout {readout} κ",
                _unit(e["kappa"], 1e6, 3, "MHz"),
                _unit(e["kappa_target"], 1e6, 3, "MHz"),
                "",
                "",
                [],
            ),
            (
                "chi",
                f"χ with {readout}",
                _unit(e["chi"], 1e6, 3, "MHz"),
                _unit(e["chi_target"], 1e6, 3, "MHz"),
                "",
                "",
                [],
            ),
        ]
        shown = [q for q in quantities if any(q[2:5])]
        number = {q[0]: _CIRCLED[n % len(_CIRCLED)] for n, q in enumerate(shown)}
        for n, (key, label, sim, target, fab, delta, needs) in enumerate(shown):
            follows = " ".join(number[k] for k in needs if k in number)
            rows.append(
                [
                    f"**{e['mode']}** ({e['kind']})" if n == 0 else "",
                    number[key],
                    label + (f" ← {follows}" if follows else ""),
                    sim,
                    target,
                    fab,
                    delta,
                    e["study"] if n == 0 else "",
                ]
            )
    return {"columns": DISPLAY_COLUMNS, "rows": rows, "aside": ["From study"]}


def _number(value: Any) -> float | None:
    """A number, also one YAML read as text (``7.0e9``: YAML 1.1 wants
    ``7.0e+9``)."""
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _unit(number: Any, scale: float, decimals: int, unit: str) -> str:
    text = _fmt(number, scale, decimals)
    return f"{text} {unit}" if text else ""


def _band_text(band: Any, unit: str = "") -> str:
    if not band:
        return ""
    low, high = band
    return f"{low / 1e9:.3f}–{high / 1e9:.3f}{unit.replace(' ', chr(160))}"


def _latest_per_qubit(found: list[tuple]) -> list:
    """One per qubit, by name: of ``(qubit, wrote back, started,
    row)``, the latest that wrote back, else the latest."""
    best: dict[str, tuple] = {}
    for qubit, wrote, started, row in found:
        rank = (wrote, started)
        if qubit not in best or rank > best[qubit][0]:
            best[qubit] = (rank, row)
    return [best[q][1] for q in sorted(best)]


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
