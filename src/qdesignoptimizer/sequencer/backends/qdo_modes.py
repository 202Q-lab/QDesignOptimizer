"""What kind of element each mode is, and the frequencies it can take:
the project's ``modes.yaml``, and the report's picture of where every
mode's target sits.

``modes.yaml``, beside ``design_variables.json``::

    coupler_1to2:
      kind: squid      # transmon (the default for a mode with a junction),
                       # squid, ats, linc or resonator
      asymmetry: 0     # squid: |Ej1 - Ej2| / (Ej1 + Ej2), 0 if not given
    qubit_1:
      range: [3.9, 4.4]  # GHz, any kind: the band as given

A squid's band is worked out from its frequency at zero flux (simulated,
or from the junctions fabricated) down to half a flux quantum, where its
Ej is ``asymmetry`` times as large. An ATS or a LINC has no such model
here: give its ``range``. A kind is one entry of ``KINDS``; a new one
(its band from its own formula) is one more entry.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, Callable

import yaml

MODES_FILE = "modes.yaml"

Band = tuple[float, float]


def _fixed(mode: dict, f_max: float | None, alpha: float | None) -> Band | None:
    return None


def _squid(mode: dict, f_max: float | None, alpha: float | None) -> Band | None:
    from qdesignoptimizer.estimation.junction import squid_frequency_range

    if not f_max or not alpha:
        return None
    return squid_frequency_range(f_max, abs(alpha), float(mode.get("asymmetry") or 0))


#: kind -> (what it is, its band from (its entry, f at its highest, |α|)
#: when no ``range`` is given).
KINDS: dict[str, tuple[str, Callable[..., Band | None]]] = {
    "transmon": ("fixed-frequency transmon", _fixed),
    "squid": ("SQUID: tunes down from zero flux", _squid),
    "ats": ("asymmetrically threaded SQUID: its range as given", _fixed),
    "linc": ("LINC coupler: its range as given", _fixed),
    "resonator": ("linear resonator", _fixed),
}


def load(project_root: Path) -> dict[str, dict]:
    """``{mode: {"kind", ...}}`` from the project's ``modes.yaml`` ({}
    without one).

    Raises:
        ValueError: For an unknown kind or a range that is not two
            numbers, naming the mode.
    """
    path = Path(project_root) / MODES_FILE
    try:
        data = yaml.safe_load(path.read_text("utf-8"))
    except FileNotFoundError:
        return {}
    except yaml.YAMLError as exc:
        raise ValueError(f"{MODES_FILE}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{MODES_FILE}: a mode per key, not {type(data).__name__}")
    out = {}
    for name, entry in data.items():
        entry = dict(entry or {})
        kind = str(entry.get("kind") or "transmon").lower()
        if kind not in KINDS:
            raise ValueError(
                f"{MODES_FILE}: {name}: kind {kind!r} is none of {sorted(KINDS)}"
            )
        entry["kind"] = kind
        if "range" in entry:
            band = entry["range"]
            if (
                not isinstance(band, (list, tuple))
                or len(band) != 2
                or not all(isinstance(v, (int, float)) for v in band)
            ):
                raise ValueError(f"{MODES_FILE}: {name}: range is [low, high] in GHz")
            entry["range"] = (min(band) * 1e9, max(band) * 1e9)
        out[str(name)] = entry
    return out


def kind_of(modes: dict[str, dict], mode: str, default: str = "transmon") -> str:
    return (modes.get(mode) or {}).get("kind") or default


def band(
    modes: dict[str, dict], mode: str, f_max: float | None, alpha: float | None
) -> Band | None:
    """The frequencies (Hz) ``mode`` can take: its ``range`` as given, or
    what its kind works out from ``f_max`` (its frequency at the top of
    its range) and ``alpha``; None for a fixed one."""
    entry = modes.get(mode) or {}
    if entry.get("range"):
        return entry["range"]
    return KINDS[kind_of(modes, mode)][1](entry, f_max, alpha)


def picture(lanes: list[dict], title: str = "") -> dict[str, str] | None:
    """Where every mode's target sits on one frequency axis: a lane per
    mode, ``{"mode", "kind", "target", "simulated", "fabricated",
    "band", "band_fab"}`` (Hz; any may be None). ``{"png": <base64>,
    "caption"}``, or None with no lane."""
    lanes = [lane for lane in lanes if _any_frequency(lane)]
    if not lanes:
        return None
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch

    lanes = sorted(lanes, key=_lane_order, reverse=True)
    figure = Figure(figsize=(7.5, 0.42 * len(lanes) + 1.3), dpi=150)
    axes = figure.add_subplot()
    ghz = 1e-9
    for y, lane in enumerate(lanes):
        for key, colour, alpha in (
            ("band", "#5c6bc0", 0.22),
            ("band_fab", "#ef6c00", 0.18),
        ):
            if lane.get(key):
                low, high = lane[key]
                axes.barh(
                    y,
                    (high - low) * ghz,
                    left=low * ghz,
                    height=0.55 if key == "band" else 0.3,
                    color=colour,
                    alpha=alpha,
                    linewidth=0,
                )
        if lane.get("target"):
            axes.plot(lane["target"] * ghz, y, "|", color="#263238", ms=16, mew=2)
            axes.annotate(
                f"{lane['target'] * ghz:.3f}",
                (lane["target"] * ghz, y),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                fontsize=6.5,
                color="#263238",
            )
        if lane.get("simulated"):
            axes.plot(lane["simulated"] * ghz, y, "o", color="#1565c0", ms=5)
        if lane.get("fabricated"):
            axes.plot(
                lane["fabricated"] * ghz,
                y,
                "o",
                mfc="none",
                mec="#ef6c00",
                mew=1.5,
                ms=7,
            )
    axes.set_yticks(
        range(len(lanes)),
        [
            f"{lane['mode']}" + (f"  ({lane['kind']})" if lane.get("kind") else "")
            for lane in lanes
        ],
        fontsize=8,
    )
    axes.set_ylim(-0.7, len(lanes) - 0.3)
    axes.set_xlabel("Frequency (GHz)", fontsize=8)
    axes.tick_params(axis="x", labelsize=8)
    axes.grid(axis="x", color="#eceff1")
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    handles = [
        axes.plot([], [], "|", color="#263238", ms=10, mew=2)[0],
        axes.plot([], [], "o", color="#1565c0", ms=5)[0],
        axes.plot([], [], "o", mfc="none", mec="#ef6c00", mew=1.5, ms=7)[0],
        Patch(color="#5c6bc0", alpha=0.3),
        Patch(color="#ef6c00", alpha=0.3),
    ]
    labels = [
        "target",
        "simulated",
        "from fabrication",
        "tuning range",
        "range from fabrication",
    ]
    shown = [
        (h, label)
        for h, label, key in zip(
            handles,
            labels,
            ("target", "simulated", "fabricated", "band", "band_fab"),
        )
        if any(lane.get(key) for lane in lanes)
    ]
    axes.legend(
        [h for h, _ in shown],
        [label for _, label in shown],
        fontsize=7,
        ncol=len(shown),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
    )
    if title:
        figure.suptitle(title, fontsize=9, y=0.995)
    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png")
    return {
        "png": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "caption": (
            "Where each mode's target sits (its value above the mark), with "
            "what the latest study that tuned it simulated and, where "
            "fabrication gave the junction, the frequency that follows. A "
            "band is the range a tunable mode can take (modes.yaml)."
        ),
    }


#: The editor's columns (the GUI's report pane): range in GHz.
EDITOR_COLUMNS = ["mode", "kind", "asymmetry", "low", "high"]

_HEADER = """\
# What kind of element each mode is, and the frequencies it can take:
# drawn as a band in the report's frequency picture.
#   kind:      transmon (the default for a mode with a junction), squid,
#              ats, linc or resonator
#   asymmetry: a squid's |Ej1 - Ej2| / (Ej1 + Ej2); its band runs from
#              half a flux quantum up to zero flux (0: down to 0 Hz)
#   range:     [low, high] in GHz, any kind: the band as given (an ATS's
#              or a LINC's, or one measured)
"""


def editor(project_root: Path, found: dict[str, str]) -> dict:
    """The rows to edit: every mode in ``modes.yaml`` and every one in
    ``found`` (``{mode: its kind when not given}``), ``{"rows": [{"mode",
    "kind", "asymmetry", "low", "high"}], "kinds": {kind: what it is},
    "file", "problem"}``; ``problem`` says why the file did not read
    (its modes are then left out)."""
    problem = ""
    try:
        given = load(project_root)
    except ValueError as exc:
        given, problem = {}, str(exc)
    rows = []
    for mode in sorted(set(given) | set(found)):
        entry = given.get(mode) or {}
        low, high = entry.get("range") or (None, None)
        asymmetry = entry.get("asymmetry")
        rows.append(
            {
                "mode": mode,
                "kind": entry.get("kind") or found.get(mode) or "transmon",
                "asymmetry": "" if asymmetry is None else asymmetry,
                "low": "" if low is None else round(low / 1e9, 6),
                "high": "" if high is None else round(high / 1e9, 6),
            }
        )
    return {
        "rows": rows,
        "kinds": {kind: text for kind, (text, _) in KINDS.items()},
        "file": str(Path(project_root) / MODES_FILE),
        "problem": problem,
    }


def save(project_root: Path, rows: list[dict]) -> Path:
    """Write ``modes.yaml`` from the editor's rows, keeping the file's
    comments at its top and any other key a mode has there.

    Raises:
        ValueError: For an unknown kind, a number that does not read, an
            asymmetry outside [0, 1] or half a range, naming the mode.
    """
    path = Path(project_root) / MODES_FILE
    try:
        text = path.read_text("utf-8")
    except FileNotFoundError:
        text = ""
    try:
        raw = yaml.safe_load(text) or {}
    except yaml.YAMLError:
        raw = {}
    raw = raw if isinstance(raw, dict) else {}
    out: dict[str, dict] = {}
    for row in rows:
        mode = str(row.get("mode") or "").strip()
        if not mode:
            continue
        kind = str(row.get("kind") or "transmon").strip().lower()
        if kind not in KINDS:
            raise ValueError(f"{mode}: kind {kind!r} is none of {sorted(KINDS)}")
        previous = raw.get(mode)
        entry: dict[str, Any] = {"kind": kind}
        entry.update(
            (k, v)
            for k, v in (previous if isinstance(previous, dict) else {}).items()
            if k not in ("kind", "asymmetry", "range")
        )
        asymmetry = _editor_number(mode, "asymmetry", row.get("asymmetry"))
        if asymmetry is not None:
            if not 0 <= asymmetry <= 1:
                raise ValueError(f"{mode}: asymmetry {asymmetry} is not in [0, 1]")
            entry["asymmetry"] = asymmetry
        low = _editor_number(mode, "low", row.get("low"))
        high = _editor_number(mode, "high", row.get("high"))
        if (low is None) != (high is None):
            raise ValueError(f"{mode}: give both ends of the range, or neither")
        if low is not None and high is not None:
            entry["range"] = [min(low, high), max(low, high)]
        out[mode] = entry
    header = []
    for line in text.splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            break
        header.append(line)
    while header and not header[-1].strip():
        header.pop()
    head = "\n".join(header) + "\n" if header else _HEADER
    path.write_text(
        head
        + (
            yaml.safe_dump(
                dict(sorted(out.items())), sort_keys=False, default_flow_style=None
            )
            if out
            else ""
        ),
        encoding="utf-8",
    )
    return path


def _editor_number(mode: str, what: str, value) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{mode}: {what} {value!r} is not a number") from None


def _any_frequency(lane: dict) -> bool:
    return any(lane.get(k) for k in ("target", "simulated", "fabricated", "band"))


def _lane_order(lane: dict) -> float:
    for key in ("target", "simulated", "fabricated"):
        if lane.get(key):
            return float(lane[key])
    return float((lane.get("band") or (0.0, 0.0))[1])
