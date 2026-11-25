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
from typing import Callable

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


def _any_frequency(lane: dict) -> bool:
    return any(lane.get(k) for k in ("target", "simulated", "fabricated", "band"))


def _lane_order(lane: dict) -> float:
    for key in ("target", "simulated", "fabricated"):
        if lane.get(key):
            return float(lane[key])
    return float((lane.get("band") or (0.0, 0.0))[1])
