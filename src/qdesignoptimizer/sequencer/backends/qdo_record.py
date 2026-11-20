"""What a QDO study keeps of its own run, besides QDO's files.

* ``convergence.json`` / ``convergence.png`` -- every target parameter
  and its design variable, iteration by iteration, read from the
  analysis's ``optimization_results`` (``system_optimized_params`` and
  ``design_variables`` per iteration);
* ``pictures/`` -- the study's pictures (``qdo_preview``), one per solver
  run, as the solve was about to draw it;
* ``study.log`` -- every log line written while the study ran.

Imports matplotlib only inside functions (the backends import nothing
heavy at module scope).
"""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Iterator

CONVERGENCE_JSON = "convergence.json"
CONVERGENCE_PNG = "convergence.png"
PICTURES = "pictures"
STUDY_LOG = "study.log"


def _number(value: Any) -> float | None:
    """A design variable's number (``"7020.5 um"`` -> 7020.5)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    match = re.match(r"\s*([-+0-9.eE]+)", str(value))
    try:
        return float(match.group(1)) if match else None
    except ValueError:
        return None


def _unit(value: Any) -> str:
    match = re.match(r"\s*[-+0-9.eE]+\s*([A-Za-z]*)", str(value))
    return match.group(1) if match else ""


def convergence(analysis, targets: list, params: dict, parameter) -> dict:
    """Each target's parameter and design variable per iteration.

    Args:
        analysis: What the project's ``solve`` returned.
        targets: The study's OptTargets.
        params: The parameter targets the study ran against.
        parameter: ``parameter(target)`` -> the parameter name it chases.

    Returns:
        ``{"iterations": n, "targets": [{"parameter", "target",
        "values", "design_var", "unit", "design_values"}]}``; a value an
        iteration did not record is None.
    """
    results = list(getattr(analysis, "optimization_results", None) or [])
    out = []
    for target in targets:
        name = parameter(target)
        var = getattr(target, "design_var", None)
        values, design = [], []
        unit = ""
        for result in results:
            measured = (result or {}).get("system_optimized_params") or {}
            value = measured.get(name)
            values.append(float(value) if value is not None else None)
            raw = ((result or {}).get("design_variables") or {}).get(var)
            design.append(_number(raw) if raw is not None else None)
            unit = unit or (_unit(raw) if raw is not None else "")
        out.append(
            {
                "parameter": name,
                "target": params.get(name),
                "values": values,
                "design_var": var,
                "unit": unit,
                "design_values": design,
            }
        )
    return {"iterations": len(results), "targets": out}


def plot_convergence(data: dict, path: Path, title: str = "") -> Path:
    """One row per target: the parameter against its target (left), and
    the design variable that chased it (right), per iteration."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = data["targets"] or [{}]
    fig, axes = plt.subplots(
        len(rows),
        2,
        figsize=(10, 2.6 * len(rows) + 0.6),
        dpi=110,
        squeeze=False,
    )
    for row, item in zip(axes, rows):
        left, right = row
        values = item.get("values") or []
        steps = list(range(1, len(values) + 1))
        scale, unit = _scale(item.get("parameter") or "")
        shown = [v / scale if v is not None else None for v in values]
        left.plot(steps, shown, "o-", color="#2a78d6", label="simulated")
        if item.get("target") is not None:
            left.axhline(
                item["target"] / scale,
                color="#eb6834",
                linestyle="--",
                label="target",
            )
        left.set_title(f"{item.get('parameter', '')} ({unit})", fontsize=9)
        left.set_xlabel("iteration")
        left.legend(fontsize=7)
        design = item.get("design_values") or []
        right.plot(list(range(1, len(design) + 1)), design, "s-", color="#3c9e4a")
        right.set_title(
            f"{item.get('design_var', '')} ({item.get('unit', '')})",
            fontsize=9,
        )
        right.set_xlabel("iteration")
        for axis in row:
            axis.grid(color="#eeeeee")
            axis.xaxis.get_major_locator().set_params(integer=True)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _scale(parameter: str) -> tuple[float, str]:
    """Plot units for a QDO parameter name."""
    if parameter.endswith("_freq"):
        return 1e9, "GHz"
    if parameter.endswith("_kappa") or parameter.endswith("_nonlin"):
        return 1e6, "MHz"
    if parameter.endswith("_charge_line_limited_t1"):
        return 1e-6, "us"
    return 1.0, ""


def save_convergence(
    folder: Path, analysis, targets, params, parameter, title=""
) -> list[str]:
    """Write ``convergence.json`` and ``convergence.png``; the paths."""
    folder = Path(folder)
    data = convergence(analysis, targets, params, parameter)
    (folder / CONVERGENCE_JSON).write_text(
        json.dumps(data, indent=4) + "\n", encoding="utf-8"
    )
    written = [str(folder / CONVERGENCE_JSON)]
    if data["iterations"]:
        plot_convergence(data, folder / CONVERGENCE_PNG, title)
        written.append(str(folder / CONVERGENCE_PNG))
    return written


def save_pictures(folder: Path, pictures: list[dict]) -> list[str]:
    """Write each picture as ``pictures/<n>_<title>.png``, with its
    caption in ``pictures/captions.txt``; the paths."""
    target = Path(folder) / PICTURES
    target.mkdir(exist_ok=True)
    written, captions = [], []
    for n, picture in enumerate(pictures, 1):
        slug = re.sub(r"[^0-9A-Za-z]+", "_", picture.get("title", "")).strip("_")
        path = target / f"{n}_{slug or 'study'}.png"
        path.write_bytes(base64.b64decode(picture["png"]))
        written.append(str(path))
        captions.append(f"{path.name}: {picture.get('caption', '')}")
    if captions:
        (target / "captions.txt").write_text(
            "\n".join(captions) + "\n", encoding="utf-8"
        )
    return written


@contextlib.contextmanager
def study_log(folder: Path) -> Iterator[Path]:
    """Every log line written while inside, also into ``study.log``."""
    path = Path(folder) / STUDY_LOG
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
    )
    handler.setLevel(logging.INFO)
    root = logging.getLogger()
    level = root.level
    root.addHandler(handler)
    if root.level > logging.INFO or root.level == logging.NOTSET:
        root.setLevel(logging.INFO)
    try:
        yield path
    finally:
        root.removeHandler(handler)
        root.setLevel(level)
        handler.close()
