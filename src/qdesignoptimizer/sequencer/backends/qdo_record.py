"""What a QDO study keeps of its own run, besides QDO's files.

* ``convergence.json`` / ``convergence.png`` -- every target parameter
  and its design variable, iteration by iteration, read from the
  analysis's ``optimization_results`` (``system_optimized_params`` and
  ``design_variables`` per iteration);
* ``pictures/`` -- the study's pictures (``qdo_preview``), one per solver
  run, as the solve was about to draw it;
* ``study.log`` -- every log line written while the study ran;
* ``ansys/`` -- the model as Ansys drew it, each iteration
  (``hfss_<n>.png``, ``q3d_<study>_<n>.png``);
* ``layout/`` -- the design as it ran last, as it would be made
  (``final.gds``), and a close-up of each Josephson junction drawn from
  that file (``jj_<component>.png``).

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
ANSYS = "ansys"
LAYOUT = "layout"
FINAL_GDS = "final.gds"


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


# -- the study as rendered in Ansys, and as it would be made -----------------


@contextlib.contextmanager
def ansys_pictures(folder: Path) -> Iterator[Path]:
    """While inside, every DesignAnalysis not given a folder of its own
    saves a picture of each model it renders in Ansys into ``folder``:
    ``hfss_<iteration>.png``, ``q3d_<study>_<iteration>.png``."""
    from qdesignoptimizer.design_analysis import DesignAnalysis

    folder = Path(folder)
    before = DesignAnalysis.ansys_pictures_default
    DesignAnalysis.ansys_pictures_default = str(folder)
    try:
        yield folder
    finally:
        DesignAnalysis.ansys_pictures_default = before


def save_layout(folder: Path, design) -> list[str]:
    """The design as it ran last, as it would be made: ``layout/final.gds``
    (qiskit-metal's GDS renderer, with the design's own GDS options), and
    a close-up of each Josephson junction drawn from that file,
    ``layout/jj_<component>.png``; the paths."""
    target = Path(folder) / LAYOUT
    target.mkdir(exist_ok=True)
    gds = target / FINAL_GDS
    renderer = _gds_renderer(design)
    if not renderer.export_to_gds(str(gds)) or not gds.exists():
        raise RuntimeError("qiskit-metal wrote no GDS file")
    # The cheesing layers mark where holes go (or may not): not metal.
    helpers = set()
    for option in ("cheese", "no_cheese"):
        try:
            helpers.add(int(renderer.options[option]["datatype"]))
        except (KeyError, TypeError, ValueError):
            pass
    written = [str(gds)]
    captions = []
    for junction in junctions(design):
        slug = re.sub(r"[^0-9A-Za-z]+", "_", junction["component"]).strip("_")
        path = target / f"jj_{slug or 'junction'}.png"
        found = plot_junction(gds, junction, path, skip=helpers)
        written.append(str(path))
        where = (
            "drawn from final.gds"
            if found
            else "final.gds has no junction cell here (see the GDS "
            "renderer's path_filename option); the dashed line is where the "
            "design puts it"
        )
        captions.append(
            f"{path.name}: {junction['component']}, junction line "
            f"{junction['length_um']:.1f} um long, {junction['width_um']:.2f} um "
            f"wide in the design; {where}"
        )
    if captions:
        (target / "captions.txt").write_text(
            "\n".join(captions) + "\n", encoding="utf-8"
        )
    return written


def _gds_renderer(design):
    try:
        renderer = design.renderers.gds
    except (AttributeError, KeyError):
        renderer = None
    if renderer is None:
        from qiskit_metal.renderers.renderer_gds.gds_renderer import QGDSRenderer

        renderer = QGDSRenderer(design)
    return renderer


def junctions(design) -> list[dict]:
    """Each junction of the design: ``{"component", "name", "x", "y",
    "ends", "length_um", "width_um"}`` -- its line's middle and ends,
    its length and width, all in um."""
    import math

    table = design.qgeometry.tables.get("junction")
    if table is None:
        return []
    names = {c.id: name for name, c in design.components.items()}
    unit = _um_per_unit(design)
    out = []
    for _, row in table.iterrows():
        coords = list(row["geometry"].coords)
        if len(coords) < 2:
            continue
        (x0, y0), (x1, y1) = coords[0], coords[-1]
        out.append(
            {
                "component": str(names.get(row["component"], row["component"])),
                "name": str(row["name"]),
                "x": (x0 + x1) / 2 * unit,
                "y": (y0 + y1) / 2 * unit,
                "ends": [(x0 * unit, y0 * unit), (x1 * unit, y1 * unit)],
                "length_um": math.hypot(x1 - x0, y1 - y0) * unit,
                "width_um": float(row.get("width", 0) or 0) * unit,
            }
        )
    return out


def _um_per_unit(design) -> float:
    units = str(getattr(design.template_options, "units", "mm") or "mm")
    return {"mm": 1e3, "um": 1.0, "nm": 1e-3, "m": 1e6}.get(units, 1e3)


def plot_junction(
    gds: Path, junction: dict, path: Path, skip: set[int] | None = None
) -> bool:
    """A close-up of the GDS file around a junction, a window a little
    larger than the junction's line, in um from the junction's middle;
    whether the file has metal in the middle of that line (the junction
    cell is drawn in). Polygons of a datatype in ``skip`` (the cheesing
    helper layers) are left out."""
    import gdstk
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as Patch

    library = gdstk.read_gds(str(gds))
    scale = library.unit / 1e-6  # um per file unit
    polygons = [
        p
        for cell in library.top_level()
        if isinstance(cell, gdstk.Cell)
        for p in cell.get_polygons()
        if p.datatype not in (skip or set())
    ]
    cx, cy = junction["x"], junction["y"]
    half = max(junction["length_um"] * 0.9, 15.0)
    palette = ["#2a78d6", "#eb6834", "#3c9e4a", "#8e5ac8", "#c2a01f", "#d94c8a"]
    kinds: dict[tuple[int, int], str] = {}
    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=120)
    for polygon in polygons:
        points = np.asarray(polygon.points) * scale
        if (
            points[:, 0].max() < cx - half
            or points[:, 0].min() > cx + half
            or points[:, 1].max() < cy - half
            or points[:, 1].min() > cy + half
        ):
            continue
        kind = (polygon.layer, polygon.datatype)
        color = kinds.setdefault(kind, palette[len(kinds) % len(palette)])
        ax.add_patch(
            Patch(
                points - (cx, cy),
                closed=True,
                fc=color,
                ec="#333333",
                lw=0.4,
                alpha=0.75,
            )
        )
    (x0, y0), (x1, y1) = junction["ends"]
    found = _metal_at(polygons, scale, (cx, cy))
    ax.plot(
        [x0 - cx, x1 - cx],
        [y0 - cy, y1 - cy],
        color="#d62728",
        lw=1.2,
        ls="-" if found else "--",
        label="junction line (design)",
    )
    for (layer, datatype), color in kinds.items():
        ax.plot([], [], "s", color=color, label=f"layer {layer}/{datatype}")
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect("equal")
    ax.set_xlabel(f"x (um) from the junction's middle, x = {cx:.1f} um")
    ax.set_ylabel(f"y (um) from y = {cy:.1f} um")
    ax.grid(color="#eeeeee", lw=0.5)
    ax.set_title(
        f"{junction['component']}: junction in final.gds"
        + ("" if found else " (no junction cell)"),
        fontsize=9,
    )
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return found


def _metal_at(polygons, scale, point) -> bool:
    """Whether a polygon covers ``point`` (um)."""
    import numpy as np
    from matplotlib.path import Path as Shape

    return any(
        Shape(np.asarray(p.points) * scale).contains_point(point) for p in polygons
    )


def final_pictures(folder: Path) -> list[dict]:
    """The pictures a report shows of a completed study: the model as
    Ansys drew it in the last iteration (HFSS, then each Q3D study), then
    the junction close-ups from ``layout/final.gds``. ``[{"title",
    "path", "caption"}]``, only files that exist."""
    folder = Path(folder)
    out: list[dict] = []
    latest: dict[str, tuple[int, Path]] = {}
    for path in (folder / ANSYS).glob("*.png"):
        match = re.fullmatch(r"(hfss|q3d_\d+)_(\d+)", path.stem)
        if not match:
            continue
        kind, iteration = match.group(1), int(match.group(2))
        if kind not in latest or iteration > latest[kind][0]:
            latest[kind] = (iteration, path)
    for kind in sorted(latest, key=lambda k: (k != "hfss", k)):
        iteration, path = latest[kind]
        what = (
            "HFSS eigenmode model"
            if kind == "hfss"
            else f"Q3D capacitance study {kind.split('_')[1]}"
        )
        out.append(
            {
                "title": f"{what} (iteration {iteration})",
                "path": path,
                "caption": f"as rendered in Ansys in the last iteration, "
                f"{ANSYS}/{path.name}",
            }
        )
    layout = folder / LAYOUT
    captions: dict[str, str] = {}
    if (layout / "captions.txt").exists():
        for line in (layout / "captions.txt").read_text("utf-8").splitlines():
            name, _, text = line.partition(": ")
            captions[name] = text
    for path in sorted(layout.glob("jj_*.png")):
        out.append(
            {
                "title": f"Junction, {path.stem[3:]}",
                "path": path,
                "caption": captions.get(path.name, f"from {LAYOUT}/{FINAL_GDS}"),
            }
        )
    return out


def final_variables(folder: Path) -> dict | None:
    """The design variables a completed study ran its last iteration with:
    its ``design_variables.json``, updated with the last iteration's in
    QDO's results file (``<chip>_<time>.npy``). None without either."""
    import numpy as np

    folder = Path(folder)
    try:
        values = json.loads((folder / "design_variables.json").read_text("utf-8"))
    except (OSError, ValueError):
        return None
    saved = sorted(folder.glob("*.npy"), key=lambda p: p.stat().st_mtime)
    if not saved:
        return None
    try:
        data = np.load(saved[-1], allow_pickle=True)
        results = data[0]["optimization_results"]
        last = dict(results[-1]["design_variables"])
    except Exception:  # noqa: BLE001 -- not a QDO results file
        return None
    return {**values, **{k: str(v) for k, v in last.items()}}
