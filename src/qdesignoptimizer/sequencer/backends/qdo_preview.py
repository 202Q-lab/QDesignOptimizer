"""Pictures of what a ministudy simulates: one per solver run.

A MiniStudy can be several simulations (``runs``): the HFSS eigenmode
model of its own ``qiskit_component_names`` (unless it runs capacitance
studies only), then one Q3D model per chained capacitance study, each
of that study's own components. Each is drawn from the project's built
design (``QDOProject.design``) with only those components: metal, the
etched gaps, junctions, and the airbridges that solver puts in.

**Airbridges** are not geometry in the design. They are qiskit-metal
wire bonds that the Ansys renderer adds when it draws the model, on
every path row of the design flagged ``hfss_wire_bonds`` (and
``subtract``), spaced by the renderer's ``wb_threshold``, shifted by
``wb_offset`` and spanning ``wb_size`` times the row's width.
``bond_positions`` places them exactly as
``QAnsysRenderer.auto_wirebonds`` does.

Which renderer's settings apply depends on the run (``bond_settings``):
QDO hands the MiniStudy's ``hfss_wire_bond_*`` to the HFSS renderer of
an eigenmode solve; a capacitance study is drawn by the Q3D renderer,
whose options are whatever the design set on it (TsTv2 sets them in
``add_sim_airbridges``). The caption says which was used. The HFSS run
also gets the bridges a 202Q component lists itself in
``get_air_bridge_coordinates()``, as QDO's ``DesignAnalysis`` adds them
(not for a study with ``surface_properties``).

Imports matplotlib and shapely only inside functions (the backends
import nothing heavy at module scope).
"""

from __future__ import annotations

import base64
import io
import math
from typing import Any, Iterable

#: Colours: metal, etched gap, junction, airbridge.
METAL, GAP, JUNCTION, BRIDGE = "#2a78d6", "#e9eef4", "#d62728", "#eb6834"


_UNITS_MM = {"nm": 1e-6, "um": 1e-3, "mm": 1.0, "cm": 10.0, "m": 1e3}


def _mm(value: Any) -> float:
    """A length in qiskit-metal's design unit (mm): ``"500um"`` -> 0.5; a
    bare number is mm already (as the renderer assumes)."""
    if isinstance(value, (int, float)):
        return float(value)
    import re

    match = re.fullmatch(r"\s*([-+0-9.eE]+)\s*([a-z]*)\s*", str(value))
    if not match or match.group(2) not in _UNITS_MM | {"": 1.0}:
        raise ValueError(f"cannot read {value!r} as a length")
    return float(match.group(1)) * _UNITS_MM.get(match.group(2), 1.0)


def study_components(ministudy) -> list[str]:
    """Every component a MiniStudy simulates, its chained studies'
    included, in order, once each."""
    names = _names(ministudy)
    for chained in getattr(ministudy, "capacitance_matrix_studies", None) or []:
        names += _names(chained)
    return list(dict.fromkeys(names))


def _names(study) -> list[str]:
    names = getattr(study, "qiskit_component_names", []) or []
    return list(dict.fromkeys(str(n) for n in names))


def runs_eigenmode(ministudy) -> bool:
    """Whether QDO solves the HFSS eigenmode model of a MiniStudy: only
    with modes to track and not ``run_capacitance_studies_only``
    (``DesignAnalysis``)."""
    return bool(getattr(ministudy, "modes", None)) and not getattr(
        ministudy, "run_capacitance_studies_only", False
    )


def runs(ministudy) -> list[dict]:
    """The simulations a MiniStudy is, in the order QDO runs them.

    Returns:
        One ``{"title", "solver", "components"}`` per run: ``solver``
        is ``"hfss"`` for the eigenmode model of the MiniStudy's own
        components, ``"q3d"`` for each chained capacitance study.
    """
    out = []
    if runs_eigenmode(ministudy):
        out.append(
            {
                "title": "HFSS eigenmode",
                "solver": "hfss",
                "components": _names(ministudy),
            }
        )
    chained = getattr(ministudy, "capacitance_matrix_studies", None) or []
    for n, study in enumerate(chained, 1):
        number = f" {n}" if len(chained) > 1 else ""
        out.append(
            {
                "title": f"Q3D capacitance{number}: {type(study).__name__}",
                "solver": "q3d",
                "components": _names(study),
            }
        )
    return out


def custom_bridges(design, names: list[str], ministudy) -> list[tuple]:
    """The bridges 202Q components list themselves, as segments
    ``((x0, y0), (x1, y1))``; none where QDO would not add them (a
    study with ``surface_properties``)."""
    if getattr(ministudy, "surface_properties", None):
        return []
    out = []
    for name in names:
        if name not in design.components:
            continue
        hook = getattr(design.components[name], "get_air_bridge_coordinates", None)
        for start, end in (hook() if hook else None) or []:
            out.append(
                (
                    (float(start[0]), float(start[1])),
                    (float(end[0]), float(end[1])),
                )
            )
    return out


def bond_positions(
    rows: Iterable, size: float, threshold: Any, offset: Any
) -> list[tuple]:
    """Where the renderer puts wire bonds: ``(x, y, ux, uy, span)`` per
    bond -- its centre, the unit vector across the line, its length.

    Mirrors ``QAnsysRenderer.auto_wirebonds``: per straight stretch of a
    flagged path longer than ``threshold``, ``n = length // threshold``
    bonds, evenly at ``k / (n + 1)`` of the stretch (k = 1..n), moved
    along it by ``offset``, each ``width * size`` long.
    """
    threshold, offset = _mm(threshold), _mm(offset)
    out = []
    for geometry, width in rows:
        coords = list(geometry.coords)
        for (x0, y0), (x1, y1) in zip(coords, coords[1:]):
            dx, dy = x1 - x0, y1 - y0
            length = math.hypot(dx, dy)
            if threshold <= 0 or length <= threshold:
                continue
            count = int(length // threshold)
            ux, uy = dx / length, dy / length
            step_x, step_y = dx / (count + 1), dy / (count + 1)
            for k in range(1, count + 1):
                out.append(
                    (
                        x0 + step_x * k + ux * offset,
                        y0 + step_y * k + uy * offset,
                        -uy,
                        ux,
                        float(width) * float(size),
                    )
                )
    return out


def bond_settings(design, ministudy, solver: str = "") -> tuple:
    """``(size, threshold, offset, which renderer)`` a run uses.

    Args:
        solver: ``"hfss"`` or ``"q3d"``; empty means the MiniStudy's
            own solver (Q3D when it solves no eigenmodes).
    """
    if not solver:
        solver = "hfss" if runs_eigenmode(ministudy) else "q3d"
    if solver == "q3d":
        renderer = (getattr(design, "renderers", None) or {}).get("q3d")
        options = getattr(renderer, "options", None) or {}
        return (
            options.get("wb_size", 5),
            options.get("wb_threshold", "400um"),
            options.get("wb_offset", "0um"),
            "Q3D renderer",
        )
    return (
        getattr(ministudy, "hfss_wire_bond_size", 3),
        getattr(ministudy, "hfss_wire_bond_threshold", "300um"),
        getattr(ministudy, "hfss_wire_bond_offset", "0um"),
        "HFSS renderer, with the MiniStudy's settings",
    )


def pictures(design, ministudy, note: str = "", bonds=None) -> list[dict[str, Any]]:
    """One picture per run of the study (``runs``), each of its own
    components with its own solver's bridges; ``note`` goes on every
    caption. Each is ``render``'s dict plus its ``"title"``.

    Args:
        bonds: ``bonds(solver)`` -> ``(size, threshold, offset, label)``
            or None, for a project whose solve sets the bridges itself
            rather than taking them from the study (``render``).
    """
    out = []
    for run in runs(ministudy):
        picture = render(
            design,
            ministudy,
            title=run["title"],
            note=note,
            names=run["components"],
            solver=run["solver"],
            bonds=bonds(run["solver"]) if bonds else None,
        )
        picture["title"] = run["title"]
        out.append(picture)
    return out


def render(
    design,
    ministudy,
    title: str = "",
    note: str = "",
    names: list[str] | None = None,
    solver: str = "",
    bonds: tuple | None = None,
) -> dict[str, Any]:
    """One run's picture, as a base64 PNG with a caption.

    Args:
        names: The components drawn; default the MiniStudy's own.
        solver: ``"hfss"`` or ``"q3d"``, whose bridges are drawn (see
            ``bond_settings``); the HFSS run also gets the components'
            own bridges (``custom_bridges``).
        bonds: ``(size, threshold, offset, label)`` to draw in place of
            ``bond_settings``'s: what the project's solve really uses.

    Returns:
        ``{"png": <base64>, "caption": str, "components": [...],
        "missing": [...], "bridges": int}``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if names is None:
        names = _names(ministudy)
    if not solver:
        solver = "hfss" if runs_eigenmode(ministudy) else "q3d"
    present = [n for n in names if n in design.components]
    missing = [n for n in names if n not in design.components]
    ids = {design.components[n].id for n in present}
    tables = design.qgeometry.tables

    def rows(kind):
        table = tables.get(kind)
        if table is None or table.empty:
            return table.iloc[0:0] if table is not None else None
        return table[table["component"].isin(ids)]

    shapes = []  # (geometry, colour, z)
    poly, path, junction = rows("poly"), rows("path"), rows("junction")
    for table, is_path in ((poly, False), (path, True)):
        if table is None:
            continue
        for _, row in table.iterrows():
            geometry = row["geometry"]
            if is_path:
                geometry = geometry.buffer(
                    float(row["width"]) / 2, cap_style=2, join_style=2
                )
            gap = bool(row.get("subtract", False))
            shapes.append((geometry, GAP if gap else METAL, 1 if gap else 2))

    bridged = []
    if path is not None and "hfss_wire_bonds" in path:
        flagged = path[
            (path["hfss_wire_bonds"] == True)  # noqa: E712
            & (path["subtract"] == True)  # noqa: E712
        ]
        size, threshold, offset, source = bonds or bond_settings(
            design, ministudy, solver
        )
        bridged = bond_positions(
            ((r["geometry"], r["width"]) for _, r in flagged.iterrows()),
            size,
            threshold,
            offset,
        )
        note = "; ".join(
            n
            for n in (
                f"airbridges as the {source} draws them (size {size}, "
                f"every {threshold}, offset {offset})",
                note,
            )
            if n
        )

    custom = custom_bridges(design, present, ministudy) if solver == "hfss" else []
    if custom:
        note = "; ".join(
            n
            for n in (
                f"{len(custom)} bridge(s) the components list themselves "
                "(get_air_bridge_coordinates)",
                note,
            )
            if n
        )
    for (x0, y0), (x1, y1) in custom:
        length = math.hypot(x1 - x0, y1 - y0) or 1e-3
        bridged.append(
            (
                (x0 + x1) / 2,
                (y0 + y1) / 2,
                (x1 - x0) / length,
                (y1 - y0) / length,
                length,
            )
        )

    fig, ax = plt.subplots(figsize=(9, 7), dpi=130)
    bounds = None
    for geometry, colour, z in sorted(shapes, key=lambda s: s[2]):
        for part in getattr(geometry, "geoms", [geometry]):
            if part.is_empty:
                continue
            xs, ys = part.exterior.xy
            ax.fill(xs, ys, color=colour, linewidth=0, zorder=z)
            for hole in part.interiors:
                hx, hy = hole.xy
                ax.fill(hx, hy, color="white", linewidth=0, zorder=z)
            b = part.bounds
            if bounds is None:
                bounds = b
            else:
                bounds = (
                    min(bounds[0], b[0]),
                    min(bounds[1], b[1]),
                    max(bounds[2], b[2]),
                    max(bounds[3], b[3]),
                )
    if junction is not None:
        for _, row in junction.iterrows():
            xs, ys = row["geometry"].xy
            ax.plot(xs, ys, color=JUNCTION, linewidth=2.5, zorder=4)
    for x, y, nx, ny, span in bridged:
        half = span / 2
        ax.plot(
            [x - nx * half, x + nx * half],
            [y - ny * half, y + ny * half],
            color=BRIDGE,
            linewidth=1.6,
            zorder=5,
            solid_capstyle="round",
        )
        ax.plot(
            [x - nx * half, x + nx * half],
            [y - ny * half, y + ny * half],
            "o",
            color=BRIDGE,
            markersize=2.5,
            zorder=5,
        )
    if bounds is not None:
        pad = 0.04 * max(bounds[2] - bounds[0], bounds[3] - bounds[1], 0.05)
        ax.set_xlim(bounds[0] - pad, bounds[2] + pad)
        ax.set_ylim(bounds[1] - pad, bounds[3] + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.grid(color="#eeeeee", linewidth=0.6, zorder=0)
    ax.set_title(title or "the study's components", fontsize=11)
    ax.legend(
        handles=[
            Patch(color=METAL, label="metal"),
            Patch(color=GAP, label="etched gap"),
            Line2D([], [], color=JUNCTION, linewidth=2.5, label="junction"),
            Line2D(
                [],
                [],
                color=BRIDGE,
                marker="o",
                markersize=3,
                label=f"airbridge ({len(bridged)})",
            ),
        ],
        loc="upper right",
        fontsize=8,
        framealpha=0.9,
    )
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    caption = (
        f"{len(present)} component(s): {', '.join(present) or 'none'}; "
        f"{len(bridged)} airbridge(s)"
        + (f"; not in the design: {', '.join(missing)}" if missing else "")
        + (f". {note}" if note else "")
    )
    return {
        "png": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "caption": caption,
        "components": present,
        "missing": missing,
        "bridges": len(bridged),
    }


def unbridged_note(design, names: list[str]) -> str | None:
    """Which of the study's CPWs the mask leaves without bridges."""
    off = [
        n
        for n in names
        if n in design.components
        and "hfss_wire_bonds" in design.components[n].options
        and not design.components[n].options.get("hfss_wire_bonds")
    ]
    return f"without airbridges: {', '.join(off)}" if off else None
