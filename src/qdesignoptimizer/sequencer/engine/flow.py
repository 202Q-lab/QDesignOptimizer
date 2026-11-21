"""The study flow: which study ran which simulation, which parameters it
estimated to tune which design variables, and which later study started
from the values it wrote back.

Built only from what the runs recorded (``report.collect``), so the same
runs always give the same diagram:

* per run: its ``simulation`` (the backend's ``report_simulation``), its
  ``targets`` (parameter -> design variable, per iteration), ``ran_with``
  (the design variables it started from, ``report_inputs``) and
  ``committed`` (what it wrote back to the store);
* an arrow A -> B for the design variables B started from that A wrote
  back -- A the last run before B that wrote that variable, and only
  when B started from exactly the value A wrote (a value changed by hand
  in between is not A's);
* the variables a run tunes that no run here wrote come from "the store
  before these studies".

``mermaid`` writes the graph as a Mermaid flowchart (GitHub, VS Code and
Notion draw it); ``draw`` as a PNG, with matplotlib.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import Any

START = "start"
#: Design variables named on an arrow before "+N more".
SHOWN_ON_ARROW = 6

_NUMBER = re.compile(r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*([A-Za-z]*)\s*$")


def graph(runs: list[dict]) -> dict[str, Any]:
    """``{"nodes": [...], "edges": [...], "left_out": [...]}``: the store
    before the runs, then the runs in the order they started; ``edges``
    as ``{"source", "target", "variables"}``, sorted; ``left_out`` the
    ids of the runs not shown (``superseded``)."""
    ordered = sorted(runs, key=lambda r: (str(r.get("started") or ""), str(r["id"])))
    dropped = superseded(ordered)
    left_out = [r["id"] for r in ordered if id(r) in dropped]
    ordered = [r for r in ordered if id(r) not in dropped]
    nodes: list[dict[str, Any]] = [
        {
            "key": START,
            "title": "Store before these studies",
            "lines": [],
            "status": "",
        }
    ]
    edges: dict[tuple[str, str], list[str]] = {}
    for index, run in enumerate(ordered):
        key = f"s{index + 1}"
        nodes.append(_node(key, run))
        tuned = {t.get("design_var") for t in run.get("targets") or []}
        tuned.discard(None)
        tuned.discard("")
        ran_with = run.get("ran_with") or {}
        for name in sorted(set(ran_with) | tuned):
            source = _source(name, ran_with.get(name), ordered[:index])
            if source is None:
                # From the store as it was, or changed by hand: only the
                # ones this run tunes are worth an arrow.
                if name not in tuned:
                    continue
                source_key = START
            else:
                source_key = f"s{ordered.index(source) + 1}"
            edges.setdefault((source_key, key), []).append(name)
    return {
        "nodes": nodes,
        "edges": [
            {"source": s, "target": t, "variables": sorted(v)}
            for (s, t), v in sorted(edges.items(), key=lambda e: (_order(e[0])))
        ],
        "left_out": left_out,
    }


def left_out_text(flow: dict[str, Any]) -> str:
    """One line naming the runs the flow leaves out, or ""."""
    if not flow.get("left_out"):
        return ""
    return (
        "Not shown (replaced by a later run of the same study, and no "
        "study shown started from its values): " + ", ".join(flow["left_out"])
    )


def superseded(ordered: list[dict]) -> set[int]:
    """The runs a flow leaves out (as ``id()`` of the run dicts): each
    run that a later run of the same study replaced and that no run still
    shown started from -- a failed attempt run again, or values written
    back and overwritten before any study used them. The last run of each
    study stays, and so does every run a shown run built on. ``ordered``:
    the runs in the order they started."""
    dropped: set[int] = set()
    for index in range(len(ordered) - 1, -1, -1):
        run = ordered[index]
        later = ordered[index + 1 :]
        replaced = any(_same_study(run, other) for other in later)
        if not replaced:
            continue
        used = False
        for position, other in enumerate(later, index + 1):
            if id(other) in dropped:
                continue
            ran_with = other.get("ran_with") or {}
            if any(
                _source(name, value, ordered[:position]) is run
                for name, value in ran_with.items()
            ):
                used = True
                break
        if not used:
            dropped.add(id(run))
    return dropped


def mermaid(flow: dict[str, Any]) -> str:
    """The graph as a Mermaid flowchart (top to bottom)."""
    lines = ["flowchart TD"]
    for node in flow["nodes"]:
        text = "<br/>".join(
            [f"<b>{_escape(node['title'])}</b>"] + [_escape(x) for x in node["lines"]]
        )
        shape = ('(["', '"])') if node["key"] == START else ('["', '"]')
        lines.append(f"    {node['key']}{shape[0]}{text}{shape[1]}")
    for edge in flow["edges"]:
        label = _escape(_arrow_label(edge["variables"], "<br/>"))
        lines.append(f'    {edge["source"]} -->|"{label}"| {edge["target"]}')
    failed = [n["key"] for n in flow["nodes"] if _failed(n)]
    if failed:
        lines.append("    classDef failed stroke:#c62828,stroke-dasharray:5 3;")
        lines.append(f"    class {','.join(failed)} failed;")
    return "\n".join(lines)


#: Header colour of a study's card, by the kind of simulation it ran.
KIND_COLOURS = {
    "HFSS eigenmode": "#2f5d9e",
    "Q3D capacitance": "#2e7d5b",
    "HFSS eigenmode + Q3D capacitance": "#5b4a9e",
}


def draw(flow: dict[str, Any], path: Path) -> Path:
    """The graph as a PNG at ``path``: a card per study, top to bottom in
    the order they ran -- its kind of simulation (the header's colour),
    the model as Ansys drew it in its last iteration, and the design
    variables it tuned from first to final value, each with the parameter
    it was tuned for and how far that ended from its target. An arrow per
    hand-over, labelled with the design variables it carries."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    char = 0.06  # inches a 7 pt monospace character takes
    row_h, head_h, pad, gap = 0.19, 0.62, 0.14, 0.55
    image_w, image_h = 3.6, 2.25  # Ansys pictures are 1600 x 1000
    left = 0.2
    nodes = flow["nodes"]
    keys = [n["key"] for n in nodes]
    tables = {n["key"]: _table_rows(n) for n in nodes}
    text_w = max(
        [char * len(r) for rows in tables.values() for r in rows]
        + [0.075 * len(n["title"]) - image_w for n in nodes]
        + [3.0]
    )
    width = image_w + 0.3 + text_w + 0.2

    heights = []
    for node in nodes:
        if node["key"] == START:
            heights.append(0.42)
            continue
        body = max(image_h, row_h * len(tables[node["key"]]))
        heights.append(head_h + body + 2 * pad)
    skips = [keys.index(e["target"]) - keys.index(e["source"]) for e in flow["edges"]]
    lanes = _lanes(
        [(keys.index(e["source"]), keys.index(e["target"])) for e in flow["edges"]]
    )
    gaps = [gap] * len(keys)
    for edge, skip in zip(flow["edges"], skips):
        if skip == 1:
            count = len(_arrow_label(edge["variables"], "\n").split("\n"))
            index = keys.index(edge["source"])
            gaps[index] = max(gaps[index], 0.14 * count + 0.3)
    widest_label = max(
        [
            max(len(x) for x in _arrow_label(e["variables"], "\n").split("\n"))
            for e in flow["edges"]
        ]
        or [10]
    )
    fig_w = (
        left
        + width
        + 0.35
        + LANE_STEP * max(list(lanes.values()) + [0])
        + char * widest_label
        + 0.4
    )
    note = "\n".join(
        textwrap.wrap(left_out_text(flow), max(40, int((fig_w - 2 * left) / char)))
    )
    total = (
        sum(heights)
        + sum(gaps[:-1])
        + 0.4
        + (0.14 * (note.count("\n") + 1) + 0.3 if note else 0)
        + 0.4  # the legend
    )
    legend_y = sum(heights) + sum(gaps[:-1]) + 0.2 + 0.3
    fig = plt.figure(figsize=(fig_w, total), dpi=130)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(total, 0)
    ax.axis("off")

    boxes: dict[str, tuple[float, float]] = {}  # key -> (top, bottom)
    y = 0.2
    for node, height, below in zip(nodes, heights, gaps):
        boxes[node["key"]] = (y, y + height)
        if node["key"] == START:
            ax.add_patch(
                FancyBboxPatch(
                    (left, y),
                    width,
                    height,
                    boxstyle="round,pad=0,rounding_size=0.2",
                    facecolor="#eeeeee",
                    edgecolor="#4a5a75",
                    linewidth=1.0,
                )
            )
            ax.text(
                left + width / 2,
                y + height / 2,
                node["title"],
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold",
            )
            y += height + below
            continue
        _card(ax, node, tables[node["key"]], left, y, width, height, mpimg)
        y += height + below

    placed: list[tuple[float, float, float, float]] = []  # label boxes
    for number, (edge, skip) in enumerate(zip(flow["edges"], skips)):
        label = _arrow_label(edge["variables"], "\n")
        if skip == 1:
            x = left + image_w / 2 + 0.15
            top, bottom = boxes[edge["source"]][1], boxes[edge["target"]][0]
            ax.add_patch(
                FancyArrowPatch(
                    (x, top),
                    (x, bottom),
                    arrowstyle="-|>",
                    mutation_scale=14,
                    color="#4a5a75",
                    linewidth=1.4,
                )
            )
            ax.text(x + 0.12, (top + bottom) / 2, label, va="center", fontsize=6.5)
            continue
        # Past the cards in between: round the right-hand side.
        x = left + width
        side = x + 0.35 + LANE_STEP * lanes[number]
        y0 = boxes[edge["source"]][1] - 0.15
        y1 = boxes[edge["target"]][0] + 0.3
        ax.plot([x, side, side], [y0, y0, y1], color="#7a869a", linewidth=1.0)
        ax.add_patch(
            FancyArrowPatch(
                (side, y1),
                (x, y1),
                arrowstyle="-|>",
                mutation_scale=12,
                color="#7a869a",
            )
        )
        lines = label.split("\n")
        label_w = 0.056 * max(len(x) for x in lines) + 0.1
        label_h = 0.14 * len(lines) + 0.06
        label_y = _free_spot(
            placed, side + 0.08, label_w, label_h, (y0 + y1) / 2, y0, y1
        )
        placed.append((side + 0.08, side + 0.08 + label_w, label_y, label_h))
        ax.text(
            side + 0.08,
            label_y,
            label,
            va="center",
            fontsize=6.5,
            zorder=4,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
        )
    # Legend: the header colours, then how a card reads.
    x = left
    kinds = sorted({n.get("kind") or "" for n in nodes if n["key"] != START})
    for kind in kinds:
        ax.add_patch(
            FancyBboxPatch(
                (x, legend_y - 0.07),
                0.14,
                0.14,
                boxstyle="square,pad=0",
                facecolor=KIND_COLOURS.get(kind, "#5f6b7a"),
                edgecolor="none",
            )
        )
        label = kind or "simulation not recorded"
        ax.text(x + 0.2, legend_y, label, va="center", fontsize=7.5)
        x += 0.2 + char * 1.2 * len(label) + 0.3
    ax.text(
        x,
        legend_y,
        "✓ written back to the store   · dashed red: failed",
        va="center",
        fontsize=7.5,
        color="#555555",
    )
    if note:
        ax.text(left, total - 0.15, note, va="bottom", fontsize=7, color="#555555")
    fig.savefig(path)
    plt.close(fig)
    return Path(path)


#: Inches between the lanes on the right that arrows skipping cards take.
LANE_STEP = 0.3


def _lanes(spans: list[tuple[int, int]]) -> dict[int, int]:
    """A lane per arrow that skips cards (``spans``: its source and target
    card), numbered from the cards outwards, so that two arrows passing
    the same card never share one: the shorter arrows first, each in the
    innermost lane free over its whole span. Keyed by the arrow's
    position in ``spans``; arrows to the next card take none."""
    taken: list[list[tuple[int, int]]] = []
    lanes: dict[int, int] = {}
    order = sorted(
        (i for i, (a, b) in enumerate(spans) if b - a > 1),
        key=lambda i: (spans[i][1] - spans[i][0], spans[i][0], i),
    )
    for i in order:
        a, b = spans[i]
        for lane, used in enumerate(taken):
            if all(b <= c or d <= a for c, d in used):
                used.append((a, b))
                lanes[i] = lane
                break
        else:
            taken.append([(a, b)])
            lanes[i] = len(taken) - 1
    return lanes


def _free_spot(placed, x, w, h, want, low, high) -> float:
    """A label's centre height near ``want``, between ``low`` and
    ``high``, where it overlaps no label already ``placed``."""

    def clear(y: float) -> bool:
        return all(
            x + w <= x0
            or x1 <= x
            or y + h / 2 <= yc - ph / 2
            or yc + ph / 2 <= y - h / 2
            for x0, x1, yc, ph in placed
        )

    step = 0.1
    for n in range(int((high - low) / step) + 1):
        for y in (want + n * step, want - n * step):
            if low + h / 2 <= y <= high - h / 2 and clear(y):
                return y
    return want


def _card(ax, node, rows, left, y, width, height, mpimg) -> None:
    """One study's card: a header in its simulation's colour, the model
    picture on the left, what it tuned on the right."""
    from matplotlib.patches import FancyBboxPatch

    failed = _failed(node)
    colour = KIND_COLOURS.get(node.get("kind") or "", "#5f6b7a")
    head_h, pad = 0.62, 0.14
    image_w, image_h = 3.6, 2.25
    ax.add_patch(
        FancyBboxPatch(
            (left, y),
            width,
            height,
            boxstyle="round,pad=0,rounding_size=0.08",
            facecolor="#ffffff",
            edgecolor="#c62828" if failed else colour,
            linestyle="--" if failed else "-",
            linewidth=1.6,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (left, y),
            width,
            head_h,
            boxstyle="round,pad=0,rounding_size=0.08",
            facecolor=colour,
            edgecolor="none",
        )
    )
    ax.text(
        left + 0.14,
        y + 0.1,
        node["title"],
        va="top",
        fontsize=9,
        fontweight="bold",
        color="white",
    )
    ax.text(
        left + 0.14,
        y + 0.36,
        node.get("subtitle", ""),
        va="top",
        fontsize=7.5,
        color="white",
    )
    status = node.get("status") or ""
    ax.text(
        left + width - 0.14,
        y + 0.1,
        status.upper(),
        va="top",
        ha="right",
        fontsize=8,
        fontweight="bold",
        color="#ffd2d2" if failed else "white",
    )
    top = y + head_h + pad
    image = node.get("image")
    shown = False
    if image and Path(image).is_file():
        try:
            picture = _crop(mpimg.imread(str(image)))
            if picture is None:
                raise ValueError("a blank picture")
            rows_px, cols_px = picture.shape[:2]
            # Fitted into the picture's place, its proportions kept.
            scale = min(image_w / cols_px, image_h / rows_px)
            w, h = cols_px * scale, rows_px * scale
            x0 = left + 0.15 + (image_w - w) / 2
            y0 = top + (image_h - h) / 2
            ax.imshow(
                picture,
                extent=(x0, x0 + w, y0 + h, y0),
                aspect="auto",
                zorder=2,
            )
            shown = True
        except Exception:  # noqa: BLE001 -- the card goes on without
            shown = False
    if not shown:
        ax.add_patch(
            FancyBboxPatch(
                (left + 0.15, top),
                image_w,
                image_h,
                boxstyle="round,pad=0,rounding_size=0.04",
                facecolor="#f2f2f2",
                edgecolor="#cccccc",
            )
        )
        ax.text(
            left + 0.15 + image_w / 2,
            top + image_h / 2,
            "no model picture\n(none kept for this study)",
            ha="center",
            va="center",
            fontsize=7,
            color="#888888",
        )
    x = left + 0.15 + image_w + 0.3
    for n, text in enumerate(rows):
        ax.text(
            x,
            top + n * 0.19,
            text,
            va="top",
            fontsize=7,
            family="monospace",
            fontweight="bold" if n == 0 else "normal",
            color="#333333" if n else "#4a5a75",
        )


def _crop(picture):
    """An Ansys picture cut to the model: the white margin around it
    trimmed, Ansys's logo (top right) and axis icon (bottom left) left
    out of the reckoning. None when there is nothing else in it (a blank
    picture)."""
    import numpy as np

    pixels = np.asarray(picture)
    rgb = pixels[..., :3].astype(float)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    ink = (rgb < 0.93).any(axis=-1)
    rows, cols = ink.shape
    ink[: int(rows * 0.1), int(cols * 0.8) :] = False  # the logo
    ink[int(rows * 0.8) :, : int(cols * 0.15)] = False  # the axis icon
    found_rows = np.flatnonzero(ink.any(axis=1))
    found_cols = np.flatnonzero(ink.any(axis=0))
    if not len(found_rows) or not len(found_cols):
        return None
    margin = int(0.02 * max(rows, cols))
    r0 = max(found_rows[0] - margin, 0)
    r1 = min(found_rows[-1] + margin + 1, rows)
    c0 = max(found_cols[0] - margin, 0)
    c1 = min(found_cols[-1] + margin + 1, cols)
    return pixels[r0:r1, c0:c1]


def _table_rows(node: dict) -> list[str]:
    """A card's text: a header row, then per design variable tuned its
    first -> final value, ✓ when written back, and the parameter it was
    tuned for with how far that ended from its target."""
    tuned = node.get("tuned") or []
    if node["key"] == START:
        return []
    if not tuned:
        return ["(no targets recorded)"]
    name_w = max(len("design variable"), *(len(t["variable"]) for t in tuned))
    change_w = max(len("first -> final"), *(len(t["change"]) for t in tuned))
    rows = [
        f"{'design variable':<{name_w}}  {'first -> final':<{change_w}}    "
        "tuned for (off target)"
    ]
    for t in tuned:
        rows.append(
            f"{t['variable']:<{name_w}}  {t['change']:<{change_w}}  "
            f"{'✓' if t['written'] else ' '} "
            f" {t['parameter']}" + (f" ({t['off']})" if t["off"] else "")
        )
    if not node.get("wrote"):
        rows.append("")
        rows.append("wrote nothing back")
    return rows


# -- helpers ---------------------------------------------------------------


def _node(key: str, run: dict) -> dict[str, Any]:
    """A run's box: what it was and ran, then parameter -> variable."""
    simulation = run.get("simulation") or {}
    iterations = max((len(t.get("values") or []) for t in run["targets"]), default=0)
    ran = " + ".join(simulation.get("runs") or []) or "simulation not recorded"
    facts = [ran]
    if simulation.get("passes"):
        facts.append(f"{simulation['passes']} passes")
    if iterations:
        facts.append(f"{iterations} iteration{'s' if iterations != 1 else ''}")
    lines = [
        f"{run.get('task') or ''} · {run.get('status') or ''}",
        " · ".join(facts),
    ]
    committed = run.get("committed") or {}
    for target in run.get("targets") or []:
        var = _short(target.get("design_var") or "")
        written = " ✓" if var and target.get("design_var") in committed else ""
        off = _off((target.get("values") or [None])[-1], target.get("target"))
        lines.append(
            f"{target.get('parameter', '')} → {var}{written}"
            + (f"  ({off} off)" if off else "")
        )
    if not run.get("committed"):
        lines.append("wrote nothing back")
    tuned = []
    for target in run.get("targets") or []:
        values = [v for v in target.get("design_values") or [] if v is not None]
        unit = target.get("unit") or ""
        tuned.append(
            {
                "variable": _short(target.get("design_var") or ""),
                "change": (
                    f"{_num(values[0])} -> {_num(values[-1])} {unit}".rstrip()
                    if values
                    else ""
                ),
                "written": target.get("design_var") in committed,
                "parameter": target.get("parameter", ""),
                "off": _off((target.get("values") or [None])[-1], target.get("target")),
            }
        )
    kinds = []
    for name in simulation.get("runs") or []:
        kind = str(name).split(":")[0].rstrip("0123456789 ")
        if kind not in kinds:
            kinds.append(kind)
    return {
        "key": key,
        "title": run["id"],
        "lines": lines,
        "status": run.get("status") or "",
        "kind": " + ".join(kinds),
        "subtitle": " · ".join([t for t in [run.get("task")] if t] + facts),
        "image": _model_picture(run),
        "tuned": tuned,
        "wrote": bool(committed),
    }


def _model_picture(run: dict) -> str | None:
    """The model as Ansys drew it in the study's last iteration: its
    first picture kept under ``ansys/`` (HFSS before Q3D)."""
    for picture in run.get("pictures") or []:
        picture_path = Path(picture.get("path") or "")
        if picture_path.parent.name == "ansys" or picture_path.name.startswith(
            ("hfss_", "q3d_")
        ):
            return str(picture_path)
    return None


def _num(value: Any) -> str:
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return str(value)


def _source(name: str, value: Any, before: list[dict]) -> dict | None:
    """The last run before that wrote ``name``, when ``value`` is what
    it wrote."""
    for run in reversed(before):
        committed = run.get("committed") or {}
        if name in committed:
            return run if _same(committed[name], value) else None
    return None


def _same_study(a: dict, b: dict) -> bool:
    """Runs of one study: the same task, project and campaign (a
    requeued copy ``<id>_rN`` included)."""
    return bool(a.get("task")) and (
        a.get("task"),
        a.get("project"),
        a.get("campaign"),
    ) == (
        b.get("task"),
        b.get("project"),
        b.get("campaign"),
    )


def _same(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return False
    if str(a).replace(" ", "") == str(b).replace(" ", ""):
        return True
    ma, mb = _NUMBER.match(str(a)), _NUMBER.match(str(b))
    if not (ma and mb) or ma.group(2) != mb.group(2):
        return False
    x, y = float(ma.group(1)), float(mb.group(1))
    return abs(x - y) <= 1e-9 * max(abs(x), abs(y), 1e-300)


def _off(value: Any, target: Any) -> str:
    """How far the last iteration ended from the target, relative."""
    try:
        if value is None or not target:
            return ""
        return f"{(float(value) / float(target) - 1) * 100:+.1f} %"
    except (TypeError, ValueError):
        return ""


def _short(name: str) -> str:
    return name[len("design_var_") :] if name.startswith("design_var_") else name


def _arrow_label(variables: list[str], sep: str) -> str:
    names = [_short(v) for v in variables]
    shown = names[:SHOWN_ON_ARROW]
    if len(names) > SHOWN_ON_ARROW:
        shown.append(f"+{len(names) - SHOWN_ON_ARROW} more")
    return sep.join(shown)


def _failed(node: dict) -> bool:
    return node.get("status") not in ("", "done")


def _escape(text: str) -> str:
    return str(text).replace('"', "#quot;").replace("|", "#124;")


def _order(key: tuple[str, str]) -> tuple[int, int]:
    def index(k: str) -> int:
        return 0 if k == START else int(k[1:])

    return index(key[0]), index(key[1])
