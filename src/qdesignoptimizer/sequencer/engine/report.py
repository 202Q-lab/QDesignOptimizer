"""A report of completed studies: a brief one, a full one, and a table
to paste into Notion (or a spreadsheet).

Which studies: any completed ones, picked by their keys; by default the
latest run of every study (a requeued copy ``<id>_rN`` counts as the
same study). A study still running is left out.

What a backend adds, when it has these (the QDO backend does):

* ``report_targets(folder)``: each target of the study with its values
  per iteration (``{"parameter", "target", "values", "design_var",
  "unit", "design_values"}``);
* ``report_value(parameter, value)``: a value as it is read, with unit;
* ``report_table(runs, fabrication)``: ``{"title", "columns", "rows",
  "notes"}``, a row per qubit, say, with columns for what fabrication
  gave and what follows from it;
* ``report_pictures(folder)``: ``[{"title", "path", "caption"}]``, the
  pictures the full report shows of a study (QDO: the model as Ansys drew
  it in the last iteration, and the junctions in the final GDS). The
  written report copies them into its ``pictures/`` folder;
* ``report_simulation(folder)``: ``{"runs": [...], "passes"}``, the
  simulations a study ran (``HFSS eigenmode``, ``Q3D capacitance: ...``),
  and ``"layout"``: why it has no final layout, when it has none;
* ``report_inputs(folder)``: the design variables a study started from.

The last two, with the targets and what each study wrote back, make the
full report's study flow (``flow``): which study ran which simulation to
tune what, and which study started from the values another wrote.

What fabrication gave is kept in the project's ``fabrication.yaml``
(``{name: value}``, e.g. ``qubit_1: 8.1 kOhm``), so every report after
uses it.
"""

from __future__ import annotations

import re
import shutil
import time
from pathlib import Path
from typing import Any

import yaml

from . import flow as fl
from . import state as st
from .job import parse_completed

FABRICATION_FILE = "fabrication.yaml"
REPORTS = "reports"
FLOW_PNG = "study_flow.png"
FLOW_TEXT = (
    "Each card is a study, in the order they ran. Its header's colour is "
    "the kind of simulation, with its runs, passes and iterations; the "
    "picture is the model as Ansys drew it in the last iteration; the "
    "table has each design variable it tuned, first -> final value "
    "(\u2713: written back to the store), and the parameter it was tuned "
    "for, with how far that ended from its target. An arrow carries the "
    "design variables a study started from as an earlier study wrote them "
    "back."
)

_REQUEUED = re.compile(r"_r\d+$")


def study_of(run_id: str) -> str:
    """The study a run is of: a requeued copy ``<id>_rN`` is ``<id>``'s."""
    return _REQUEUED.sub("", run_id)


def completed_runs(workspace) -> list[dict]:
    """Every completed study that has finished, newest first:
    ``{"key", "id", "study", "status", "started", "project"}``."""
    out = []
    for project, path in workspace.completed():
        result = st.result_of(path)
        status = result.get("status", "")
        if status in ("", st.RUNNING, st.COMMITTING):
            continue
        run_id = parse_completed(path.name)[0]
        name = workspace.name_of(project)
        out.append(
            {
                "key": workspace.key(name, path.name),
                "id": run_id,
                "study": study_of(run_id),
                "task": result.get("study", ""),
                "status": status,
                "started": result.get("started", ""),
                "project": name,
            }
        )
    return out


def default_selection(runs: list[dict]) -> list[str]:
    """The keys of the latest run of every study (``runs`` newest first)."""
    seen: set[tuple[str, str]] = set()
    keys = []
    for run in runs:
        study = (run["project"], run["study"])
        if study not in seen:
            seen.add(study)
            keys.append(run["key"])
    return keys


def resolve(workspace, wanted: list[str]) -> list[str]:
    """Keys of completed runs from what a person typed: a completed
    folder's key, or a study id (its latest run).

    Raises:
        ValueError: For one that names no completed run.
    """
    runs = completed_runs(workspace)
    by_key = {r["key"]: r for r in runs}
    keys = []
    for item in wanted:
        if item in by_key:
            keys.append(item)
            continue
        latest = [
            r
            for r in runs
            if item in (r["id"], r["study"])
            or item in (f"{r['project']}/{r['id']}", f"{r['project']}/{r['study']}")
        ]
        if not latest:
            raise ValueError(f"no completed run of {item!r}")
        keys.append(latest[0]["key"])
    return list(dict.fromkeys(keys))


def collect(workspace, keys: list[str]) -> list[dict]:
    """What the report says of each run, in the order of ``keys``."""
    out = []
    for key in keys:
        project, local = workspace.locate(key)
        folder = project.queue.completed_dir / local
        if not folder.is_dir():
            raise ValueError(f"{key} is not a completed study")
        result = st.result_of(folder)
        try:
            job = yaml.safe_load((folder / "job.yaml").read_text("utf-8")) or {}
        except (OSError, yaml.YAMLError):
            job = {}
        backend = project.backend
        targets = _hook(backend, "report_targets", folder)
        pictures = _hook(backend, "report_pictures", folder)
        simulation = _hook_dict(backend, "report_simulation", folder)
        ran_with = _hook_dict(backend, "report_inputs", folder)
        out.append(
            {
                "key": key,
                "project": workspace.name_of(project),
                "folder": folder,
                "id": parse_completed(folder.name)[0],
                "task": result.get("study", ""),
                "subject": result.get("subject", ""),
                "status": result.get("status", "?"),
                "message": result.get("message", ""),
                "started": result.get("started", ""),
                "finished": result.get("finished", ""),
                "campaign": result.get("campaign", "") or job.get("campaign", ""),
                "sequence": (job.get("sequence") or {}),
                "requeued_from": job.get("requeued_from") or "",
                "settings": job.get("settings") or {},
                "notes": job.get("notes") or "",
                "why": job.get("why") or result.get("why", ""),
                "added_by": result.get("added_by", "") or job.get("added_by", ""),
                "outputs": result.get("outputs") or {},
                "updates": result.get("updates") or {},
                "committed": result.get("committed") or {},
                "checks": result.get("checks") or {},
                "targets": targets,
                "pictures": pictures,
                "simulation": simulation,
                "ran_with": ran_with,
                "_backend": backend,
            }
        )
    return out


# -- fabrication -----------------------------------------------------------


def fabrication_path(project) -> Path:
    """Where a project keeps what fabrication gave."""
    return Path(project.out).parent / FABRICATION_FILE


def load_fabrication(project) -> dict[str, str]:
    """``{name: value}`` from the project's fabrication file ({} if none)."""
    try:
        data = yaml.safe_load(fabrication_path(project).read_text("utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items() if v not in (None, "")}


def save_fabrication(project, values: dict[str, str]) -> Path:
    """Write the project's fabrication file (empty values dropped)."""
    kept = {k: str(v).strip() for k, v in values.items() if str(v or "").strip()}
    path = fabrication_path(project)
    path.write_text(
        "# What fabrication gave, per junction's qubit: Ej (GHz), the\n"
        "# normal resistance Rn (kOhm) or Lj (nH) -- the unit says which.\n"
        "# Optional: gap_ueV, the superconducting gap for Rn (180 for Al).\n"
        + (yaml.safe_dump(kept, sort_keys=True) if kept else ""),
        encoding="utf-8",
    )
    return path


def parse_fabrication(text: str) -> dict[str, str]:
    """``name: value`` lines (as the GUI's box holds them) to a dict.

    Raises:
        ValueError: For a line that is not ``name: value``.
    """
    out = {}
    for n, line in enumerate(str(text or "").splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name, sep, value = line.partition(":")
        if not sep or not name.strip():
            raise ValueError(f"line {n}: {line!r} is not 'name: value'")
        out[name.strip()] = value.strip()
    return out


# -- writing ---------------------------------------------------------------


def table(runs: list[dict], fabrication: dict[str, str]) -> dict | None:
    """The backend's table for these runs (the first backend that makes
    one), or None."""
    for backend in dict.fromkeys(r["_backend"] for r in runs):
        hook = getattr(backend, "report_table", None)
        if hook is None:
            continue
        # Plain data: the backend may be in a worker process.
        own = [
            {
                **{k: v for k, v in r.items() if k != "_backend"},
                "folder": str(r["folder"]),
            }
            for r in runs
            if r["_backend"] is backend
        ]
        made = hook(own, dict(fabrication))
        if made and made.get("rows"):
            return made
    return None


def markdown_table(columns: list[str], rows: list[list[Any]]) -> str:
    """A Markdown table: Notion makes a table of it when pasted."""

    def cell(value: Any) -> str:
        return (
            str("" if value is None else value).replace("|", "\\|").replace("\n", " ")
        )

    lines = [
        "| " + " | ".join(cell(c) for c in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def tsv_table(columns: list[str], rows: list[list[Any]]) -> str:
    """Tab-separated: a spreadsheet (or a Notion database) takes it as cells."""

    def cell(value: Any) -> str:
        return str("" if value is None else value).replace("\t", " ").replace("\n", " ")

    return "\n".join("\t".join(cell(v) for v in line) for line in [columns, *rows])


def brief(runs: list[dict], made: dict | None = None) -> str:
    """One line per study: where it ended against its targets."""
    lines = [_header(runs), ""]
    rows = []
    for run in runs:
        rows.append(
            [
                run["id"],
                run["started"][:16],
                run["status"],
                _worst_line(run) or _outputs_line(run),
                "yes" if run["committed"] else "no",
            ]
        )
    lines.append(
        markdown_table(["Study", "Started", "Status", "Result", "Written"], rows)
    )
    if made:
        lines += ["", f"## {made.get('title') or 'Table'}", "", _table_md(made)]
    return "\n".join(lines) + "\n"


def full(
    runs: list[dict],
    made: dict | None = None,
    pictures: bool = True,
    flow: bool = True,
    flow_image: str | None = None,
) -> str:
    """Every study in full: what it was, every target per iteration,
    the design variables it ran with and suggests, its checks, and its
    pictures (``pictures``; their paths as the runs have them).

    First, with ``flow``, the study flow (``flow.graph``): as a Mermaid
    diagram, after the picture ``flow_image`` links to when given."""
    lines = [_header(runs)]
    if made:
        lines += ["", f"## {made.get('title') or 'Table'}", "", _table_md(made)]
    if flow and runs:
        lines += ["", "## Study flow", "", FLOW_TEXT]
        if flow_image:
            lines += ["", f"![Study flow]({_link(flow_image)})"]
        graph = fl.graph(runs)
        lines += ["", "```mermaid", fl.mermaid(graph), "```"]
        if graph["left_out"]:
            lines += ["", f"*{fl.left_out_text(graph)}*"]
    for run in runs:
        lines += ["", f"## {run['id']}", ""]
        facts = [
            ("Study", run["task"]),
            ("Subject", run["subject"]),
            ("Project", run["project"]),
            ("Campaign", run["campaign"]),
            (
                "Sequence",
                (run["sequence"].get("id") or "")
                + (
                    f" ({run['sequence'].get('name')})"
                    if run["sequence"].get("name")
                    else ""
                ),
            ),
            (
                "Status",
                (
                    f"{run['status']}: {run['message']}"
                    if run["message"]
                    else run["status"]
                ),
            ),
            ("Ran", f"{run['started']} to {run['finished']}"),
            ("Settings", ", ".join(f"{k}={v}" for k, v in run["settings"].items())),
            ("Queued by", run["added_by"]),
            ("Why", run["why"]),
            ("Notes", run["notes"]),
            ("Folder", str(run["folder"])),
            ("No final layout", (run.get("simulation") or {}).get("layout", "")),
        ]
        lines += [f"- **{name}:** {value}" for name, value in facts if value]
        if run["targets"]:
            lines += ["", "### Targets", ""]
            rows = []
            for t in run["targets"]:
                values = [v for v in t.get("values") or []]
                rows.append(
                    [
                        t.get("parameter", ""),
                        _value(run, t.get("parameter", ""), t.get("target")),
                        _value(run, t.get("parameter", ""), _last(values)),
                        _off(_last(values), t.get("target")),
                        " → ".join(
                            _value(run, t.get("parameter", ""), v) for v in values
                        ),
                    ]
                )
            lines.append(
                markdown_table(
                    ["Parameter", "Target", "Reached", "Off", "Per iteration"], rows
                )
            )
            lines += ["", "### Design variables", ""]
            rows = []
            for t in run["targets"]:
                design = [v for v in t.get("design_values") or [] if v is not None]
                var = t.get("design_var", "")
                unit = t.get("unit", "")
                rows.append(
                    [
                        var,
                        f"{_num(design[0])} {unit}" if design else "",
                        f"{_num(design[-1])} {unit}" if design else "",
                        "yes" if var in run["committed"] else "no",
                    ]
                )
            lines.append(
                markdown_table(["Design variable", "First", "Final", "Written"], rows)
            )
        elif run["outputs"]:
            lines += ["", "### Results", ""]
            lines.append(
                markdown_table(
                    ["Parameter", "Value"],
                    [[k, _value(run, k, v)] for k, v in sorted(run["outputs"].items())],
                )
            )
        if run["checks"]:
            lines += ["", "### Checks", ""]
            lines += [
                f"- {name}: {'ok' if (c or {}).get('ok') else 'failed'}"
                + (
                    f" -- {(c or {}).get('message')}"
                    if (c or {}).get("message")
                    else ""
                )
                for name, c in run["checks"].items()
            ]
        if pictures and run.get("pictures"):
            lines += ["", "### Pictures"]
            for picture in run["pictures"]:
                lines += [
                    "",
                    f"**{picture['title']}**",
                    "",
                    f"![{picture['title']}]({_link(picture['path'])})",
                ]
                if picture.get("caption"):
                    lines += ["", f"*{picture['caption']}*"]
    return "\n".join(lines) + "\n"


def write(project, runs: list[dict], made: dict | None) -> Path:
    """Write the brief and full reports, the study flow's picture
    (``study_flow.png``) and the table into ``out/reports/<now>/``; the
    folder."""
    folder = Path(project.out) / REPORTS / time.strftime("%Y-%m-%dT%H-%M-%S")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "report_brief.md").write_text(brief(runs, made), encoding="utf-8")
    image = None
    if runs:
        try:
            image = fl.draw(fl.graph(runs), folder / FLOW_PNG).name
        except Exception:  # noqa: BLE001 -- the Mermaid diagram stays
            image = None
    (folder / "report.md").write_text(
        full(_copy_pictures(runs, folder), made, flow_image=image),
        encoding="utf-8",
    )
    if made:
        (folder / "table.md").write_text(_table_md(made) + "\n", encoding="utf-8")
        (folder / "table.tsv").write_text(
            tsv_table(made["columns"], made["rows"]) + "\n", encoding="utf-8"
        )
    return folder


# -- helpers ---------------------------------------------------------------


def _hook(backend, name: str, folder: Path) -> list:
    """A backend's optional report hook for a run folder ([] without it,
    or when it fails: the report goes on without)."""
    hook = getattr(backend, name, None)
    if hook is None:
        return []
    try:
        return list(hook(folder) or [])
    except Exception:  # noqa: BLE001 -- the report goes on without
        return []


def _hook_dict(backend, name: str, folder: Path) -> dict:
    """A backend's optional report hook that gives a dict ({} without
    it, or when it fails)."""
    hook = getattr(backend, name, None)
    if hook is None:
        return {}
    try:
        return dict(hook(folder) or {})
    except Exception:  # noqa: BLE001 -- the report goes on without
        return {}


def _copy_pictures(runs: list[dict], folder: Path) -> list[dict]:
    """The runs with their pictures copied to ``<folder>/pictures/<run>/``
    and linked from there, so the report folder stands on its own."""
    out = []
    for run in runs:
        copied = []
        for picture in run.get("pictures") or []:
            source = Path(picture["path"])
            if not source.is_file():
                continue
            target = folder / copied_path(run, picture)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            copied.append({**picture, "path": target.relative_to(folder)})
        out.append({**run, "pictures": copied})
    return out


def copied_path(run: dict, picture: dict) -> Path:
    """Where ``write`` puts a copy of a run's picture, relative to the
    report's folder: ``pictures/<run folder>/<file>``, or
    ``<n>_<file>`` when another of the run's pictures has the same file
    name (``n``: its place in the run's pictures)."""
    name = Path(picture["path"]).name
    pictures = run.get("pictures") or []
    if sum(Path(p["path"]).name == name for p in pictures) > 1:
        name = f"{pictures.index(picture) + 1}_{name}"
    return Path("pictures") / Path(run["folder"]).name / name


def _link(path: Any) -> str:
    """A Markdown link target: forward slashes, spaces escaped."""
    return Path(path).as_posix().replace(" ", "%20")


def _header(runs: list[dict]) -> str:
    projects = sorted({r["project"] for r in runs})
    return (
        f"# Simulation report\n\n{time.strftime('%Y-%m-%d %H:%M')} · "
        f"{len(runs)} {'study' if len(runs) == 1 else 'studies'} · "
        + ", ".join(projects)
    )


def _table_md(made: dict) -> str:
    out = markdown_table(made["columns"], made["rows"])
    notes = made.get("notes") or []
    return out + ("\n\n" + "\n".join(f"- {n}" for n in notes) if notes else "")


def _worst_line(run: dict) -> str:
    """``5 targets, worst resonator_1_freq -2.50 %``: the brief's result."""
    offs = []
    for t in run["targets"]:
        reached, goal = _last(t.get("values") or []), t.get("target")
        try:
            offs.append((abs(float(reached) / float(goal) - 1), t))
        except (TypeError, ValueError, ZeroDivisionError):
            continue
    if not offs:
        return ""
    _, worst = max(offs, key=lambda o: o[0])
    count = len(run["targets"])
    return (
        f"{count} target{'s' if count != 1 else ''}, worst "
        f"{worst.get('parameter', '')} "
        f"{_off(_last(worst.get('values') or []), worst.get('target'))}"
    )


def _outputs_line(run: dict) -> str:
    return (
        "; ".join(
            f"{k} {_value(run, k, v)}" for k, v in sorted(run["outputs"].items())
        )[:400]
        or run["message"]
    )


def _value(run: dict, parameter: str, value: Any) -> str:
    if value is None:
        return ""
    hook = getattr(run["_backend"], "report_value", None)
    if hook is not None:
        try:
            return str(hook(parameter, value))
        except Exception:  # noqa: BLE001 -- shown plain instead
            pass
    return _num(value)


def _num(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.6g}"
    return str(value)


def _last(values: list) -> Any:
    found = [v for v in values if v is not None]
    return found[-1] if found else None


def _off(value: Any, target: Any) -> str:
    try:
        if value is None or not target:
            return ""
        return f"{(float(value) / float(target) - 1) * 100:+.2f} %"
    except (TypeError, ValueError):
        return ""
