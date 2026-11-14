"""The page and its parts, as pure functions of plain data."""

from __future__ import annotations

import json
from typing import Any

from dash import dcc, html

from . import form_layout, theme

POLL_MS = 2000


def page(title: str) -> html.Div:
    """The static skeleton; callbacks fill every part."""
    return html.Div(
        id="page-root",
        className="qs-page",
        children=[
            dcc.Store(id="snapshot"),
            # {"kind": "queue"|"completed"|"draft", "key"}
            dcc.Store(id="selected"),
            # The measurement plan popup: None (closed), {"screen": "start"} or
            # {"key": <draft key>}.
            dcc.Store(id="draft"),
            dcc.Store(id="rev", data=0),
            # What the details pane's editors held when last drawn or
            # saved: a selection change while they differ asks first, as
            # moving to another study redraws the pane and drops edits.
            dcc.Store(id="details-orig"),
            dcc.Store(id="pending-select"),
            dcc.Store(id="message"),
            # Written by assets/queue_drag.js: the order after a drop, and
            # whether a drag is under way (the refresh pauses then).
            dcc.Store(id="drag-order"),
            dcc.Store(id="dragging", data=False),
            dcc.Interval(id="poll", interval=POLL_MS),
            dcc.ConfirmDialog(id="confirm-start"),
            html.Div(
                className="qs-header",
                children=[
                    html.Span(f"Queue: {title}", className="qs-title"),
                    html.Span(id="project-path", className="qs-muted"),
                ],
            ),
            html.Div(id="runner-banner"),
            html.Div(
                className="qs-bar",
                children=[
                    html.Button(
                        "Check queue",
                        id="btn-check",
                        className="qs-btn",
                        title="rebuild every queued study from its files; " "no solve",
                    ),
                    html.Button(
                        "Start",
                        id="btn-start",
                        className="qs-btn qs-btn-primary",
                    ),
                    html.Button(
                        "Stop after this study",
                        id="btn-stop",
                        className="qs-btn qs-btn-danger",
                    ),
                    html.Button("Reload catalog", id="btn-reload", className="qs-btn"),
                ],
            ),
            html.Div(id="message-view", className="qs-message"),
            html.Div(
                className="qs-main",
                children=[
                    html.Div(
                        children=[
                            html.Div(
                                className="qs-card",
                                children=[
                                    html.H3("Queue", className="qs-h"),
                                    html.Div(id="queue-table"),
                                    html.Div(
                                        className="qs-bar",
                                        children=[
                                            dcc.Dropdown(
                                                id="add-choice",
                                                placeholder="Add ...",
                                                style={"minWidth": "380px"},
                                            ),
                                            html.Button(
                                                "Add to queue",
                                                id="btn-add",
                                                className="qs-btn",
                                                title="a study goes straight "
                                                "in; a measurement plan opens the "
                                                "measurement plan popup first",
                                            ),
                                            html.Button(
                                                "New measurement plan...",
                                                id="btn-new-plan",
                                                className="qs-btn " "qs-btn-primary",
                                                title="put a measurement plan "
                                                "together, edit each study, "
                                                "then launch it",
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                            html.Div(
                                className="qs-card",
                                style={"marginTop": "16px"},
                                children=[
                                    html.H3("Completed", className="qs-h"),
                                    html.Div(id="completed-table"),
                                ],
                            ),
                        ]
                    ),
                    html.Div(
                        id="details-card",
                        className="qs-card",
                        children=[
                            html.Div(id="details-problems"),
                            html.Div(id="details"),
                            dcc.Loading(
                                html.Div(id="preview-view"),
                                type="dot",
                            ),
                            html.Pre(
                                id="inspect",
                                className="qs-pre",
                                style={"display": "none"},
                            ),
                            html.H4("Runner log", style={"margin": "12px 0 4px"}),
                            html.Pre(id="log", className="qs-pre"),
                        ],
                    ),
                ],
            ),
            html.Div(
                id="draft-modal",
                className="qs-modal",
                style={"display": "none"},
                children=html.Div(
                    className="qs-modal-panel",
                    children=[
                        # How a launch going on in the background stands;
                        # its own callback, so it can refresh without
                        # redrawing the fields above it.
                        html.Div(id="draft-status"),
                        html.Div(id="draft-body"),
                    ],
                ),
            ),
        ],
    )


def chip(status: str) -> html.Span:
    """A status chip in the status's own colours."""
    fg, bg = theme.STATUS_COLORS.get(status, ("#37474f", "#eceff1"))
    return html.Span(status, className="qs-chip", style={"color": fg, "background": bg})


def _title(words: dict, key: str, fallback: str) -> str:
    word = words.get(key, fallback)
    return word[:1].upper() + word[1:]


def runner_banner(snap: dict) -> Any:
    """One line on the runner."""
    runner = snap.get("runner") or {}
    status = runner.get("status", "idle")
    colour = theme.RUN_COLORS.get(status, theme.MUTED)
    parts: list[Any] = [html.B(f"Runner {status}", style={"color": colour})]
    if runner.get("alive") and runner.get("current"):
        parts.append(f"  -- now on {runner['current']}")
    if runner.get("stop_requested"):
        parts.append("  -- stop requested")
    if runner.get("message"):
        parts.append(f"  ({runner['message']})")
    if not runner.get("alive") and runner.get("finished"):
        parts.append(html.Span(f"  ended {runner['finished']}", className="qs-muted"))
    if runner.get("elsewhere"):
        parts.append(
            html.Span(
                f"  -- {runner['elsewhere']}",
                style={"color": theme.RUN_COLORS.get("running", theme.MUTED)},
            )
        )
    problems = snap.get("catalog_problems") or []
    if problems:
        # A template or measurement plan that cannot be read is missing from
        # the Add menu; say why rather than let it vanish.
        return html.Div(
            [
                html.Div(parts),
                html.Div(
                    className="qs-problems",
                    children=[
                        html.B("Not in the Add menu: "),
                        html.Ul([html.Li(p) for p in problems[:8]]),
                    ],
                ),
            ]
        )
    return html.Div(parts)


def queue_table(snap: dict, selected: dict | None) -> Any:
    """One row per queued study, in run order; rows can be dragged."""
    words = snap.get("words") or {}
    rows = snap.get("rows") or []
    multi = bool(snap.get("multi"))
    head = html.Tr(
        [
            html.Th("#"),
            *([html.Th("Project")] if multi else []),
            html.Th(_title(words, "plan", "measurement plan")),
            html.Th(_title(words, "subject", "subject")),
            html.Th(_title(words, "task", "task")),
            html.Th("Settings"),
            html.Th("Notes"),
            html.Th(""),
        ]
    )
    body = []
    picked = (
        (selected or {}).get("key") if (selected or {}).get("kind") == "queue" else None
    )
    for row in rows:
        problems = row.get("problems") or []
        note = (
            html.Span(
                f"cannot run: {problems[0]}",
                style={"color": theme.STATUS_COLORS["failed"][0]},
            )
            if problems
            else html.Span(row.get("notes", ""), className="qs-muted")
        )
        if row.get("hold"):
            note = html.Span(
                [
                    html.B("on hold", style={"color": theme.RUN_COLORS["held"]}),
                    " -- the runner stops here. ",
                    note,
                ]
            )
        body.append(
            html.Tr(
                className="qs-row",
                style=(
                    {"background": theme.SELECTED_BG} if row["id"] == picked else {}
                ),
                draggable="true",
                **{"data-step": row["id"]},
                children=[
                    html.Td(str(row["n"]), className="qs-muted"),
                    *(
                        [html.Td(row.get("project", ""), className="qs-small")]
                        if multi
                        else []
                    ),
                    html.Td(
                        row.get("plan", ""),
                        title=row.get("plan_name", ""),
                    ),
                    html.Td(row.get("subject", "")),
                    # Selecting is on this cell, not the row: a click on a
                    # row's own buttons must not also select it -- two
                    # callbacks writing ``selected`` at once make Dash drop
                    # the older response whole, message and all.
                    html.Td(
                        [
                            row.get("label", ""),
                            html.Br(),
                            html.Span(row["id"], className="qs-small"),
                        ],
                        id={"type": "qrow", "index": row["id"]},
                        className="qs-pick",
                        n_clicks=0,
                    ),
                    html.Td(row.get("settings", ""), className="qs-small"),
                    html.Td(note),
                    html.Td(_actions(row, row["n"] == 1, row["n"] == len(rows))),
                ],
            )
        )
    if not body:
        body = [
            html.Tr(
                html.Td(
                    "The queue is empty. Add a study or a " "measurement plan below.",
                    colSpan=8 if multi else 7,
                    className="qs-muted",
                )
            )
        ]
    return html.Table(
        className="qs-table", children=[html.Thead(head), html.Tbody(body)]
    )


def _actions(row: dict, first: bool, last: bool) -> Any:
    def icon(action, text, title, disabled=False):
        return html.Button(
            text,
            id={"type": "act", "action": action, "index": row["id"]},
            className="qs-icon",
            title=title,
            n_clicks=0,
            disabled=disabled,
        )

    return html.Span(
        [
            icon("up", "↑", "move up", first),
            icon("down", "↓", "move down", last),
            icon(
                "hold",
                "▶" if row.get("hold") else "⏸",
                (
                    "release: let the runner reach it"
                    if row.get("hold")
                    else "hold: the runner stops here"
                ),
            ),
            icon(
                "remove",
                "✕",
                "take out of the queue (kept in " "out/removed/)",
            ),
        ]
    )


def completed_table(snap: dict, selected: dict | None) -> Any:
    """The latest completed studies, newest first."""
    rows = snap.get("completed") or []
    multi = bool(snap.get("multi"))
    picked = (
        (selected or {}).get("key")
        if (selected or {}).get("kind") == "completed"
        else None
    )
    body = []
    for row in rows:
        body.append(
            html.Tr(
                className="qs-row",
                style=(
                    {"background": theme.SELECTED_BG} if row["name"] == picked else {}
                ),
                children=[
                    html.Td(row.get("started", ""), className="qs-small"),
                    *(
                        [html.Td(row.get("project", ""), className="qs-small")]
                        if multi
                        else []
                    ),
                    html.Td(chip(row.get("status", "?"))),
                    html.Td(
                        [
                            row.get("id", ""),
                            html.Br(),
                            html.Span(
                                row.get("message", "")[:120],
                                className="qs-small",
                            ),
                        ],
                        id={"type": "crow", "index": row["name"]},
                        className="qs-pick",
                        n_clicks=0,
                    ),
                    html.Td(
                        html.Button(
                            "Requeue",
                            id={"type": "requeue", "index": row["name"]},
                            className="qs-btn qs-btn-small",
                            n_clicks=0,
                        )
                    ),
                ],
            )
        )
    if not body:
        body = [
            html.Tr(
                html.Td(
                    "Nothing has run yet.",
                    colSpan=5 if multi else 4,
                    className="qs-muted",
                )
            )
        ]
    head = html.Tr(
        [
            html.Th("Started"),
            *([html.Th("Project")] if multi else []),
            html.Th("Status"),
            html.Th("Study"),
            html.Th(""),
        ]
    )
    return html.Table(
        className="qs-table", children=[html.Thead(head), html.Tbody(body)]
    )


def duration(minutes: float) -> str:
    """``5 s``, ``3 min``, ``1.5 h``: a time per iteration, readable."""
    if minutes < 1:
        return f"{minutes * 60:.0f} s"
    if minutes < 90:
        return f"{minutes:g} min"
    return f"{minutes / 60:.1f} h"


def plural(word: str) -> str:
    """``study`` -> ``studies``, ``experiment`` -> ``experiments``."""
    if word.endswith("y") and word[-2:-1] not in tuple("aeiou"):
        return word[:-1] + "ies"
    return word + "s"


def add_options(snap: dict) -> list[dict]:
    """The Add menu: every study, then every measurement plan."""
    words = snap.get("words") or {}
    task_word = _title(words, "task", "task")
    plan_word = _title(words, "plan", "measurement plan")
    options = []
    for item in snap.get("catalog") or []:
        if item["kind"] == "task":
            minutes = item.get("minutes")
            eta = f", ~{duration(minutes)}/iteration" if minutes else ""
            label = f"{task_word}: {item['label']} ({item['key']}{eta})"
        else:
            label = (
                f"{plan_word}: {item['label']} "
                f"({item['count']} {plural(words.get('task', 'task'))})"
            )
        options.append({"label": label, "value": f"{item['kind']}:{item['key']}"})
    return options


def details(data: dict | None, words: dict) -> Any:
    """The selected study: its job.yaml fields and its files."""
    if not data:
        return html.Div(
            "Select a study to see and edit its files.", className="qs-muted"
        )
    editable = data["kind"] in ("queue", "draft")
    job = data.get("job") or {}
    task_word = words.get("task", "task")
    head = [
        html.H3(
            (
                f"{data['key'].rsplit('/', 1)[-1]} (in the measurement plan draft)"
                if data["kind"] == "draft"
                else data["key"]
            ),
            style={"margin": "0 0 2px", "wordBreak": "break-all"},
        ),
        html.Div(
            f"{task_word} {job.get(task_word, job.get('task', ''))}"
            f"  --  {job.get('generated_by', '')}",
            className="qs-muted",
        ),
    ]
    parts = head
    if editable:
        parts.append(_job_form(job, data.get("settings_keys") or ()))
    else:
        parts.append(_result_view(data))
    tabs = []
    for spec in data.get("files") or []:
        tabs.append(
            dcc.Tab(
                label=spec["label"],
                value=spec["name"],
                # One pane per file, so creating or deleting this file
                # redraws this pane alone, not the editors of the others.
                children=[
                    html.Div(
                        id={"type": "file-pane", "name": spec["name"]},
                        children=form_layout.file_form(spec, editable),
                    )
                ],
            )
        )
    if data["kind"] == "completed":
        tabs.append(
            dcc.Tab(
                label="result.json",
                value="result.json",
                children=[
                    html.Pre(
                        json.dumps(data.get("result") or {}, indent=1, default=str),
                        className="qs-pre qs-file",
                    )
                ],
            )
        )
    parts.append(
        dcc.Tabs(
            id="file-tabs",
            value=(data.get("files") or [{"name": "result.json"}])[0]["name"],
            children=tabs,
            className="qs-tabs",
        )
    )
    buttons: list[Any] = []
    if editable:
        buttons.append(html.Button("Inspect", id="btn-inspect", className="qs-btn"))
    else:
        buttons += [
            html.Button(
                "Requeue fresh",
                id="btn-requeue",
                className="qs-btn qs-btn-primary",
                title="queue a copy as it was queued: design variables "
                "and parameter targets it left to the project are "
                "pulled again when it starts",
            ),
            html.Button(
                "Requeue exact copy",
                id="btn-requeue-exact",
                className="qs-btn",
                disabled=not data.get("has_resolved"),
                title=(
                    "queue a copy with the values this run used, "
                    "project values included, to run it again as it was"
                    if data.get("has_resolved")
                    else "this run kept no resolved/ copy of its values"
                ),
            ),
        ]
    buttons.append(
        html.Button(
            "Picture",
            id="btn-preview",
            className="qs-btn",
            title="draw only the components this study simulates, with "
            "the airbridges the solve adds (builds the design the first "
            "time: a few seconds)",
        )
    )
    task = str(job.get(words.get("task", "task"), job.get("task", "")))
    suggestion = "_".join(reversed(task.split(".", 1))).lower()
    buttons += [
        dcc.Input(
            id="tpl-name",
            value=f"{suggestion}_tpl" if suggestion else "",
            placeholder="template name",
            style={"width": "170px"},
        ),
        html.Button(
            "Save as template",
            id="btn-save-template",
            className="qs-btn",
            title="write ministudies/<name>.yaml and opt_targets/"
            "<name>.yaml from this study, for the Add menu",
        ),
    ]
    parts.append(html.Div(className="qs-bar", children=buttons))
    return html.Div(parts)


def _setting_names(job: dict, keys) -> list[str]:
    return list(keys or []) + sorted(
        k for k in (job.get("settings") or {}) if k not in (keys or [])
    )


def _setting_text(value) -> str:
    return "" if value is None else str(value)


def _job_form(job: dict, keys=()) -> Any:
    def field(label, child):
        return html.Div(className="qs-field", children=[html.Label(label), child])

    settings = job.get("settings") or {}
    rows = [
        html.Div(
            className="qs-frow",
            children=[
                html.Label(name),
                dcc.Input(
                    id={"type": "jf", "name": f"setting:{name}"},
                    value=_setting_text(settings.get(name)),
                    placeholder="project default",
                    className="qs-input",
                ),
            ],
        )
        for name in _setting_names(job, keys)
    ]
    return html.Div(
        [
            html.Div([html.B("How it is solved"), *rows], className="qs-form"),
            html.Div(
                className="qs-bar",
                children=[
                    html.Label("checks"),
                    dcc.Input(
                        id={"type": "jf", "name": "checks"},
                        value=", ".join(job.get("checks") or []),
                        style={"width": "180px"},
                    ),
                    html.Label("on failure"),
                    dcc.Dropdown(
                        id={"type": "jf", "name": "on_failure"},
                        options=["abort", "continue"],
                        value=job.get("on_failure", "abort"),
                        clearable=False,
                        style={"minWidth": "120px"},
                    ),
                    dcc.Checklist(
                        id={"type": "jf", "name": "save"},
                        options=[
                            {
                                "label": " write back on success",
                                "value": "save",
                            }
                        ],
                        value=["save"] if job.get("save") else [],
                        inline=True,
                    ),
                ],
            ),
            field(
                "notes",
                dcc.Input(
                    id={"type": "jf", "name": "notes"},
                    value=job.get("notes", ""),
                    style={"width": "100%"},
                ),
            ),
            html.Button(
                "Save job.yaml",
                id="btn-save-job",
                className="qs-btn qs-btn-primary",
            ),
        ]
    )


def _result_view(data: dict) -> Any:
    res = data.get("result") or {}
    lines = [
        html.Div(
            [chip(res.get("status", "?")), f"  {res.get('message', '')}"],
            style={"margin": "8px 0"},
        ),
        html.Div(
            f"started {res.get('started', '')}  finished " f"{res.get('finished', '')}",
            className="qs-muted",
        ),
    ]
    for key in ("committed", "updates", "outputs"):
        if res.get(key):
            lines.append(
                html.Div(
                    className="qs-field",
                    children=[
                        html.Label(key),
                        html.Pre(
                            json.dumps(res[key], indent=1, default=str),
                            className="qs-pre",
                            style={"maxHeight": "120px"},
                        ),
                    ],
                )
            )
    return html.Div(lines)


def details_problems(snap: dict, selected: dict | None) -> Any:
    """Why the selected queued study cannot run, from the latest poll.

    Kept out of ``details`` so it can refresh every poll without
    redrawing the editors under it.
    """
    if not selected or selected.get("kind") != "queue":
        return None
    row = next(
        (r for r in snap.get("rows") or [] if r["id"] == selected.get("key")),
        None,
    )
    if row is None:
        return html.Div(
            "No longer in the queue (it may have started).",
            className="qs-problems",
        )
    parts = []
    if row.get("problems"):
        parts.append(
            html.Div(
                [html.B("Cannot run yet: "), "; ".join(row["problems"])],
                className="qs-problems",
            )
        )
    if row.get("warnings"):
        parts.append(
            html.Div(
                [html.B("Warning: "), "; ".join(row["warnings"])],
                className="qs-problems",
            )
        )
    return html.Div(parts) if parts else None


def job_fields(job: dict, keys=()) -> dict:
    """The study form's field values exactly as ``_job_form`` draws them
    (what "unsaved" is measured against)."""
    settings = job.get("settings") or {}
    return {
        **{
            f"setting:{n}": _setting_text(settings.get(n))
            for n in _setting_names(job, keys)
        },
        "checks": ", ".join(job.get("checks") or []),
        "on_failure": job.get("on_failure", "abort"),
        "save": ["save"] if job.get("save") else [],
        "notes": job.get("notes", ""),
    }
