"""The page and its parts, as pure functions of plain data."""

from __future__ import annotations

import json
from typing import Any

from dash import dcc, html

from . import form_layout, theme

POLL_MS = 2000


def page(title: str) -> html.Div:
    """The static skeleton; callbacks fill every part.

    Left: what you make and edit -- New buttons, a browser of campaigns,
    study sequence templates and study templates, and the editor for
    whatever is selected. Right: what runs -- the runner, the queue, the
    studies on hold, the completed ones and the log.
    """
    return html.Div(
        id="page-root",
        className="qs-page",
        children=[
            dcc.Store(id="snapshot"),
            # What the editor shows: {"kind": "queue"|"hold"|"completed"|
            # "campaign"|"sequence"|"template"|"new-study"|"new-campaign"|
            # "new-sequence", "key"}
            dcc.Store(id="selected"),
            dcc.Store(id="rev", data=0),
            # What the editor's fields held when last drawn or saved: a
            # selection change while they differ asks first, as moving to
            # another study redraws the editor and drops edits.
            dcc.Store(id="details-orig"),
            # Where a click wanted to go while the editor had unsaved
            # edits: the warning asks to stay or discard them.
            dcc.Store(id="pending-select"),
            dcc.Store(id="browser-open", data=dict(BROWSER_SECTIONS)),
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
            html.Div(id="message-view", className="qs-message"),
            html.Div(
                className="qs-main",
                children=[
                    # -- left: make and edit ------------------------------
                    html.Div(
                        children=[
                            html.Div(
                                className="qs-bar",
                                children=[
                                    html.Button(
                                        "+ New campaign",
                                        id="btn-open-new-campaign",
                                        className="qs-btn",
                                        title="a folder for one question, "
                                        "with its README as the lab log",
                                    ),
                                    html.Button(
                                        "+ New study sequence",
                                        id="btn-open-new-sequence",
                                        className="qs-btn",
                                        title="a template: an ordered list "
                                        "of studies, saved in study_sequences/",
                                    ),
                                    html.Button(
                                        "+ New study",
                                        id="btn-open-new-study",
                                        className="qs-btn qs-btn-primary",
                                        title="from a template or from "
                                        "scratch; it starts on hold",
                                    ),
                                    html.Div(
                                        dcc.Dropdown(
                                            id="new-project",
                                            clearable=False,
                                            placeholder="for project ...",
                                        ),
                                        id="new-project-box",
                                        style={"display": "none"},
                                        title="the project the New buttons make things in",
                                    ),
                                ],
                            ),
                            html.Div(
                                id="browser",
                                className="qs-card",
                                style={"marginTop": "10px"},
                            ),
                            html.Div(
                                id="details-card",
                                className="qs-card",
                                style={"marginTop": "10px"},
                                children=[
                                    html.Div(
                                        id="unsaved-warning",
                                        className="qs-warning",
                                        style={"display": "none"},
                                        children=[
                                            html.Span(id="unsaved-text"),
                                            html.Div(
                                                className="qs-bar",
                                                children=[
                                                    html.Button(
                                                        "Stay",
                                                        id="btn-unsaved-stay",
                                                        className="qs-btn qs-btn-primary",
                                                    ),
                                                    html.Button(
                                                        "Discard them",
                                                        id="btn-unsaved-discard",
                                                        className="qs-btn qs-btn-danger",
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                    html.Div(id="details-problems"),
                                    # Only redrawn when something is
                                    # picked: the first study after a start
                                    # takes seconds (the worker imports the
                                    # backend), and the old one must not
                                    # look like the new one.
                                    dcc.Loading(
                                        html.Div(id="details"),
                                        type="dot",
                                        delay_show=300,
                                    ),
                                    dcc.Loading(
                                        html.Div(id="preview-view"),
                                        type="dot",
                                    ),
                                    html.Pre(
                                        id="inspect",
                                        className="qs-pre",
                                        style={"display": "none"},
                                    ),
                                ],
                            ),
                        ]
                    ),
                    # -- right: run ----------------------------------------
                    html.Div(
                        children=[
                            html.Div(id="runner-banner"),
                            html.Div(
                                className="qs-bar",
                                children=[
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
                                    html.Button(
                                        "Check queue",
                                        id="btn-check",
                                        className="qs-btn",
                                        title="build every queued study from "
                                        "its files, as the runner will, and "
                                        "list what would stop one; no solve",
                                    ),
                                    html.Button(
                                        "Reload catalog",
                                        id="btn-reload",
                                        className="qs-btn",
                                        title="read the templates and the "
                                        "adapter again",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="qs-card",
                                style={"marginTop": "10px"},
                                children=[
                                    html.H3("Queue", className="qs-h"),
                                    html.Div(id="queue-table"),
                                    html.Div(
                                        className="qs-bar",
                                        children=[
                                            dcc.Dropdown(
                                                id="add-choice",
                                                placeholder="Add a study or a study sequence ...",
                                                style={"minWidth": "300px"},
                                            ),
                                            html.Div(
                                                dcc.Dropdown(
                                                    id="add-campaign",
                                                    placeholder="no campaign",
                                                ),
                                                style={"minWidth": "180px"},
                                                title="the campaign it is " "queued in",
                                            ),
                                            html.Button(
                                                "Add to queue",
                                                id="btn-add",
                                                className="qs-btn",
                                            ),
                                        ],
                                    ),
                                    html.H3(
                                        "On hold",
                                        className="qs-h",
                                        style={"marginTop": "14px"},
                                    ),
                                    html.Div(
                                        "Not in the run order: the runner "
                                        "skips these until you queue them (▶).",
                                        className="qs-small",
                                    ),
                                    html.Div(id="held-table"),
                                ],
                            ),
                            html.Div(
                                className="qs-card",
                                style={"marginTop": "10px"},
                                children=[
                                    html.H3("Completed", className="qs-h"),
                                    html.Div(id="completed-table"),
                                ],
                            ),
                            html.Div(
                                className="qs-card",
                                style={"marginTop": "10px"},
                                children=[
                                    html.H4("Runner log", style={"margin": "0 0 4px"}),
                                    html.Pre(id="log", className="qs-pre"),
                                ],
                            ),
                        ]
                    ),
                ],
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
        # A template or study sequence that cannot be read is missing from
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
            html.Th(_title(words, "sequence", "study sequence")),
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
                        row.get("sequence", ""),
                        title=row.get("sequence_name", ""),
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
                            *_campaign_tag(row.get("campaign")),
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
                    "The queue is empty. Add a study or a " "study sequence below.",
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
            icon("park", "⏸", "put on hold: out of the run order until queued again"),
            icon(
                "remove",
                "✕",
                "take out of the queue (kept in " "out/removed/)",
            ),
        ]
    )


def held_table(snap: dict, selected: dict | None) -> Any:
    """The studies on hold: not run until moved to the queue."""
    rows = snap.get("held") or []
    multi = bool(snap.get("multi"))
    picked = (
        (selected or {}).get("key") if (selected or {}).get("kind") == "hold" else None
    )
    if not rows:
        return html.Div(
            "Nothing on hold.", className="qs-muted", style={"margin": "4px 0"}
        )
    body = []
    for row in rows:
        problems = row.get("problems") or []

        def icon(action, text, title, row=row):
            return html.Button(
                text,
                id={"type": "act", "action": action, "index": row["id"]},
                className="qs-icon",
                title=title,
                n_clicks=0,
            )

        body.append(
            html.Tr(
                className="qs-row",
                style=(
                    {"background": theme.SELECTED_BG} if row["id"] == picked else {}
                ),
                children=[
                    *(
                        [html.Td(row.get("project", ""), className="qs-small")]
                        if multi
                        else []
                    ),
                    html.Td(row.get("subject", "")),
                    html.Td(
                        [
                            row.get("label", ""),
                            html.Br(),
                            html.Span(
                                row["id"].rsplit("/", 1)[-1], className="qs-small"
                            ),
                            *_campaign_tag(row.get("campaign")),
                        ],
                        id={"type": "hrow", "index": row["id"]},
                        className="qs-pick",
                        n_clicks=0,
                    ),
                    html.Td(row.get("settings", ""), className="qs-small"),
                    html.Td(
                        html.Span(
                            f"cannot run: {problems[0]}",
                            style={"color": theme.STATUS_COLORS["failed"][0]},
                        )
                        if problems
                        else html.Span(row.get("notes", ""), className="qs-muted")
                    ),
                    html.Td(
                        html.Span(
                            [
                                icon(
                                    "unpark",
                                    "▶",
                                    "queue it: move it to the end of the queue",
                                ),
                                icon("remove", "✕", "remove (kept in out/removed/)"),
                            ]
                        )
                    ),
                ],
            )
        )
    return html.Table(className="qs-table", children=[html.Tbody(body)])


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
                            *_campaign_tag(row.get("campaign")),
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


def _campaign_tag(campaign: str | None) -> list:
    """A study's campaign as a small tag on its own line ([] for none)."""
    if not campaign:
        return []
    return [html.Br(), html.Span(f"campaign {campaign}", className="qs-tag")]


def campaign_options(snap: dict) -> list[dict]:
    """The Add bar's campaign picker: every campaign, sub-campaigns
    indented under their parent."""
    multi = bool(snap.get("multi"))
    return [
        {
            "label": (f"{c['project']} / " if multi else "") + c["key"],
            # Two projects may both have a campaign of this name.
            "value": f"{c['project']}::{c['key']}" if multi else c["key"],
        }
        for c in snap.get("campaigns") or []
    ]


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
    """The Add menu: every study, then every study sequence."""
    words = snap.get("words") or {}
    task_word = _title(words, "task", "task")
    sequence_word = _title(words, "sequence", "study sequence")
    options = []
    for item in snap.get("catalog") or []:
        if item["kind"] == "task":
            minutes = item.get("minutes")
            eta = f", ~{duration(minutes)}/iteration" if minutes else ""
            label = f"{task_word}: {item['label']} ({item['key']}{eta})"
        else:
            label = (
                f"{sequence_word}: {item['label']} "
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
    editable = data["kind"] in ("queue", "hold")
    job = data.get("job") or {}
    task_word = words.get("task", "task")
    head = [
        html.H3(
            (
                f"{data['key'].rsplit('/', 1)[-1]} (on hold)"
                if data["kind"] == "hold"
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
        parts.append(
            _job_form(job, data.get("settings_keys") or (), data.get("campaigns"))
        )
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


def _job_form(job: dict, keys=(), campaigns=None) -> Any:
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
            html.Div(
                className="qs-bar",
                children=[
                    html.Label("campaign"),
                    dcc.Dropdown(
                        id={"type": "jf", "name": "campaign"},
                        options=sorted(
                            set(campaigns or [])
                            | ({job["campaign"]} if job.get("campaign") else set())
                        ),
                        value=job.get("campaign") or None,
                        placeholder="none: the project's design variables",
                        style={"minWidth": "260px"},
                    ),
                ],
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
    """Why the selected queued (or held) study cannot run, from the
    latest poll.

    Kept out of ``details`` so it can refresh every poll without
    redrawing the editors under it.
    """
    kind = (selected or {}).get("kind")
    if kind not in ("queue", "hold"):
        return None
    rows = snap.get("rows" if kind == "queue" else "held") or []
    row = next((r for r in rows if r["id"] == (selected or {}).get("key")), None)
    if row is None:
        return html.Div(
            (
                "No longer in the queue (it may have started)."
                if kind == "queue"
                else "No longer on hold (queued or removed)."
            ),
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
        "campaign": job.get("campaign") or None,
    }


# -- the left side: browser and editor views -----------------------------------


def _pick(kind: str, key: str, text, picked: dict | None, small: str = "", depth=0):
    """One clickable line of the browser."""
    chosen = (picked or {}).get("kind") == kind and (picked or {}).get("key") == key
    return html.Div(
        [
            html.Span("\u2003" * depth),
            html.Span(text),
            html.Span(f"  {small}", className="qs-small") if small else None,
        ],
        id={"type": "brow", "kind": kind, "index": key},
        className="qs-pick qs-brow",
        style={"background": theme.SELECTED_BG} if chosen else {},
        n_clicks=0,
    )


#: The browser's sections, and which start open.
BROWSER_SECTIONS = {"campaigns": True, "sequences": True, "templates": False}


def browser(snap: dict, selected: dict | None, opened: dict | None = None) -> Any:
    """Campaigns, study sequence templates and study templates; a click
    opens one in the editor. ``opened`` says which sections are open
    (kept in a store, so a refresh does not undo a collapse)."""
    opened = {**BROWSER_SECTIONS, **(opened or {})}
    words = snap.get("words") or {}
    multi = bool(snap.get("multi"))
    campaigns = snap.get("campaigns") or []
    sequences = [c for c in snap.get("catalog") or [] if c["kind"] == "sequence"]
    tasks = [c for c in snap.get("catalog") or [] if c["kind"] == "task"]

    def section(name, title, items, empty):
        is_open = bool(opened.get(name))
        return html.Div(
            [
                html.Div(
                    html.B(f"{'▾' if is_open else '▸'} {title} ({len(items)})"),
                    id={"type": "bsec", "index": name},
                    className="qs-pick",
                    n_clicks=0,
                ),
                (
                    html.Div(items or html.Div(empty, className="qs-muted qs-small"))
                    if is_open
                    else None
                ),
            ],
            style={"marginBottom": "6px"},
        )

    return html.Div(
        [
            section(
                "campaigns",
                "Campaigns",
                [
                    _pick(
                        "campaign",
                        (f"{c['project']}::" if multi else "") + c["key"],
                        c["key"].rpartition("/")[2],
                        selected,
                        c["title"],
                        c["depth"],
                    )
                    for c in campaigns
                ],
                "none yet: + New campaign",
            ),
            section(
                "sequences",
                _title(words, "sequence", "study sequence") + " templates",
                [
                    _pick(
                        "sequence",
                        c["key"],
                        c["label"],
                        selected,
                        # The name it is saved and queued under: two
                        # templates may share a display name.
                        f"{c['key'].rpartition('/')[2]} · {c.get('count', 0)} studies"
                        + ("" if c.get("source") else ", in the adapter's code"),
                    )
                    for c in sequences
                ],
                "none yet: + New study sequence",
            ),
            section(
                "templates",
                _title(words, "task", "study") + " templates",
                [
                    _pick("template", c["key"], c["label"], selected, c["key"])
                    for c in tasks
                ],
                "none: add ministudies/*.yaml",
            ),
        ]
    )


def editor_message(text: str) -> Any:
    return html.Div(text, className="qs-muted")


def template_view(data: dict) -> Any:
    """A study template: what it is, and a button that makes a study
    from it, on hold."""
    return html.Div(
        [
            html.H3(data["label"], className="qs-h"),
            html.Div(
                f"{data['key']}  --  {data.get('family') or 'study template'}",
                className="qs-muted",
            ),
            html.Pre(data.get("summary") or "", className="qs-pre qs-light"),
            _new_study_bar(data.get("campaigns") or [], data["key"]),
        ]
    )


def _new_study_bar(campaigns: list, task: str = "") -> Any:
    hidden = {"display": "none"}
    return html.Div(
        className="qs-bar",
        children=[
            # The form's other fields, when the bar is drawn without it (a
            # study template's view): the callback reads them all.
            *(
                [
                    html.Div(dcc.Dropdown(id="new-study-template"), style=hidden),
                    dcc.Input(id="new-study-name", style=hidden),
                ]
                if task
                else []
            ),
            html.Div(
                dcc.Dropdown(
                    id="new-study-campaign",
                    options=list(campaigns),
                    placeholder="no campaign",
                ),
                style={"minWidth": "220px"},
            ),
            html.Button(
                (
                    "Make a study from it (on hold)"
                    if task
                    else "Make the study (on hold)"
                ),
                id="btn-make-study",
                className="qs-btn qs-btn-primary",
            ),
            dcc.Store(id="new-study-task", data=task),
        ],
    )


def new_study_view(snap: dict, project: str = "") -> Any:
    """+ New study: pick a template, or name a blank one (in ``project``
    when several are open)."""
    multi = bool(snap.get("multi"))
    tasks = [
        c
        for c in snap.get("catalog") or []
        if c["kind"] == "task" and (not multi or c["key"].startswith(f"{project}/"))
    ]
    campaigns = [
        c["key"]
        for c in snap.get("campaigns") or []
        if not multi or c["project"] == project
    ]
    return html.Div(
        [
            html.H3("New study", className="qs-h"),
            html.Div(
                "It is made on hold: edit it here, then queue it (▶ in the "
                "On hold list).",
                className="qs-small",
            ),
            html.Div(
                className="qs-frow",
                children=[
                    html.Label("from template"),
                    dcc.Dropdown(
                        id="new-study-template",
                        options=[
                            {"label": f"{t['label']} ({t['key']})", "value": t["key"]}
                            for t in tasks
                        ],
                        placeholder="blank: a MiniStudy from scratch",
                    ),
                ],
            ),
            html.Div(
                className="qs-frow",
                children=[
                    html.Label("name (blank study)"),
                    dcc.Input(
                        id="new-study-name",
                        className="qs-input",
                        placeholder="purcell_filter",
                    ),
                ],
            ),
            _new_study_bar(campaigns),
        ]
    )


def new_campaign_view() -> Any:
    """+ New campaign: its name (parent/name for a sub-campaign), its
    question, and whether it keeps its own design variables."""
    return html.Div(
        [
            html.H3("New campaign", className="qs-h"),
            html.Div(
                "A campaign answers one question; its README.md is the lab "
                "log. A study in it reads and writes the campaign's design "
                "variables; Promote copies them one level up.",
                className="qs-small",
            ),
            html.Div(
                className="qs-frow",
                children=[
                    html.Label("name"),
                    dcc.Input(
                        id="campaign-key",
                        className="qs-input",
                        placeholder="tuneup_chip or tuneup_chip/q1_readout",
                    ),
                ],
            ),
            html.Div(
                className="qs-frow",
                children=[
                    html.Label("question"),
                    dcc.Input(
                        id="campaign-question",
                        className="qs-input",
                        placeholder="Can every mode reach its target?",
                    ),
                ],
            ),
            dcc.Checklist(
                id="campaign-own-store",
                options=[
                    {"label": " its own copy of the design variables", "value": "own"}
                ],
                value=["own"],
            ),
            html.Button(
                "Make campaign",
                id="btn-new-campaign",
                className="qs-btn qs-btn-primary",
            ),
        ]
    )


def campaign_view(data: dict, multi: bool = False) -> Any:
    """A campaign: its README, where its design variables are, how many
    studies it has, and Promote."""
    counts = data.get("counts") or {}
    index = f"{data['project']}::{data['key']}"
    return html.Div(
        [
            html.H3(f"Campaign {data['key']}", className="qs-h"),
            html.Div(data["folder"], className="qs-muted"),
            html.Div(
                className="qs-frow",
                children=[
                    html.Label("design variables"),
                    html.Span(data["store"], className="qs-mono"),
                ],
            ),
            html.Div(
                className="qs-frow",
                children=[
                    html.Label("studies"),
                    html.Span(", ".join(f"{n} {k}" for k, n in counts.items())),
                ],
            ),
            html.Div(
                className="qs-bar",
                children=[
                    html.Button(
                        "Promote",
                        id={"type": "promote", "index": index},
                        className="qs-btn",
                        n_clicks=0,
                        disabled=not data.get("own_store"),
                        title=(
                            f"copy its changed values into {data.get('promote_to')}"
                            if data.get("own_store")
                            else "it uses its parent's design variables: nothing to promote"
                        ),
                    ),
                    html.Span(
                        (
                            f"to {data.get('promote_to')}"
                            if data.get("own_store")
                            else "shares its parent's design variables"
                        ),
                        className="qs-small",
                    ),
                ],
            ),
            html.H4("README.md", style={"margin": "12px 0 4px"}),
            dcc.Markdown(data.get("readme") or "(empty)", className="qs-readme"),
        ]
    )


def sequence_editor(view: dict) -> Any:
    """A study sequence template: its name and file, defaults for every
    study, and the studies in order. Saved as study_sequences/<file>.yaml."""
    d = view.get("defaults") or {}
    locked = not view.get("new") and not view.get("from_code")

    def row(label, child):
        return html.Div(className="qs-frow", children=[html.Label(label), child])

    title = "New study sequence" if view.get("new") else f"Study sequence {view['key']}"
    return html.Div(
        [
            dcc.Store(
                id="seq-meta",
                data={
                    k: view.get(k)
                    for k in ("key", "new", "from_code", "project", "keep")
                },
            ),
            dcc.Store(id="seq-tasks", data=view.get("tasks") or []),
            html.H3(title, className="qs-h"),
            (
                html.Div(
                    "Built in the adapter's code: save it under a new file "
                    "name to change it.",
                    className="qs-small",
                )
                if view.get("from_code")
                else None
            ),
            row(
                "file name",
                dcc.Input(
                    id={"type": "seq", "name": "key"},
                    value="" if view.get("from_code") else view.get("key", ""),
                    disabled=locked,
                    className="qs-input",
                    placeholder="tune_Q2 (saved as study_sequences/tune_Q2.yaml)",
                ),
            ),
            row(
                "name",
                dcc.Input(
                    id={"type": "seq", "name": "name"},
                    value=view.get("name", ""),
                    className="qs-input",
                ),
            ),
            row(
                "why this order",
                dcc.Textarea(
                    id={"type": "seq", "name": "why_ordered"},
                    value=view.get("why_ordered", ""),
                    className="qs-input",
                    style={"height": "48px"},
                ),
            ),
            row(
                "campaign",
                dcc.Dropdown(
                    id={"type": "seq", "name": "campaign"},
                    options=sorted(
                        set(view.get("campaigns") or [])
                        | ({view["campaign"]} if view.get("campaign") else set())
                    ),
                    value=view.get("campaign") or None,
                    placeholder="none (or pick one when queueing it)",
                ),
            ),
            html.B("For every study", style={"display": "block", "marginTop": "8px"}),
            row(
                "settings",
                dcc.Input(
                    id={"type": "seq", "name": "d_settings"},
                    value=d.get("settings", ""),
                    className="qs-input",
                    placeholder="iterations: 3",
                ),
            ),
            row(
                "checks",
                dcc.Input(
                    id={"type": "seq", "name": "d_checks"},
                    value=d.get("checks", ""),
                    className="qs-input",
                    placeholder=", ".join(view.get("check_names") or []),
                ),
            ),
            html.Div(
                className="qs-bar",
                children=[
                    dcc.Checklist(
                        id={"type": "seq", "name": "d_save"},
                        options=[{"label": " write back on success", "value": "save"}],
                        value=["save"] if d.get("save") else [],
                        inline=True,
                    ),
                    html.Label("on failure"),
                    dcc.Dropdown(
                        id={"type": "seq", "name": "d_on_failure"},
                        options=["abort", "continue"],
                        value=d.get("on_failure", "abort"),
                        clearable=False,
                        style={"minWidth": "120px"},
                    ),
                ],
            ),
            html.B("Studies, in order", style={"display": "block", "marginTop": "8px"}),
            html.Div(
                id="seq-rows-view",
                children=seq_rows(view.get("studies") or [], view.get("tasks") or []),
            ),
            html.Div(
                className="qs-bar",
                children=[
                    html.Button(
                        "Save template",
                        id="btn-seq-save",
                        className="qs-btn qs-btn-primary",
                    ),
                    html.Button(
                        "Add to queue",
                        id="btn-seq-queue",
                        className="qs-btn",
                        title="queue every study of the saved template (in its campaign)",
                        disabled=bool(view.get("new")),
                        n_clicks=0,
                    ),
                ],
            ),
        ]
    )


def seq_rows(rows: list[dict], tasks: list[dict]) -> Any:
    """The studies of a study sequence being edited, each with its own
    settings and notes; reorder, remove, add."""
    body = []
    for n, r in enumerate(rows):

        def icon(action, text, title, disabled=False, n=n):
            return html.Button(
                text,
                id={"type": "seqact", "action": action, "index": n},
                className="qs-icon",
                title=title,
                n_clicks=0,
                disabled=disabled,
            )

        body.append(
            html.Div(
                className="qs-seqrow",
                children=[
                    html.Div(
                        className="qs-bar",
                        style={"flexWrap": "nowrap"},
                        children=[
                            html.Span(str(n + 1), className="qs-muted"),
                            html.Div(
                                dcc.Dropdown(
                                    id={"type": "seqrow", "field": "task", "index": n},
                                    options=tasks,
                                    value=r.get("task") or None,
                                    clearable=False,
                                ),
                                style={"flex": "1", "minWidth": "0"},
                            ),
                            icon("up", "↑", "move up", n == 0),
                            icon("down", "↓", "move down", n == len(rows) - 1),
                            icon("remove", "✕", "take out"),
                        ],
                    ),
                    html.Div(
                        className="qs-bar",
                        children=[
                            dcc.Input(
                                id={"type": "seqrow", "field": "settings", "index": n},
                                value=r.get("settings", ""),
                                className="qs-input",
                                placeholder="settings: as for every study",
                                style={"flex": "1", "minWidth": "140px"},
                            ),
                            dcc.Input(
                                id={"type": "seqrow", "field": "notes", "index": n},
                                value=r.get("notes", ""),
                                className="qs-input",
                                placeholder="notes",
                                style={"flex": "1", "minWidth": "140px"},
                            ),
                            # Per-study checks, save, on_failure, as written
                            # in the file: kept as they are.
                            dcc.Store(
                                id={"type": "seqrow", "field": "extra", "index": n},
                                data=r.get("extra") or {},
                            ),
                        ],
                    ),
                ],
            )
        )
    return html.Div(
        [
            html.Div(
                body
                or html.Div(
                    "No studies yet: add the first below.", className="qs-muted"
                )
            ),
            html.Div(
                className="qs-bar",
                children=[
                    html.Div(
                        dcc.Dropdown(
                            id="seq-add-task",
                            options=tasks,
                            placeholder="add a study ...",
                        ),
                        style={"minWidth": "260px"},
                    ),
                    html.Button("Add", id="btn-seq-add", className="qs-btn"),
                ],
            ),
        ]
    )


def sequence_fields(view: dict) -> dict:
    """The sequence editor's fields exactly as ``sequence_editor`` draws
    them (what "unsaved" is measured against)."""
    d = view.get("defaults") or {}
    return seq_state(
        {
            "key": "" if view.get("from_code") else view.get("key", ""),
            "name": view.get("name", ""),
            "why_ordered": view.get("why_ordered", ""),
            "campaign": view.get("campaign") or None,
            "d_settings": d.get("settings", ""),
            "d_checks": d.get("checks", ""),
            "d_save": ["save"] if d.get("save") else [],
            "d_on_failure": d.get("on_failure", "abort"),
        },
        view.get("studies") or [],
    )


def seq_state(fields: dict, rows: list[dict]) -> dict:
    """The sequence editor's fields and study rows, normalised so drawn
    and live values compare equal (None and "" alike)."""
    return {
        "fields": {k: (v if v not in (None, "") else "") for k, v in fields.items()},
        "rows": [
            [r.get("task") or "", r.get("settings") or "", r.get("notes") or ""]
            for r in rows
        ],
    }
