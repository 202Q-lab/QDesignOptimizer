"""The study sequence popup: put a study sequence together before it is queued.

Two screens. **Start**: pick the project (several open), name the
study sequence, optionally start from a study sequence's studies, or resume a draft
left earlier. **Draft**: the study sequence's studies in order -- add one,
move it, remove it, click it to edit its files in the details pane
(which moves beside the popup while it is open) -- then Launch.

Ids are pattern ids, so one callback serves every button whether or not
it is on the screen: ``{"type": "dbtn", "name": ...}`` for buttons,
``{"type": "dval", "name": ...}`` for inputs, ``{"type": "drow",
"index": <study key>}`` to select a study, ``{"type": "dact", "action":
"up"|"down"|"remove", "index": <study key>}``, ``{"type":
"draft-resume", "index": <draft key>}``.
"""

from __future__ import annotations

from typing import Any

from dash import dcc, html

from . import theme


def _button(name: str, text: str, primary=False, title="", disabled=False):
    return html.Button(
        text,
        id={"type": "dbtn", "name": name},
        className="qs-btn" + (" qs-btn-primary" if primary else ""),
        title=title,
        disabled=disabled,
        n_clicks=0,
    )


def start_screen(projects: list[str], sequences: list[dict], drafts: list[dict]) -> Any:
    """Name the study sequence and say where it starts from."""
    parts: list[Any] = [html.H3("New study sequence", className="qs-h")]
    if len(projects) > 1:
        parts.append(
            html.Div(
                className="qs-frow",
                children=[
                    html.Label("project"),
                    dcc.Dropdown(
                        id={"type": "dval", "name": "project"},
                        options=projects,
                        value=projects[0],
                        clearable=False,
                    ),
                ],
            )
        )
    parts += [
        html.Div(
            className="qs-frow",
            children=[
                html.Label("name"),
                dcc.Input(
                    id={"type": "dval", "name": "new-label"},
                    placeholder="what this study sequence does",
                    className="qs-input",
                ),
            ],
        ),
        html.Div(
            className="qs-frow",
            children=[
                html.Label("start from"),
                dcc.Dropdown(
                    id={"type": "dval", "name": "from"},
                    options=sequences,
                    placeholder="empty: add studies one by one",
                ),
            ],
        ),
        html.Div(
            className="qs-bar",
            children=[
                _button("create", "Create", primary=True),
                _button("close", "Cancel"),
            ],
        ),
    ]
    if drafts:
        parts.append(html.H4("Or continue a draft", className="qs-h"))
        for draft in drafts:
            parts.append(
                html.Div(
                    className="qs-bar",
                    children=[
                        html.Button(
                            "Continue",
                            id={"type": "draft-resume", "index": draft["key"]},
                            className="qs-btn qs-btn-small",
                            n_clicks=0,
                        ),
                        html.Span(
                            f"{draft.get('label') or draft['name']}  --  "
                            f"{draft['count']} stud"
                            f"{'y' if draft['count'] == 1 else 'ies'}, "
                            f"{draft.get('project', '')} "
                            f"{draft.get('created', '')}",
                            className="qs-small",
                        ),
                    ],
                )
            )
    return html.Div(parts)


def launch_progress(status: dict | None) -> Any:
    """One line on a launch going on in the background."""
    if not status or status.get("state") != "running":
        return None
    total = status.get("total") or 0
    checked = status.get("checked") or 0
    current = status.get("current") or ""
    text = (
        f"Launching: checking {current} ({checked + 1} of {total})"
        if current
        else (
            "Launching: queueing the studies"
            if checked >= total
            else "Launching: starting"
        )
    )
    share = 100 * checked / total if total else 0
    return html.Div(
        className="qs-problems",
        style={"background": "#e3f2fd", "borderColor": "#90caf9"},
        children=[
            html.B(text),
            html.Div(
                style={
                    "height": "6px",
                    "background": "#cfd8dc",
                    "borderRadius": "3px",
                    "marginTop": "6px",
                },
                children=html.Div(
                    style={
                        "width": f"{share:.0f}%",
                        "height": "6px",
                        "background": "#1565c0",
                        "borderRadius": "3px",
                    }
                ),
            ),
            html.Div(
                "Each check builds the design; the GUI stays usable.",
                className="qs-small",
            ),
        ],
    )


def draft_screen(view: dict, selected: dict | None, launching: bool = False) -> Any:
    """The study sequence's studies, in order, and what to do with them."""
    picked = (
        (selected or {}).get("key") if (selected or {}).get("kind") == "draft" else None
    )
    studies = view.get("studies") or []
    rows = []
    for study in studies:
        problems = study.get("problems") or []
        rows.append(
            html.Tr(
                className="qs-row",
                style=(
                    {"background": theme.SELECTED_BG} if study["key"] == picked else {}
                ),
                children=[
                    html.Td(str(study["n"]), className="qs-muted"),
                    html.Td(
                        [
                            study.get("label", ""),
                            html.Br(),
                            html.Span(
                                study["key"].rsplit("/", 1)[-1],
                                className="qs-small",
                            ),
                        ],
                        id={"type": "drow", "index": study["key"]},
                        className="qs-pick",
                        n_clicks=0,
                    ),
                    html.Td(
                        study.get("settings", "")
                        + ("" if study.get("save") else "  (no write-back)"),
                        className="qs-small",
                    ),
                    html.Td(
                        html.Span(
                            f"cannot run: {problems[0]}",
                            style={"color": theme.STATUS_COLORS["failed"][0]},
                        )
                        if problems
                        else ""
                    ),
                    html.Td(_actions(study, len(studies))),
                ],
            )
        )
    if not rows:
        rows = [
            html.Tr(
                html.Td(
                    "No studies yet: add the first one below.",
                    colSpan=5,
                    className="qs-muted",
                )
            )
        ]
    blocked = launching or not studies or any(s.get("problems") for s in studies)
    return html.Div(
        [
            html.H3(
                "Study sequence"
                + (f" for {view['project']}" if view.get("project") else ""),
                className="qs-h",
            ),
            html.Div(
                className="qs-frow",
                children=[
                    html.Label("name"),
                    dcc.Input(
                        id={"type": "dval", "name": "label"},
                        value=view.get("label", ""),
                        className="qs-input",
                        # Saved when the field is left or Enter is
                        # pressed, not per keystroke: a redraw of the
                        # popup reads the name back from the draft.
                        debounce=True,
                    ),
                ],
            ),
            (
                html.Div(f"started from {view['from']}", className="qs-small")
                if view.get("from")
                else None
            ),
            html.Div(
                "Click a study to edit it on the right; nothing is queued "
                "until you launch.",
                className="qs-muted qs-small",
                style={"margin": "6px 0"},
            ),
            html.Table(
                className="qs-table",
                children=[
                    html.Thead(
                        html.Tr(
                            [
                                html.Th("#"),
                                html.Th("Study"),
                                html.Th("Settings"),
                                html.Th(""),
                                html.Th(""),
                            ]
                        )
                    ),
                    html.Tbody(rows),
                ],
            ),
            html.Div(
                className="qs-bar",
                children=[
                    dcc.Dropdown(
                        id={"type": "dval", "name": "add"},
                        options=view.get("tasks") or [],
                        placeholder="Add a study ...",
                        style={"minWidth": "300px"},
                    ),
                    _button("add", "Add to study sequence"),
                ],
            ),
            html.Div(
                className="qs-bar",
                children=[
                    dcc.Input(
                        id={"type": "dval", "name": "blank-name"},
                        placeholder="name of a new ministudy",
                        className="qs-input",
                        style={"width": "220px"},
                    ),
                    _button(
                        "blank",
                        "New ministudy from scratch",
                        title="a MiniStudy with every field to fill in "
                        "and no opt targets yet; save it as a template "
                        "afterwards to reuse it",
                    ),
                ],
            ),
            html.Div(
                className="qs-bar",
                style={"marginTop": "18px"},
                children=[
                    _button(
                        "launch",
                        f"Launch study sequence ({len(studies)})",
                        primary=True,
                        disabled=blocked,
                        title=(
                            "it is being launched"
                            if launching
                            else (
                                "fix the studies that cannot run first"
                                if studies and blocked
                                else "check every study (in the background), "
                                "then queue them in this order as one study sequence"
                            )
                        ),
                    ),
                    _button(
                        "close",
                        "Close (keep the draft)",
                        title="come back to it from New study sequence",
                    ),
                    _button("discard", "Discard"),
                ],
            ),
        ]
    )


def _actions(study: dict, count: int) -> Any:
    def icon(action, text, title, disabled=False):
        return html.Button(
            text,
            id={"type": "dact", "action": action, "index": study["key"]},
            className="qs-icon",
            title=title,
            disabled=disabled,
            n_clicks=0,
        )

    return html.Span(
        [
            icon("up", "↑", "move up", study["n"] == 1),
            icon("down", "↓", "move down", study["n"] == count),
            icon("remove", "✕", "take out of the study sequence"),
        ]
    )
