"""Draw a study file's form (``forms``) as Dash components.

Ids, for the callbacks:

* ``{"type": "ff", "file": <name>, "path": <forms.key>}`` -- a field;
* ``{"type": "fop", "file": <name>, "path": <forms.key>, "op": ...}`` --
  duplicate / toggle / remove one list item;
* ``{"type": "fo", "file": <name>, "key": <value name>, "part":
  "set"|"value"}`` -- an override row of an optional file;
* ``{"type": "file-save", "name": <name>}`` -- the file's Save button;
* ``{"type": "fadd", "file": <name>, "path": <forms.key>}`` and its
  ``{"type": "fadd-choice", ...}`` -- add a new item (a capacitance
  study, an opt target) to a list;
* ``{"type": "tload", "file": <name>}`` / ``tload-choice`` -- append an
  opt-target template; ``{"type": "tsave", ...}`` / ``tsave-name`` --
  save the opt targets alone as a template.
"""

from __future__ import annotations

from typing import Any

from dash import dcc, html

from . import forms

_ON = "on"


def file_form(spec: dict, editable: bool) -> list:
    """One file's pane: what it is, its form, its Save button."""
    name = spec["name"]
    out = [
        html.Div(
            spec.get("about", ""),
            className="qs-muted",
            style={"margin": "6px 0"},
        )
    ]
    if spec.get("problem"):
        out.append(html.Div(spec["problem"], className="qs-problems"))
        return out
    project = spec.get("project")
    if spec.get("optional") and editable and project is not None:
        out.append(
            _overrides(name, spec.get("data"), project, spec.get("sources") or {})
        )
    elif spec.get("data") is None:
        out.append(
            html.Div(
                (
                    "Not in the folder: the project's values are used, "
                    "read when the study starts."
                    if spec.get("optional")
                    else "Missing."
                ),
                className="qs-muted",
            )
        )
        return out
    else:
        addable = spec.get("addable") if editable else {}
        out.append(
            _section(
                forms.build(
                    spec["data"],
                    spec.get("label", ""),
                    lists=[forms.unkey(k) for k in (addable or {})],
                ),
                name,
                editable,
                addable=addable or {},
            )
        )
        if editable and spec.get("target_templates") is not None:
            out.append(_target_templates(name, spec["target_templates"]))
    if editable:
        out.append(
            html.Div(
                className="qs-bar",
                children=[
                    html.Button(
                        f"Save {spec.get('label') or name}",
                        id={"type": "file-save", "name": name},
                        className="qs-btn qs-btn-primary",
                        n_clicks=0,
                    )
                ],
            )
        )
    return out


def form_values(spec: dict, editable: bool = True) -> dict:
    """Every input's value exactly as ``file_form`` draws it, keyed
    ``<file>|<id part>`` (what "unsaved" is measured against)."""
    name = spec["name"]
    out: dict[str, Any] = {}
    if spec.get("problem"):
        return out
    project = spec.get("project")
    if spec.get("optional") and editable and project is not None:
        for row in forms.overrides(spec.get("data"), project):
            out[f"{name}|fo|{row.name}|set"] = [_ON] if row.own else []
            out[f"{name}|fo|{row.name}|value"] = row.value
        return out
    if spec.get("data") is None:
        return out
    addable = spec.get("addable") if editable else {}
    lists = [forms.unkey(k) for k in (addable or {})]
    for item in forms.fields_of(forms.build(spec["data"], lists=lists)):
        out[f"{name}|ff|{forms.key(item.path)}"] = _input_value(item)
    return out


def current_values(ff_values, ff_ids, fo_values, fo_ids) -> dict:
    """The inputs on the page, keyed as ``form_values`` keys them."""
    out = {}
    for i, v in zip(ff_ids or [], ff_values or []):
        out[f"{i['file']}|ff|{i['path']}"] = v
    for i, v in zip(fo_ids or [], fo_values or []):
        out[f"{i['file']}|fo|{i['key']}|{i['part']}"] = v
    return out


def _input_value(item: forms.Field) -> Any:
    if item.kind == "bool":
        return [_ON] if item.value else []
    return item.value


def _field(item: forms.Field, name: str, editable: bool) -> Any:
    ident = {"type": "ff", "file": name, "path": forms.key(item.path)}
    if item.kind == "bool":
        control: dcc.Checklist | dcc.Input = dcc.Checklist(
            id=ident,
            options=[{"label": "", "value": _ON, "disabled": not editable}],
            value=_input_value(item),
            inline=True,
        )
    else:
        control = dcc.Input(
            id=ident,
            value=item.value,
            type="text",
            disabled=not editable,
            placeholder="none" if item.kind == "none" else "",
            className="qs-input" + (" qs-mono" if item.kind == "list" else ""),
        )
    return html.Div(
        className="qs-frow",
        children=[html.Label(item.label, title=item.kind), control],
    )


def _section(section: forms.Section, name: str, editable: bool, depth=0, addable=None):
    addable = addable or {}
    body = []
    if section.item and editable:
        path = forms.key(section.path)

        def op(action, text, title):
            return html.Button(
                text,
                id={"type": "fop", "file": name, "path": path, "op": action},
                className="qs-btn qs-btn-small",
                title=title,
                n_clicks=0,
            )

        body.append(
            html.Div(
                className="qs-bar",
                children=[
                    op("duplicate", "Duplicate", "add a copy below, to edit"),
                    op(
                        "toggle",
                        "Switch on" if not section.active else "Switch off",
                        "a switched-off entry stays in the file but is "
                        "not used when the study runs",
                    ),
                    op("remove", "Remove", "delete it from the file"),
                ],
            )
        )
    if section.note:
        body.append(html.Div(section.note, className="qs-muted qs-small"))
    body += [_field(f, name, editable) for f in section.fields]
    body += [_section(c, name, editable, depth + 1, addable) for c in section.children]
    choices = addable.get(forms.key(section.path))
    if editable and choices:
        path = forms.key(section.path)
        body.append(
            html.Div(
                className="qs-bar",
                children=[
                    dcc.Dropdown(
                        id={"type": "fadd-choice", "file": name, "path": path},
                        options=choices,
                        placeholder=f"add to {section.title} ...",
                        style={"minWidth": "300px"},
                    ),
                    html.Button(
                        "Add",
                        id={"type": "fadd", "file": name, "path": path},
                        className="qs-btn qs-btn-small",
                        n_clicks=0,
                    ),
                ],
            )
        )
    title: list[Any] = [html.B(section.title or name)]
    if not section.active:
        title.append(html.Span("  switched off", className="qs-off-tag"))
    if depth == 0 and not section.item:
        return html.Div(className="qs-form", children=[*title, *body])
    return html.Details(
        open=section.active and depth < 3,
        className="qs-sec" + ("" if section.active else " qs-off"),
        children=[
            html.Summary(title),
            html.Div(className="qs-sec-body", children=body),
        ],
    )


def _target_templates(name: str, templates: list[str]) -> Any:
    return html.Div(
        className="qs-form",
        children=[
            html.B("The project's opt targets"),
            html.Div(
                className="qs-bar",
                children=[
                    dcc.Dropdown(
                        id={"type": "tload-choice", "file": name},
                        options=templates,
                        placeholder="add one of the project's opt targets ...",
                        style={"minWidth": "260px"},
                    ),
                    html.Button(
                        "Add targets",
                        id={"type": "tload", "file": name},
                        className="qs-btn qs-btn-small",
                        n_clicks=0,
                    ),
                ],
            ),
            html.Div(
                className="qs-bar",
                children=[
                    dcc.Input(
                        id={"type": "tsave-name", "file": name},
                        placeholder="template name",
                        className="qs-input",
                        style={"width": "200px"},
                    ),
                    html.Button(
                        "Save these targets as a template",
                        id={"type": "tsave", "file": name},
                        className="qs-btn qs-btn-small",
                        title="writes opt_targets/<name>.yaml: targets "
                        "alone, to add to any study",
                        n_clicks=0,
                    ),
                ],
            ),
        ],
    )


def _overrides(name: str, own, project: dict, sources: dict) -> Any:
    rows = forms.overrides(own, project, sources)
    head = html.Tr(
        [html.Th("set here"), html.Th("name"), html.Th("value")]
        + [html.Th("project")]
        + ([html.Th("last written by")] if sources else [])
    )
    body = []
    for row in rows:
        body.append(
            html.Tr(
                [
                    html.Td(
                        dcc.Checklist(
                            id={
                                "type": "fo",
                                "file": name,
                                "key": row.name,
                                "part": "set",
                            },
                            options=[{"label": "", "value": _ON}],
                            value=[_ON] if row.own else [],
                        )
                    ),
                    html.Td(row.name, className="qs-small"),
                    html.Td(
                        dcc.Input(
                            id={
                                "type": "fo",
                                "file": name,
                                "key": row.name,
                                "part": "value",
                            },
                            value=row.value,
                            type="text",
                            className="qs-input",
                        )
                    ),
                    html.Td(row.project, className="qs-muted qs-small"),
                ]
                + (
                    [html.Td(row.source or "-", className="qs-small")]
                    if sources
                    else []
                )
            )
        )
    return html.Div(
        [
            html.Div(
                "Switch on what this study sets for itself; everything "
                "else is the project's value when the study starts.",
                className="qs-muted qs-small",
            ),
            html.Div(
                className="qs-scroll",
                children=html.Table(
                    className="qs-table",
                    children=[html.Thead(head), html.Tbody(body)],
                ),
            ),
        ]
    )
