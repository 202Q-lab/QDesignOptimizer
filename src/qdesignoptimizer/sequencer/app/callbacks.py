"""Every callback. Each one reads or edits the folders through ``Bridge``."""

from __future__ import annotations

import json

import yaml
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update

from ..engine.job import numbers
from . import form_layout, forms, layout
from .queue_state import Bridge, GuiError


def _msg(text: str, ok: bool = True) -> dict:
    return {"text": text, "ok": ok}


def _clicked() -> bool:
    """Whether the trigger was a real click (not a re-render's 0)."""
    return bool(ctx.triggered_id) and bool(
        ctx.triggered and ctx.triggered[0].get("value")
    )


def register_callbacks(app: Dash, bridge: Bridge) -> None:
    """Wire the page to ``bridge``."""

    @app.callback(
        Output("snapshot", "data"),
        Input("poll", "n_intervals"),
        Input("rev", "data"),
        State("snapshot", "data"),
        State("dragging", "data"),
    )
    def _poll(_n, _rev, old, dragging):
        # A redraw mid-drag would pull the row out from under the
        # pointer; the drop itself bumps ``rev`` and refreshes.
        if dragging and ctx.triggered_id == "poll":
            return no_update
        try:
            snap = bridge.snapshot()
        except Exception as exc:  # noqa: BLE001 -- shown, not raised
            snap = {
                "error": f"{type(exc).__name__}: {exc}",
                "rows": [],
                "completed": [],
                "runner": {},
                "words": {},
                "catalog": [],
            }
        return no_update if snap == old else snap

    @app.callback(
        Output("queue-table", "children"),
        Output("completed-table", "children"),
        Output("runner-banner", "children"),
        Output("add-choice", "options"),
        Output("project-path", "children"),
        Output("btn-start", "disabled"),
        Output("btn-stop", "disabled"),
        Output("btn-start", "children"),
        Input("snapshot", "data"),
        Input("selected", "data"),
    )
    def _render(snap, selected):
        snap = snap or {}
        runner = snap.get("runner") or {}
        alive = bool(runner.get("alive"))
        live = (snap.get("words") or {}).get("live", "live")
        banner = layout.runner_banner(snap)
        if snap.get("error"):
            banner = f"Cannot read the project: {snap['error']}"
        return (
            layout.queue_table(snap, selected),
            layout.completed_table(snap, selected),
            banner,
            layout.add_options(snap),
            snap.get("project", ""),
            alive or not snap.get("rows"),
            not alive or bool(runner.get("stop_requested")),
            f"Start ({live} the queue)",
        )

    @app.callback(
        Output("selected", "data"),
        Output("pending-select", "data"),
        Output("message", "data", allow_duplicate=True),
        Input({"type": "qrow", "index": ALL}, "n_clicks"),
        Input({"type": "crow", "index": ALL}, "n_clicks"),
        Input({"type": "drow", "index": ALL}, "n_clicks"),
        State({"type": "ff", "file": ALL, "path": ALL}, "value"),
        State({"type": "ff", "file": ALL, "path": ALL}, "id"),
        State({"type": "fo", "file": ALL, "key": ALL, "part": ALL}, "value"),
        State({"type": "fo", "file": ALL, "key": ALL, "part": ALL}, "id"),
        State({"type": "jf", "name": ALL}, "value"),
        State({"type": "jf", "name": ALL}, "id"),
        State("details-orig", "data"),
        State("pending-select", "data"),
        State("selected", "data"),
        prevent_initial_call=True,
    )
    def _select(
        _q,
        _c,
        _d,
        ff,
        ff_ids,
        fo,
        fo_ids,
        fields,
        field_ids,
        orig,
        pending,
        current,
    ):
        if not _clicked():
            return no_update, no_update, no_update
        kind = {"qrow": "queue", "crow": "completed", "drow": "draft"}[
            ctx.triggered_id["type"]
        ]
        target = {"kind": kind, "key": ctx.triggered_id["index"]}
        if target == current:
            return no_update, None, no_update
        dirty = _unsaved(
            form_layout.current_values(ff, ff_ids, fo, fo_ids),
            fields,
            field_ids,
            orig,
        )
        if dirty and pending != target:
            # Moving away redraws the pane and drops these edits: ask
            # once; a second click on the same study goes ahead.
            return (
                no_update,
                target,
                _msg(
                    f"Unsaved edits in {', '.join(dirty)}. Save them, or "
                    f"click {target['key']} again to discard them.",
                    False,
                ),
            )
        return target, None, no_update

    @app.callback(
        Output("details-problems", "children"),
        Input("snapshot", "data"),
        Input("selected", "data"),
    )
    def _details_problems(snap, selected):
        return layout.details_problems(snap or {}, selected)

    @app.callback(
        Output("details", "children"),
        Output("details-orig", "data"),
        Input("selected", "data"),
        State("snapshot", "data"),
    )
    def _details(selected, snap):
        # Redrawn on selection only -- never on a poll, a save or a
        # create/delete (those update their own pane), any of which
        # would wipe a half-typed edit in another tab.
        if not selected:
            return layout.details(None, {}), None
        try:
            data = bridge.details(selected["kind"], selected["key"])
        except GuiError as exc:
            return str(exc), None
        editable = selected["kind"] in ("queue", "draft")
        forms_now = {}
        for spec in data.get("files") or []:
            forms_now.update(form_layout.form_values(spec, editable))
        orig = {
            "forms": forms_now,
            "job": layout.job_fields(
                data.get("job") or {}, data.get("settings_keys") or ()
            ),
        }
        return layout.details(data, (snap or {}).get("words") or {}), orig

    @app.callback(
        Output("rev", "data", allow_duplicate=True),
        Output("message", "data", allow_duplicate=True),
        Input({"type": "act", "action": ALL, "index": ALL}, "n_clicks"),
        State("rev", "data"),
        prevent_initial_call=True,
    )
    def _act(_clicks, rev):
        if not _clicked():
            return no_update, no_update
        job_id, action = ctx.triggered_id["index"], ctx.triggered_id["action"]
        try:
            if action == "remove":
                text = bridge.remove(job_id)
            elif action == "hold":
                text = bridge.toggle_hold(job_id)
            else:
                bridge.move(job_id, -1 if action == "up" else 1)
                text = f"moved {job_id}"
            return (rev or 0) + 1, _msg(text)
        except GuiError as exc:
            return (rev or 0) + 1, _msg(str(exc), False)

    @app.callback(
        Output("rev", "data", allow_duplicate=True),
        Output("message", "data", allow_duplicate=True),
        Input("drag-order", "data"),
        State("rev", "data"),
        prevent_initial_call=True,
    )
    def _drop(dropped, rev):
        order = (dropped or {}).get("order")
        if not order:
            return no_update, no_update
        try:
            bridge.reorder(order)
        except GuiError as exc:
            return (rev or 0) + 1, _msg(str(exc), False)
        return (rev or 0) + 1, _msg("queue reordered")

    @app.callback(
        Output("rev", "data", allow_duplicate=True),
        Output("message", "data", allow_duplicate=True),
        Output("draft", "data", allow_duplicate=True),
        Input("btn-add", "n_clicks"),
        State("add-choice", "value"),
        State("rev", "data"),
        prevent_initial_call=True,
    )
    def _add(_n, choice, rev):
        """A study goes straight into the queue; a study sequence opens the
        study sequence popup with its studies, to be edited and launched."""
        if not choice:
            return (
                no_update,
                _msg("pick something to add first", False),
                no_update,
            )
        kind, key = choice.split(":", 1)
        try:
            if kind == "sequence":
                draft = bridge.new_draft(from_sequence=key)
                return (
                    (rev or 0) + 1,
                    _msg(
                        "edit the study sequence's studies, then launch it "
                        "(nothing is queued yet)"
                    ),
                    {"key": draft},
                )
            return (rev or 0) + 1, _msg(bridge.add(kind, key)), no_update
        except GuiError as exc:
            return (rev or 0) + 1, _msg(str(exc), False), no_update

    @app.callback(
        Output("rev", "data", allow_duplicate=True),
        Output("message", "data", allow_duplicate=True),
        Output("details-orig", "data", allow_duplicate=True),
        Input("btn-save-job", "n_clicks"),
        State("selected", "data"),
        State({"type": "jf", "name": ALL}, "value"),
        State({"type": "jf", "name": ALL}, "id"),
        State("details-orig", "data"),
        State("rev", "data"),
        prevent_initial_call=True,
    )
    def _save_job(n, selected, values, ids, orig, rev):
        if (
            not n
            or not selected
            or selected.get("kind")
            not in (
                "queue",
                "draft",
            )
        ):
            return no_update, no_update, no_update
        form = {i["name"]: v for i, v in zip(ids, values)}
        try:
            fields = {
                "settings": _settings_from(form),
                "checks": [
                    c.strip()
                    for c in (form.get("checks") or "").split(",")
                    if c.strip()
                ],
                "on_failure": form.get("on_failure") or "abort",
                "save": bool(form.get("save")),
                "notes": form.get("notes") or "",
            }
            text = bridge.save_job(selected["key"], fields)
        except (GuiError, ValueError) as exc:
            return no_update, _msg(str(exc), False), no_update
        orig = dict(orig or {})
        orig["job"] = form
        return (rev or 0) + 1, _msg(text), orig

    @app.callback(
        Output("rev", "data", allow_duplicate=True),
        Output("message", "data", allow_duplicate=True),
        Output("details-orig", "data", allow_duplicate=True),
        Output({"type": "file-pane", "name": ALL}, "children"),
        Input({"type": "file-save", "name": ALL}, "n_clicks"),
        Input({"type": "fop", "file": ALL, "path": ALL, "op": ALL}, "n_clicks"),
        Input({"type": "fadd", "file": ALL, "path": ALL}, "n_clicks"),
        Input({"type": "tload", "file": ALL}, "n_clicks"),
        Input({"type": "tsave", "file": ALL}, "n_clicks"),
        State({"type": "ff", "file": ALL, "path": ALL}, "value"),
        State({"type": "ff", "file": ALL, "path": ALL}, "id"),
        State({"type": "fo", "file": ALL, "key": ALL, "part": ALL}, "value"),
        State({"type": "fo", "file": ALL, "key": ALL, "part": ALL}, "id"),
        State({"type": "fadd-choice", "file": ALL, "path": ALL}, "value"),
        State({"type": "fadd-choice", "file": ALL, "path": ALL}, "id"),
        State({"type": "tload-choice", "file": ALL}, "value"),
        State({"type": "tload-choice", "file": ALL}, "id"),
        State({"type": "tsave-name", "file": ALL}, "value"),
        State({"type": "tsave-name", "file": ALL}, "id"),
        State("selected", "data"),
        State("details-orig", "data"),
        State("rev", "data"),
        prevent_initial_call=True,
    )
    def _file(
        _s,
        _o,
        _a,
        _l,
        _t,
        ff,
        ff_ids,
        fo,
        fo_ids,
        add_choice,
        add_ids,
        load_choice,
        load_ids,
        tpl_name,
        tpl_ids,
        selected,
        orig,
        rev,
    ):
        """Save a file's form, or duplicate / switch off / remove one of
        its items (the form's other edits are saved with it). Only that
        file's pane is redrawn, so edits in the other tabs survive."""
        panes = [no_update] * len(ctx.outputs_list[3])
        if (
            not _clicked()
            or not selected
            or selected.get("kind")
            not in (
                "queue",
                "draft",
            )
        ):
            return no_update, no_update, no_update, panes
        trig = ctx.triggered_id
        name = trig.get("name") or trig.get("file")
        key = selected["key"]
        edits = {
            i["path"]: v for i, v in zip(ff_ids or [], ff or []) if i["file"] == name
        }
        rows: dict = {}
        for i, v in zip(fo_ids or [], fo or []):
            if i["file"] == name:
                own, text = rows.get(i["key"], (False, ""))
                if i["part"] == "set":
                    own = bool(v)
                else:
                    text = v
                rows[i["key"]] = (own, text)

        def picked(values, ids, **match):
            return next(
                (
                    v
                    for i, v in zip(ids or [], values or [])
                    if all(i.get(k) == m for k, m in match.items())
                ),
                None,
            )

        try:
            if trig["type"] == "fop":
                text = bridge.form_op(
                    key,
                    name,
                    forms.unkey(trig["path"]),
                    trig["op"],
                    edits,
                )
            elif trig["type"] == "fadd":
                text = bridge.form_add(
                    key,
                    name,
                    forms.unkey(trig["path"]),
                    picked(add_choice, add_ids, file=name, path=trig["path"]) or "",
                    edits,
                )
            elif trig["type"] == "tload":
                text = bridge.load_targets(
                    key,
                    name,
                    picked(load_choice, load_ids, file=name) or "",
                    edits,
                )
            elif trig["type"] == "tsave":
                text = bridge.save_targets_template(
                    key,
                    name,
                    picked(tpl_name, tpl_ids, file=name) or "",
                    edits,
                )
            else:
                text = bridge.save_form(
                    key,
                    name,
                    edits=edits if not rows else None,
                    overrides=rows if rows else None,
                )
            spec = bridge.file_spec(key, name)
        except GuiError as exc:
            return no_update, _msg(str(exc), False), no_update, panes
        panes = [
            (
                form_layout.file_form(spec, True)
                if out["id"]["name"] == name
                else no_update
            )
            for out in ctx.outputs_list[3]
        ]
        orig = dict(orig or {})
        kept = {
            k: v
            for k, v in (orig.get("forms") or {}).items()
            if not k.startswith(f"{name}|")
        }
        kept.update(form_layout.form_values(spec, True))
        orig["forms"] = kept
        return (
            (rev or 0) + 1,
            _msg(text, "cannot run" not in text),
            orig,
            panes,
        )

    def _requeue_one(name, rev, exact=False):
        try:
            text = bridge.requeue(name, exact)
        except GuiError as exc:
            return (rev or 0) + 1, _msg(str(exc), False), no_update
        new_id = text.split()[-1]
        return (rev or 0) + 1, _msg(text), {"kind": "queue", "key": new_id}

    @app.callback(
        Output("rev", "data", allow_duplicate=True),
        Output("message", "data", allow_duplicate=True),
        Output("selected", "data", allow_duplicate=True),
        Input({"type": "requeue", "index": ALL}, "n_clicks"),
        State("rev", "data"),
        prevent_initial_call=True,
    )
    def _requeue_row(_rows, rev):
        if not _clicked():
            return no_update, no_update, no_update
        return _requeue_one(ctx.triggered_id["index"], rev)

    # Its own callback: the button exists only while a completed study
    # is selected, and a callback whose static input is absent from the
    # page never fires, which would silence the row buttons above too.
    @app.callback(
        Output("rev", "data", allow_duplicate=True),
        Output("message", "data", allow_duplicate=True),
        Output("selected", "data", allow_duplicate=True),
        Input("btn-requeue", "n_clicks"),
        State("selected", "data"),
        State("rev", "data"),
        prevent_initial_call=True,
    )
    def _requeue_selected(n, selected, rev):
        if not n or not selected or selected.get("kind") != "completed":
            return no_update, no_update, no_update
        return _requeue_one(selected["key"], rev)

    @app.callback(
        Output("rev", "data", allow_duplicate=True),
        Output("message", "data", allow_duplicate=True),
        Output("selected", "data", allow_duplicate=True),
        Input("btn-requeue-exact", "n_clicks"),
        State("selected", "data"),
        State("rev", "data"),
        prevent_initial_call=True,
    )
    def _requeue_exact(n, selected, rev):
        if not n or not selected or selected.get("kind") != "completed":
            return no_update, no_update, no_update
        return _requeue_one(selected["key"], rev, exact=True)

    @app.callback(
        Output("message", "data", allow_duplicate=True),
        Output("rev", "data", allow_duplicate=True),
        Input("btn-check", "n_clicks"),
        Input("btn-stop", "n_clicks"),
        Input("btn-reload", "n_clicks"),
        State("rev", "data"),
        prevent_initial_call=True,
    )
    def _control(_check, _stop, _reload, rev):
        trig = ctx.triggered_id
        try:
            if trig == "btn-check":
                bad = {k: v for k, v in bridge.check_all().items() if v}
                if not bad:
                    return (
                        _msg("every queued study rebuilds from its " "files"),
                        (rev or 0) + 1,
                    )
                first = next(iter(bad.items()))
                return (
                    _msg(
                        f"{len(bad)} cannot run; {first[0]}: " f"{first[1][0]}",
                        False,
                    ),
                    (rev or 0) + 1,
                )
            if trig == "btn-stop":
                return _msg(bridge.stop()), (rev or 0) + 1
            bridge.reload()
            return _msg("catalog reloaded"), (rev or 0) + 1
        except GuiError as exc:
            return _msg(str(exc), False), (rev or 0) + 1

    @app.callback(
        Output("confirm-start", "displayed"),
        Output("confirm-start", "message"),
        Input("btn-start", "n_clicks"),
        State("snapshot", "data"),
        prevent_initial_call=True,
    )
    def _ask_start(_n, snap):
        words = (snap or {}).get("words") or {}
        count = len((snap or {}).get("rows") or [])
        return True, (
            f"{words.get('live', 'Run').capitalize()} the {count} queued "
            f"{words.get('task', 'task')}(s), in order? This uses the real "
            f"solver and writes the {words.get('store', 'store')} for "
            f"those marked 'write back'."
        )

    @app.callback(
        Output("message", "data", allow_duplicate=True),
        Output("rev", "data", allow_duplicate=True),
        Input("confirm-start", "submit_n_clicks"),
        State("rev", "data"),
        prevent_initial_call=True,
    )
    def _start(_n, rev):
        try:
            return _msg(bridge.start()), (rev or 0) + 1
        except GuiError as exc:
            return _msg(str(exc), False), (rev or 0) + 1

    @app.callback(
        Output("inspect", "children"),
        Output("inspect", "style"),
        Input("btn-inspect", "n_clicks"),
        State("selected", "data"),
        prevent_initial_call=True,
    )
    def _inspect(_n, selected):
        if not selected or selected.get("kind") not in ("queue", "draft"):
            return no_update, no_update
        try:
            text = json.dumps(bridge.describe(selected["key"]), indent=1, default=str)
        except Exception as exc:  # noqa: BLE001 -- shown, not raised
            text = f"cannot inspect {selected['key']}: {exc}"
        return text, {"display": "block"}

    @app.callback(
        Output("log", "children"),
        Input("poll", "n_intervals"),
        Input("rev", "data"),
    )
    def _log(_n, _rev):
        try:
            return bridge.log_tail() or "no runner log yet"
        except Exception as exc:  # noqa: BLE001
            return str(exc)

    @app.callback(
        Output("message-view", "children"),
        Output("message-view", "style"),
        Input("message", "data"),
    )
    def _show(message):
        if not message:
            return "", {}
        colour = "#1b5e20" if message.get("ok") else "#b71c1c"
        return message.get("text", ""), {"color": colour}


def _settings_from(form: dict) -> dict:
    """The study's settings from its per-setting fields; an empty field
    is left out (the project's default applies)."""
    out = {}
    for name, text in form.items():
        if not name.startswith("setting:"):
            continue
        text = (text or "").strip()
        if not text:
            continue
        try:
            out[name[len("setting:") :]] = numbers(yaml.safe_load(text))
        except yaml.YAMLError as exc:
            raise ValueError(f"{name[8:]}: {text!r} does not read") from exc
    return out


def _unsaved(now: dict, fields, field_ids, orig) -> list[str]:
    """Names of the files (and job.yaml) whose inputs differ from what
    was last drawn or saved."""
    orig = orig or {}
    before = orig.get("forms") or {}
    job = orig.get("job") or {}
    dirty = sorted({k.split("|", 1)[0] for k, v in now.items() if before.get(k) != v})
    if any(v != job.get(i["name"]) for i, v in zip(field_ids or [], fields or [])):
        dirty.append("job.yaml")
    return dirty


def register_template_callbacks(app, bridge) -> None:
    """ "Save as template" in the details pane."""

    @app.callback(
        Output("message", "data", allow_duplicate=True),
        Output("rev", "data", allow_duplicate=True),
        Input("btn-save-template", "n_clicks"),
        State("tpl-name", "value"),
        State("selected", "data"),
        State("rev", "data"),
        prevent_initial_call=True,
    )
    def _save_template(n, name, selected, rev):
        if not n or not selected:
            return no_update, no_update
        try:
            text = bridge.save_as_template(
                selected["kind"], selected["key"], name or ""
            )
        except GuiError as exc:
            return _msg(str(exc), False), no_update
        return _msg(text), (rev or 0) + 1


def register_draft_callbacks(app, bridge) -> None:
    """The study sequence popup (``draft_layout``)."""
    from . import draft_layout

    @app.callback(
        Output("draft", "data", allow_duplicate=True),
        Input("btn-new-sequence", "n_clicks"),
        prevent_initial_call=True,
    )
    def _open(n):
        return {"screen": "start"} if n else no_update

    @app.callback(
        Output("draft-body", "children"),
        Output("draft-modal", "style"),
        Output("page-root", "className"),
        Input("draft", "data"),
        Input("rev", "data"),
        Input("selected", "data"),
        State("snapshot", "data"),
    )
    def _render(state, _rev, selected, snap):
        if not state:
            return None, {"display": "none"}, "qs-page"
        shown = ({"display": "block"}, "qs-page qs-drafting")
        if state.get("key"):
            try:
                view = bridge.draft_view(state["key"])
            except GuiError as exc:
                return html.Div(str(exc)), *shown
            status = bridge.launch_status(state["key"]) or {}
            return (
                draft_layout.draft_screen(
                    view, selected, status.get("state") == "running"
                ),
                *shown,
            )
        snap = snap or {}
        sequences = [
            {"value": c["key"], "label": f"{c['label']} ({c['key']})"}
            for c in snap.get("catalog") or []
            if c["kind"] == "sequence"
        ]
        return (
            draft_layout.start_screen(
                snap.get("projects") or [], sequences, bridge.drafts()
            ),
            *shown,
        )

    @app.callback(
        Output("draft", "data", allow_duplicate=True),
        Output("rev", "data", allow_duplicate=True),
        Output("message", "data", allow_duplicate=True),
        Output("selected", "data", allow_duplicate=True),
        Input({"type": "dbtn", "name": ALL}, "n_clicks"),
        Input({"type": "dact", "action": ALL, "index": ALL}, "n_clicks"),
        Input({"type": "draft-resume", "index": ALL}, "n_clicks"),
        State({"type": "dval", "name": ALL}, "value"),
        State({"type": "dval", "name": ALL}, "id"),
        State("draft", "data"),
        State("selected", "data"),
        State("rev", "data"),
        prevent_initial_call=True,
    )
    def _act(_b, _a, _r, values, ids, state, selected, rev):
        if not _clicked():
            return no_update, no_update, no_update, no_update
        trig = ctx.triggered_id
        form = {i["name"]: v for i, v in zip(ids or [], values or [])}
        key = (state or {}).get("key")
        bump = (rev or 0) + 1
        on_draft = (selected or {}).get("kind") == "draft"
        close = None, bump, None, (None if on_draft else no_update)
        try:
            # Every action redraws the popup from the draft, so the name
            # typed into it is saved first (whatever the action).
            if key and form.get("label") is not None:
                bridge.draft_label(key, form["label"])
            if trig["type"] == "draft-resume":
                return {"key": trig["index"]}, bump, no_update, no_update
            if trig["type"] == "dact":
                study = trig["index"]
                if trig["action"] == "remove":
                    text = bridge.draft_remove(study)
                    gone = on_draft and selected.get("key") == study
                    return (
                        no_update,
                        bump,
                        _msg(text),
                        None if gone else no_update,
                    )
                bridge.draft_move(study, -1 if trig["action"] == "up" else 1)
                return no_update, bump, no_update, no_update
            name = trig["name"]
            if name == "create":
                new = bridge.new_draft(
                    project=form.get("project") or "",
                    label=form.get("new-label") or "",
                    from_sequence=form.get("from") or "",
                )
                return {"key": new}, bump, no_update, no_update
            if name == "blank":
                study = bridge.draft_blank(key, form.get("blank-name") or "")
                return (
                    no_update,
                    bump,
                    _msg(
                        "a blank ministudy: fill it in on the right, add "
                        "its opt targets, then launch"
                    ),
                    {"kind": "draft", "key": study},
                )
            if name == "add":
                if not form.get("add"):
                    return (
                        no_update,
                        no_update,
                        _msg("pick a study to add first", False),
                        no_update,
                    )
                study = bridge.draft_add(key, form["add"])
                return (
                    no_update,
                    bump,
                    _msg(f"added {study.rsplit('/', 1)[-1]}"),
                    {"kind": "draft", "key": study},
                )
            if name == "launch":
                # In the background: the popup shows how far it got
                # (``_launch_progress``) and closes when it is done.
                text = bridge.start_launch(key, form.get("label") or "")
                return no_update, bump, _msg(text), no_update
            if name == "discard":
                text = bridge.discard_draft(key)
                return None, bump, _msg(text), close[3]
            if name == "close":
                return close
        except GuiError as exc:
            return no_update, bump, _msg(str(exc), False), no_update
        return no_update, no_update, no_update, no_update

    @app.callback(
        Output("message", "data", allow_duplicate=True),
        Input({"type": "dval", "name": ALL}, "value"),
        State({"type": "dval", "name": ALL}, "id"),
        State("draft", "data"),
        prevent_initial_call=True,
    )
    def _label(values, ids, state):
        """Save the study sequence's name as soon as it is entered."""
        trig = ctx.triggered_id
        key = (state or {}).get("key")
        if not trig or trig.get("name") != "label" or not key:
            return no_update
        value = next((v for i, v in zip(ids, values) if i["name"] == "label"), None)
        if value is None:
            return no_update
        try:
            bridge.draft_label(key, value)
        except GuiError as exc:
            return _msg(str(exc), False)
        return no_update

    @app.callback(
        Output("draft-status", "children"),
        Output("draft", "data", allow_duplicate=True),
        Output("rev", "data", allow_duplicate=True),
        Output("message", "data", allow_duplicate=True),
        Output("selected", "data", allow_duplicate=True),
        Input("poll", "n_intervals"),
        Input("draft", "data"),
        State("selected", "data"),
        State("rev", "data"),
        prevent_initial_call=True,
    )
    def _launch_progress(_n, state, selected, rev):
        """Show a background launch's progress; close the popup when it
        is done, keep it open (with the reason) when it failed."""
        key = (state or {}).get("key")
        if not state:
            # The popup is closed: a failure it showed has been read.
            bridge.forget_shown_launches()
        status = bridge.launch_status(key) if key else None
        if not status:
            return None, no_update, no_update, no_update, no_update
        if status["state"] == "running":
            return (
                draft_layout.launch_progress(status),
                no_update,
                no_update,
                no_update,
                no_update,
            )
        if status["state"] != "done":
            # Failed: the reason stays in the popup until it is closed
            # or launched again; the message and the redraw come once.
            reason = html.Div(status["message"], className="qs-problems")
            if not bridge.launch_shown(key):
                return reason, no_update, no_update, no_update, no_update
            return (
                reason,
                no_update,
                (rev or 0) + 1,
                _msg(status["message"], False),
                no_update,
            )
        bridge.forget_launch(key)
        on_draft = (selected or {}).get("kind") == "draft"
        return (
            None,
            None,
            (rev or 0) + 1,
            _msg(status["message"]),
            None if on_draft else no_update,
        )

    @app.callback(
        Output("preview-view", "children", allow_duplicate=True),
        Input("btn-preview", "n_clicks"),
        State("selected", "data"),
        prevent_initial_call=True,
    )
    def _preview(n, selected):
        if not n or not selected:
            return no_update
        try:
            pictures = bridge.preview(selected["kind"], selected["key"])
        except GuiError as exc:
            return html.Div(str(exc), className="qs-problems")
        if not pictures:
            return html.Div("this study runs no simulation", className="qs-small")

        def one(data):
            return html.Div(
                [
                    html.Img(
                        src="data:image/png;base64," + data["png"],
                        style={"width": "100%", "marginTop": "8px"},
                    ),
                    html.Div(data["caption"], className="qs-small"),
                ]
            )

        # One tab per solver run: the HFSS eigenmode model, then each
        # chained capacitance study, each drawn as that solver sees it.
        return dcc.Tabs(
            [
                dcc.Tab(label=data.get("title", "study"), children=one(data))
                for data in pictures
            ]
        )

    @app.callback(
        Output("preview-view", "children"),
        Input("selected", "data"),
    )
    def _clear_preview(_selected):
        return None
