"""Every callback. Each one reads or edits the folders through ``Bridge``."""

from __future__ import annotations

import json

import yaml
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update

from ..engine.job import numbers
from . import form_layout, forms, layout
from .queue_state import Bridge, GuiError


def _queued_key(text: str) -> str:
    """The key in a ``queued <key>[ -- warning, ...]`` message (a key may
    hold spaces: a project is named after its folder)."""
    return text.removeprefix("queued ").split(" -- warning", 1)[0]


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
        Output("held-table", "children"),
        Output("completed-table", "children"),
        Output("runner-banner", "children"),
        Output("add-choice", "options"),
        Output("project-path", "children"),
        Output("btn-start", "disabled"),
        Output("btn-stop", "disabled"),
        Output("btn-start", "children"),
        Output("add-campaign", "options"),
        Output("browser", "children"),
        Output("new-project", "options"),
        Output("new-project-box", "style"),
        Input("snapshot", "data"),
        Input("selected", "data"),
        Input("browser-open", "data"),
    )
    def _render(snap, selected, opened):
        snap = snap or {}
        runner = snap.get("runner") or {}
        alive = bool(runner.get("alive"))
        live = (snap.get("words") or {}).get("live", "live")
        banner = layout.runner_banner(snap)
        if snap.get("error"):
            banner = f"Cannot read the project: {snap['error']}"
        return (
            layout.queue_table(snap, selected),
            layout.held_table(snap, selected),
            layout.completed_table(snap, selected),
            banner,
            layout.add_options(snap),
            snap.get("project", ""),
            alive or not snap.get("rows"),
            not alive or bool(runner.get("stop_requested")),
            f"Start ({live} the queue)",
            layout.campaign_options(snap),
            layout.browser(snap, selected, opened),
            snap.get("projects") or [],
            {"minWidth": "160px"} if snap.get("multi") else {"display": "none"},
        )

    @app.callback(
        Output("selected", "data"),
        Output("pending-select", "data"),
        Output("message", "data", allow_duplicate=True),
        Output("unsaved-warning", "style"),
        Output("unsaved-text", "children"),
        Input({"type": "qrow", "index": ALL}, "n_clicks"),
        Input({"type": "crow", "index": ALL}, "n_clicks"),
        Input({"type": "hrow", "index": ALL}, "n_clicks"),
        Input({"type": "brow", "kind": ALL, "index": ALL}, "n_clicks"),
        Input("btn-open-new-campaign", "n_clicks"),
        Input("btn-open-new-sequence", "n_clicks"),
        Input("btn-open-new-study", "n_clicks"),
        *_EDIT_STATES,
        State("pending-select", "data"),
        State("selected", "data"),
        prevent_initial_call=True,
    )
    def _select(*args):
        *edits, pending, current = args[7:]
        del pending  # asked again whenever the edits are still there
        if not _clicked():
            return no_update, no_update, no_update, no_update, no_update
        trig = ctx.triggered_id
        if isinstance(trig, str):  # one of the New buttons
            kind = {
                "btn-open-new-campaign": "new-campaign",
                "btn-open-new-sequence": "new-sequence",
                "btn-open-new-study": "new-study",
            }[trig]
            target = {"kind": kind, "key": ""}
        elif trig["type"] == "brow":
            target = {"kind": trig["kind"], "key": trig["index"]}
        else:
            kind = {"qrow": "queue", "crow": "completed", "hrow": "hold"}[trig["type"]]
            target = {"kind": kind, "key": trig["index"]}
        hidden = {"display": "none"}
        if target == current:
            return no_update, None, no_update, hidden, ""
        dirty = _edited(*edits)
        if dirty:
            # Moving away redraws the editor and drops these edits: ask.
            where = target["key"] or target["kind"].replace("-", " ")
            return (
                no_update,
                target,
                no_update,
                {"display": "block"},
                f"Unsaved edits in {', '.join(dirty)}. Stay to save them, or "
                f"discard them and open {where}.",
            )
        return target, None, no_update, hidden, ""

    @app.callback(
        Output("selected", "data", allow_duplicate=True),
        Output("pending-select", "data", allow_duplicate=True),
        Output("unsaved-warning", "style", allow_duplicate=True),
        Input("btn-unsaved-stay", "n_clicks"),
        Input("btn-unsaved-discard", "n_clicks"),
        State("pending-select", "data"),
        prevent_initial_call=True,
    )
    def _unsaved_choice(_stay, _discard, pending):
        """The warning's answer: stay (keep editing) or discard the edits
        and open what was clicked."""
        if not _clicked():
            return no_update, no_update, no_update
        hidden = {"display": "none"}
        if ctx.triggered_id == "btn-unsaved-discard" and pending:
            return pending, None, hidden
        return no_update, None, hidden

    @app.callback(
        Output("pending-select", "data", allow_duplicate=True),
        Output("unsaved-warning", "style", allow_duplicate=True),
        Input("selected", "data"),
        prevent_initial_call=True,
    )
    def _clear_warning(_selected):
        """Whatever moved the selection (a ▶, a requeue, a New made), the
        warning about the pane it left is moot."""
        return None, {"display": "none"}

    @app.callback(
        Output("browser-open", "data"),
        Input({"type": "bsec", "index": ALL}, "n_clicks"),
        State("browser-open", "data"),
        prevent_initial_call=True,
    )
    def _toggle_section(_clicks, opened):
        if not _clicked():
            return no_update
        opened = dict(opened or {})
        name = ctx.triggered_id["index"]
        opened[name] = not opened.get(name, True)
        return opened

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
        Input("new-project", "value"),
        State("snapshot", "data"),
    )
    def _details(selected, new_project, snap):
        # Redrawn on selection only -- never on a poll, a save or a
        # create/delete (those update their own pane), any of which
        # would wipe a half-typed edit in another tab. The one exception:
        # picking the project redraws a New view, which is made for it.
        if not selected:
            return layout.details(None, {}), None
        kind, key = selected.get("kind"), selected.get("key") or ""
        if ctx.triggered_id == "new-project" and not str(kind).startswith("new-"):
            return no_update, no_update
        snap = snap or {}
        try:
            multi = not bridge.workspace.single
            project = (new_project or "") if multi else ""
            if multi and kind.startswith("new-") and not project:
                return (
                    layout.editor_message(
                        "Pick the project first (next to the New buttons)."
                    ),
                    None,
                )
            if kind == "new-study":
                return layout.new_study_view(snap, project), None
            if kind == "new-campaign":
                return layout.new_campaign_view(), None
            if kind in ("sequence", "new-sequence"):
                view = bridge.sequence_view(key, project)
                return layout.sequence_editor(view), {
                    "seq": layout.sequence_fields(view)
                }
            if kind == "template":
                return layout.template_view(bridge.template_view(key)), None
            if kind == "campaign":
                project, _, local = key.rpartition("::")
                multi = not bridge.workspace.single
                return (
                    layout.campaign_view(
                        bridge.campaign_view(local, project if multi else ""), multi
                    ),
                    None,
                )
            data = bridge.details(kind, key)
        except GuiError as exc:
            return layout.editor_message(str(exc)), None
        editable = kind in ("queue", "hold")
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
        Output("selected", "data", allow_duplicate=True),
        Input({"type": "act", "action": ALL, "index": ALL}, "n_clicks"),
        State("rev", "data"),
        State("selected", "data"),
        *_EDIT_STATES,
        prevent_initial_call=True,
    )
    def _act(_clicks, rev, selected, *edits):
        if not _clicked():
            return no_update, no_update, no_update
        job_id, action = ctx.triggered_id["index"], ctx.triggered_id["action"]
        shown = (selected or {}).get("key") == job_id
        follow = no_update
        if shown and action in ("park", "unpark"):
            dirty = _edited(*edits)
            if dirty:
                # Moving the study redraws the editor: its edits would go.
                verb = "putting it on hold" if action == "park" else "queueing it"
                return (
                    no_update,
                    _msg(
                        f"{job_id} has unsaved edits in {', '.join(dirty)}: "
                        f"save them before {verb}",
                        False,
                    ),
                    no_update,
                )
        try:
            if action == "remove":
                text = bridge.remove(job_id)
                follow = None if shown else no_update
            elif action == "park":
                text = bridge.park(job_id)
                if shown:
                    follow = {"kind": "hold", "key": bridge.hold_key_of(job_id)}
            elif action == "unpark":
                text = bridge.unpark(job_id)
                if shown:
                    follow = {"kind": "queue", "key": _queued_key(text)}
            else:
                bridge.move(job_id, -1 if action == "up" else 1)
                text = f"moved {job_id}"
            return (rev or 0) + 1, _msg(text), follow
        except GuiError as exc:
            return (rev or 0) + 1, _msg(str(exc), False), no_update

    @app.callback(
        Output("rev", "data", allow_duplicate=True),
        Output("message", "data", allow_duplicate=True),
        Output("selected", "data", allow_duplicate=True),
        Input("btn-new-campaign", "n_clicks"),
        State("campaign-key", "value"),
        State("campaign-question", "value"),
        State("campaign-own-store", "value"),
        State("new-project", "value"),
        State("rev", "data"),
        prevent_initial_call=True,
    )
    def _new_campaign(n, key, question, own, project, rev):
        if not n:
            return no_update, no_update, no_update
        if not (key or "").strip():
            return no_update, _msg("name the campaign first", False), no_update
        key = key.strip().replace("\\", "/").strip("/")
        multi = not bridge.workspace.single
        try:
            text = bridge.new_campaign(
                key,
                question or "",
                "own" in (own or []),
                (project or "") if multi else "",
            )
        except GuiError as exc:
            return no_update, _msg(str(exc), False), no_update
        shown = f"{project}::{key}" if multi else key
        return (rev or 0) + 1, _msg(text), {"kind": "campaign", "key": shown}

    @app.callback(
        Output("rev", "data", allow_duplicate=True),
        Output("message", "data", allow_duplicate=True),
        Input({"type": "promote", "index": ALL}, "n_clicks"),
        State("rev", "data"),
        prevent_initial_call=True,
    )
    def _promote(_clicks, rev):
        if not _clicked():
            return no_update, no_update
        project, _, key = ctx.triggered_id["index"].partition("::")
        try:
            text = bridge.promote(key, "" if bridge.workspace.single else project)
        except GuiError as exc:
            return no_update, _msg(str(exc), False)
        return (rev or 0) + 1, _msg(text)

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
        Input("btn-add", "n_clicks"),
        State("add-choice", "value"),
        State("add-campaign", "value"),
        State("rev", "data"),
        prevent_initial_call=True,
    )
    def _add(_n, choice, campaign, rev):
        """A study, or every study of a study sequence template, into the
        queue (in the picked campaign). Edit a template in the editor."""
        if not choice:
            return no_update, _msg("pick something to add first", False)
        kind, key = choice.split(":", 1)
        try:
            return (rev or 0) + 1, _msg(bridge.add(kind, key, campaign or ""))
        except GuiError as exc:
            return (rev or 0) + 1, _msg(str(exc), False)

    @app.callback(
        Output("rev", "data", allow_duplicate=True),
        Output("message", "data", allow_duplicate=True),
        Output("details-orig", "data", allow_duplicate=True),
        Output("selected", "data", allow_duplicate=True),
        Input("btn-save-job", "n_clicks"),
        State("selected", "data"),
        State({"type": "jf", "name": ALL}, "value"),
        State({"type": "jf", "name": ALL}, "id"),
        State("details-orig", "data"),
        State("rev", "data"),
        prevent_initial_call=True,
    )
    def _save_job(n, selected, values, ids, orig, rev):
        if not n or not selected or selected.get("kind") not in ("queue", "hold"):
            return no_update, no_update, no_update, no_update
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
                "campaign": form.get("campaign") or "",
            }
            text = bridge.save_job(selected["key"], fields)
        except (GuiError, ValueError) as exc:
            return no_update, _msg(str(exc), False), no_update, no_update
        before = ((orig or {}).get("job") or {}).get("campaign") or ""
        if fields["campaign"] != before:
            # Another campaign, other design variables: the study is
            # drawn again, so its tabs show that campaign's values.
            return (rev or 0) + 1, _msg(text), no_update, dict(selected)
        orig = dict(orig or {})
        orig["job"] = form
        return (rev or 0) + 1, _msg(text), orig, no_update

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
            or selected.get("kind") not in ("queue", "hold")
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
        return (rev or 0) + 1, _msg(text), {"kind": "queue", "key": _queued_key(text)}

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
        Output("inspect", "children", allow_duplicate=True),
        Output("inspect", "style", allow_duplicate=True),
        Input("selected", "data"),
        prevent_initial_call=True,
    )
    def _hide_inspect(_selected):
        """Another study or template picked: its Inspect is closed."""
        return "", {"display": "none"}

    @app.callback(
        Output("inspect", "children"),
        Output("inspect", "style"),
        Input("btn-inspect", "n_clicks"),
        State("selected", "data"),
        prevent_initial_call=True,
    )
    def _inspect(_n, selected):
        # Only on a real click: the button is drawn anew with each study,
        # and that alone must not open the box.
        if (
            not _clicked()
            or not selected
            or selected.get("kind") not in ("queue", "hold")
        ):
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


#: Every input of the editor, then what it was drawn with: what
#: ``_edited`` compares to tell unsaved edits.
_EDIT_STATES = (
    State({"type": "ff", "file": ALL, "path": ALL}, "value"),
    State({"type": "ff", "file": ALL, "path": ALL}, "id"),
    State({"type": "fo", "file": ALL, "key": ALL, "part": ALL}, "value"),
    State({"type": "fo", "file": ALL, "key": ALL, "part": ALL}, "id"),
    State({"type": "jf", "name": ALL}, "value"),
    State({"type": "jf", "name": ALL}, "id"),
    State({"type": "seq", "name": ALL}, "value"),
    State({"type": "seq", "name": ALL}, "id"),
    State({"type": "seqrow", "field": ALL, "index": ALL}, "value"),
    State({"type": "seqrow", "field": ALL, "index": ALL}, "data"),
    State({"type": "seqrow", "field": ALL, "index": ALL}, "id"),
    State("details-orig", "data"),
)


def _edited(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    ff, ff_ids, fo, fo_ids, fields, field_ids, seq, seq_ids, rvalues, rdatas, rids, orig
) -> list[str]:
    """What in the editor has edits not saved yet ([] for none), from
    the values of ``_EDIT_STATES``."""
    dirty = _unsaved(
        form_layout.current_values(ff, ff_ids, fo, fo_ids), fields, field_ids, orig
    )
    if seq_ids and (orig or {}).get("seq") is not None:
        now = layout.seq_state(
            {i["name"]: v for i, v in zip(seq_ids, seq)},
            _rows_from(rvalues, rdatas, rids),
        )
        if now != orig["seq"]:
            dirty.append("the study sequence")
    return dirty


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


def register_preview_callbacks(app, bridge) -> None:
    """The pictures of what a study simulates, and a completed study's
    convergence plot."""

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
    def _clear_preview(selected):
        # A completed study opens on its convergence plot (the preview
        # button replaces it with the pictures of what it simulated).
        if not selected:
            return None
        png = bridge.convergence_png(selected.get("kind", ""), selected.get("key", ""))
        if not png:
            return None
        return html.Div(
            [
                html.Img(
                    src="data:image/png;base64," + png,
                    style={"width": "100%", "marginTop": "8px"},
                ),
                html.Div(
                    "convergence: each target parameter per iteration",
                    className="qs-small",
                ),
            ]
        )


def register_editor_callbacks(app, bridge) -> None:
    """+ New study, and the study sequence template editor."""

    @app.callback(
        Output("rev", "data", allow_duplicate=True),
        Output("message", "data", allow_duplicate=True),
        Output("selected", "data", allow_duplicate=True),
        Input("btn-make-study", "n_clicks"),
        State("new-study-task", "data"),
        State("new-study-template", "value"),
        State("new-study-name", "value"),
        State("new-study-campaign", "value"),
        State("new-project", "value"),
        State("rev", "data"),
        prevent_initial_call=True,
    )
    def _make_study(n, task, template, name, campaign, project, rev):
        if not n:
            return no_update, no_update, no_update
        try:
            key = bridge.new_study(
                task or template or "",
                name or "",
                campaign or "",
                (project or "") if not bridge.workspace.single else "",
            )
        except GuiError as exc:
            return no_update, _msg(str(exc), False), no_update
        return (
            (rev or 0) + 1,
            _msg(f"made {key.rsplit('/', 1)[-1]} on hold: edit it, then queue it (▶)"),
            {"kind": "hold", "key": key},
        )

    @app.callback(
        Output("seq-rows-view", "children"),
        Input({"type": "seqact", "action": ALL, "index": ALL}, "n_clicks"),
        Input("btn-seq-add", "n_clicks"),
        State({"type": "seqrow", "field": ALL, "index": ALL}, "value"),
        State({"type": "seqrow", "field": ALL, "index": ALL}, "data"),
        State({"type": "seqrow", "field": ALL, "index": ALL}, "id"),
        State("seq-add-task", "value"),
        State("seq-tasks", "data"),
        prevent_initial_call=True,
    )
    def _seq_rows(_acts, _add, values, datas, ids, add_task, tasks):
        trig = ctx.triggered_id
        if trig is None or (trig != "btn-seq-add" and not _clicked()):
            return no_update
        rows = _rows_from(values, datas, ids)
        if trig == "btn-seq-add":
            if not add_task:
                return no_update
            rows.append({"task": add_task, "settings": "", "notes": "", "extra": {}})
        else:
            n, action = trig["index"], trig["action"]
            if action == "remove":
                rows.pop(n)
            else:
                m = n - 1 if action == "up" else n + 1
                if 0 <= m < len(rows):
                    rows[n], rows[m] = rows[m], rows[n]
        return layout.seq_rows(rows, tasks or [])

    @app.callback(
        Output("rev", "data", allow_duplicate=True),
        Output("message", "data", allow_duplicate=True),
        Output("selected", "data", allow_duplicate=True),
        Input("btn-seq-save", "n_clicks"),
        Input("btn-seq-queue", "n_clicks"),
        State("seq-meta", "data"),
        State({"type": "seq", "name": ALL}, "value"),
        State({"type": "seq", "name": ALL}, "id"),
        State({"type": "seqrow", "field": ALL, "index": ALL}, "value"),
        State({"type": "seqrow", "field": ALL, "index": ALL}, "data"),
        State({"type": "seqrow", "field": ALL, "index": ALL}, "id"),
        State("rev", "data"),
        prevent_initial_call=True,
    )
    def _seq_save(_s, _q, meta, values, ids, rvalues, rdatas, rids, rev):
        trig = ctx.triggered_id
        if trig is None or not _clicked():
            return no_update, no_update, no_update
        meta = meta or {}
        form = {i["name"]: v for i, v in zip(ids, values)}
        multi = not bridge.workspace.single
        project = meta.get("project") or ""
        if trig == "btn-seq-queue":
            # The template as saved: its own name, not what the file
            # name field holds (empty, or a name not saved yet).
            saved = meta.get("key") or ""
            try:
                text = bridge.add(
                    "sequence",
                    f"{project}/{saved}" if multi else saved,
                    form.get("campaign") or "",
                )
            except GuiError as exc:
                return no_update, _msg(str(exc), False), no_update
            return (rev or 0) + 1, _msg(text), no_update
        key = (form.get("key") or "").strip()
        if not key:
            return (
                no_update,
                _msg("give the template a file name first", False),
                no_update,
            )
        catalog_key = f"{project}/{key}" if multi else key
        data = {
            "keep": meta.get("keep") or {},
            "name": form.get("name") or "",
            "why_ordered": form.get("why_ordered") or "",
            "campaign": form.get("campaign") or "",
            "defaults": {
                "settings": form.get("d_settings") or "",
                "checks": form.get("d_checks") or "",
                "save": bool(form.get("d_save")),
                "on_failure": form.get("d_on_failure") or "abort",
            },
            "studies": _rows_from(rvalues, rdatas, rids),
        }
        overwrite = not meta.get("new") and not meta.get("from_code")
        try:
            text = bridge.save_sequence(
                key, data, overwrite=overwrite, project=project if multi else ""
            )
        except GuiError as exc:
            return no_update, _msg(str(exc), False), no_update
        return (rev or 0) + 1, _msg(text), {"kind": "sequence", "key": catalog_key}


def _rows_from(values, datas, ids) -> list[dict]:
    """The study rows of the sequence editor, from its fields."""
    rows: dict[int, dict] = {}
    for value, data, i in zip(values or [], datas or [], ids or []):
        row = rows.setdefault(
            i["index"], {"task": "", "settings": "", "notes": "", "extra": {}}
        )
        row[i["field"]] = data if i["field"] == "extra" else (value or "")
    return [rows[n] for n in sorted(rows)]
