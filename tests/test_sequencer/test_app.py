"""The GUI's bridge to the folders, and that the Dash app builds."""

from __future__ import annotations

import json

import pytest

from qdesignoptimizer.sequencer.app import layout
from qdesignoptimizer.sequencer.app.queue_state import Bridge, GuiError
from qdesignoptimizer.sequencer.engine import state as st
from qdesignoptimizer.sequencer.engine.runner import Runner

from .helpers import make_project


@pytest.fixture
def bridge(tmp_path):
    return Bridge(project=make_project(tmp_path))


def test_snapshot_lists_queue_completed_and_catalog(bridge):
    bridge.add("sequence", "full_A")
    bridge.add("task", "B.measure")
    snap = bridge.snapshot()
    assert [r["sequence"] for r in snap["rows"]] == ["s001", "s001", ""]
    assert all(r["problems"] == [] for r in snap["rows"])
    kinds = {c["kind"] for c in snap["catalog"]}
    assert kinds == {"task", "sequence"}
    Runner(bridge.project).run()
    snap = bridge.snapshot()
    assert snap["rows"] == [] and len(snap["completed"]) == 3
    assert {c["status"] for c in snap["completed"]} == {"done"}
    assert snap["runner"]["status"] == st.FINISHED


def test_edit_files_and_study_fields(bridge):
    bridge.add("task", "A.calibrate")
    job_id = bridge.project.queue.ids()[0]
    data = bridge.details("queue", job_id)
    names = [f["name"] for f in data["files"]]
    assert names == ["task.yaml", "design_variables.json"]
    assert data["files"][1]["text"] is None  # optional, not there
    assert "checks out" in bridge.save_file(job_id, "task.yaml", "value: 2.5\n")
    with pytest.raises(GuiError, match="does not parse"):
        bridge.save_file(job_id, "task.yaml", "value: [")
    assert "cannot run" in bridge.save_file(job_id, "task.yaml", "colour: 1\n")
    bridge.save_file(job_id, "task.yaml", "value: 2.5\n")
    bridge.save_file(job_id, "design_variables.json", "{}\n")
    assert bridge.details("queue", job_id)["files"][1]["text"] == "{}\n"
    bridge.delete_optional(job_id, "design_variables.json")
    assert bridge.save_job(job_id, {"save": True, "on_failure": "continue"}) == "saved"
    job = bridge.project.job(job_id)
    assert job.save and job.on_failure == "continue"
    with pytest.raises(GuiError, match="not saved"):
        bridge.save_job(job_id, {"on_failure": "retry"})


def test_order_move_remove_requeue(bridge):
    bridge.add("task", "A.calibrate")
    bridge.add("task", "B.calibrate")
    a, b = bridge.project.queue.ids()
    bridge.reorder([b, a])
    assert bridge.project.queue.ids() == [b, a]
    bridge.move(a, -1)
    assert bridge.project.queue.ids() == [a, b]
    with pytest.raises(GuiError, match="changed"):
        bridge.reorder([a])
    assert "removed" in bridge.remove(b)
    Runner(bridge.project).run()
    (done,) = bridge.project.queue.completed()
    assert bridge.requeue(done.name) == f"queued {a}_r2"
    view = bridge.details("completed", done.name)
    assert view["result"]["status"] == "done" and view["has_inputs"]


def test_start_refuses_when_busy_or_empty(bridge):
    with pytest.raises(GuiError, match="empty"):
        bridge.start()
    bridge.add("task", "A.calibrate")
    lock = st.acquire_lock(bridge.project.out)
    try:
        with pytest.raises(GuiError, match="already going"):
            bridge.start()
    finally:
        st.release_lock(lock)


def test_rows_are_draggable(bridge):
    bridge.add("task", "A.calibrate")
    table = layout.queue_table(bridge.snapshot(), None)
    rows = [tr for tr in table.children[1].children if getattr(tr, "draggable", None)]
    assert [r.draggable for r in rows] == ["true"]
    # selection sits on a cell, so the row's buttons do not also select
    assert not getattr(rows[0], "id", None)
    assert any(
        getattr(td, "id", None)
        == {
            "type": "qrow",
            "index": rows[0].to_plotly_json()["props"]["data-step"],
        }
        for td in rows[0].children
    )


def test_dash_app_builds(bridge):
    from qdesignoptimizer.sequencer.app.server import build_app

    app = build_app(project=bridge.project)
    assert app.layout is not None and len(app.callback_map) >= 12
    # the details pane renders for both kinds
    bridge.add("task", "A.calibrate")
    job_id = bridge.project.queue.ids()[0]
    layout.details(bridge.details("queue", job_id), {})
    Runner(bridge.project).run()
    name = bridge.project.queue.completed()[0].name
    layout.details(bridge.details("completed", name), {})
    json.dumps(bridge.snapshot())  # plain data


def test_the_details_pane_is_not_redrawn_by_saves_or_polls(bridge):
    """Review 2026-11-10: every rev bump redrew the pane and wiped
    unsaved text in the other tabs."""
    from qdesignoptimizer.sequencer.app.server import build_app

    app = build_app(project=bridge.project)
    (key,) = (k for k in app.callback_map if "details.children" in k)
    inputs = {i["id"] for i in app.callback_map[key]["inputs"]}
    # The project picker redraws only a New view (round 4, 2026-11-15).
    assert inputs == {"selected", "new-project", "work-campaign"}


# -- review round 3 (2026-11-10) -----------------------------------------


def test_a_malformed_study_yaml_does_not_blank_the_queue(bridge):
    bridge.add("task", "A.calibrate")
    bridge.add("task", "B.calibrate")
    bad = bridge.project.queue.ids()[0]
    path = bridge.project.queue.folder(bad) / "job.yaml"
    path.write_text(path.read_text("utf-8") + "checks: 5\n", "utf-8")
    snap = bridge.snapshot()
    assert [r["id"] for r in snap["rows"]] == bridge.project.queue.ids()
    assert snap["rows"][0]["problems"]
    assert "removed" in bridge.remove(bad)


def test_unsaved_edits_are_detected():
    from qdesignoptimizer.sequencer.app.callbacks import _unsaved

    orig = {
        "forms": {'task.yaml|ff|["value"]': "1.0"},
        "job": {"notes": "", "checks": ""},
    }
    fids = [{"type": "jf", "name": "notes"}, {"type": "jf", "name": "checks"}]
    same = {'task.yaml|ff|["value"]': "1.0"}
    edited = {'task.yaml|ff|["value"]': "2.0"}
    assert _unsaved(same, ["", ""], fids, orig) == []
    assert _unsaved(edited, ["", ""], fids, orig) == ["task.yaml"]
    assert _unsaved(same, ["hi", ""], fids, orig) == ["job.yaml"]


def test_file_spec_draws_one_pane(bridge):
    from qdesignoptimizer.sequencer.app import form_layout

    bridge.add("task", "A.calibrate")
    job_id = bridge.project.queue.ids()[0]
    spec = bridge.file_spec(job_id, "design_variables.json")
    assert spec["text"] is None and spec["project"]["A_value"] == 0.0
    form_layout.file_form(spec, True)
    form_layout.file_form(bridge.file_spec(job_id, "task.yaml"), True)


def test_the_gui_parks_edits_and_queues_a_study(bridge):
    bridge.add("task", "A.calibrate")
    job_id = bridge.project.queue.ids()[0]
    assert "on hold" in bridge.park(job_id)
    snap = bridge.snapshot()
    assert snap["rows"] == []
    (held,) = snap["held"]
    assert held["id"] == f"@hold/{job_id}" and held["problems"] == []
    # A study on hold is edited like a queued one.
    data = bridge.details("hold", held["id"])
    assert data["kind"] == "hold" and data["job"]["task"] == "A.calibrate"
    bridge.save_job(held["id"], {"notes": "later"})
    assert bridge.snapshot()["held"][0]["notes"] == "later"
    assert bridge.unpark(held["id"]).startswith(f"queued {job_id}")
    assert [r["id"] for r in bridge.snapshot()["rows"]] == [job_id]
    bridge.park(job_id)
    assert "removed" in bridge.remove(f"@hold/{job_id}")
    assert bridge.snapshot()["held"] == []


def test_save_as_template_needs_a_backend_with_templates(bridge):
    bridge.add("task", "A.calibrate")
    job_id = bridge.project.queue.ids()[0]
    with pytest.raises(GuiError, match="no templates"):
        bridge.save_as_template("queue", job_id, "x")


def test_the_project_picker_redraws_only_a_new_view(bridge):
    from dash import no_update
    from dash._callback_context import context_value
    from dash._utils import AttributeDict

    from qdesignoptimizer.sequencer.app.server import build_app

    app = build_app(project=bridge.project)
    (key,) = (k for k in app.callback_map if "details.children" in k)
    details = app.callback_map[key]["callback"].__wrapped__
    token = context_value.set(
        AttributeDict(triggered_inputs=[{"prop_id": "new-project.value", "value": "x"}])
    )
    try:
        assert (
            details({"kind": "template", "key": "A.calibrate"}, "x", None, {})[0]
            is no_update
        )
        assert (
            details({"kind": "new-campaign", "key": ""}, "x", None, {})[0]
            is not no_update
        )
    finally:
        context_value.reset(token)


def test_play_and_pause_wait_for_unsaved_edits(bridge):
    """2026-11-15: ⏸ / ▶ on the study in the editor moves it and redraws
    the editor, so its unsaved edits must be saved first."""
    import json as _json

    from dash import no_update
    from dash._callback_context import context_value
    from dash._utils import AttributeDict

    from qdesignoptimizer.sequencer.app.server import build_app

    bridge.add("task", "A.calibrate")
    (job_id,) = bridge.project.queue.ids()
    app = build_app(project=bridge.project)
    (key,) = (
        k
        for k, v in app.callback_map.items()
        if any('"act"' in str(i["id"]) or "'act'" in str(i["id"]) for i in v["inputs"])
    )
    act = app.callback_map[key]["callback"].__wrapped__
    trig = _json.dumps({"action": "park", "index": job_id}, separators=(",", ":"))
    token = context_value.set(
        AttributeDict(triggered_inputs=[{"prop_id": trig + ".n_clicks", "value": 1}])
    )
    selected = {"kind": "queue", "key": job_id}
    empty = [[]] * 10
    try:
        # A job.yaml field changed from what was drawn: refused.
        _, msg, follow = act(
            [1],
            0,
            selected,
            "tab-a",
            *empty[:4],
            ["edited"],
            [{"type": "jf", "name": "notes"}],
            *empty[:5],
            {"job": {"notes": ""}},
        )
        assert "save them before putting it on hold" in msg["text"]
        assert follow is no_update and bridge.project.queue.ids() == [job_id]
        # Nothing edited: it goes on hold.
        _, msg, follow = act(
            [1],
            0,
            selected,
            "tab-a",
            *empty[:4],
            [""],
            [{"type": "jf", "name": "notes"}],
            *empty[:5],
            {"job": {"notes": ""}},
        )
        assert bridge.project.hold.ids() == [job_id] and follow["kind"] == "hold"
    finally:
        context_value.reset(token)


# -- inspect or edit (2026-11-15) ---------------------------------------------


def test_a_study_is_inspected_until_edit_is_pressed(bridge):
    from qdesignoptimizer.sequencer.app import layout

    bridge.add("task", "A.calibrate")
    (job_id,) = bridge.project.queue.ids()
    data = bridge.details("queue", job_id)
    seen = str(layout.details(data, {}))
    assert "Inspecting" in seen and "'index': 'edit'" in seen
    assert "disabled=True" in seen and "'display': 'none'" in seen  # no Save
    edited = str(layout.details(data, {}, editing=True))
    assert "Editing" in edited and "Done editing" in edited
    assert "disabled=True" not in edited


def test_a_run_study_offers_only_an_edit_copy(bridge):
    from qdesignoptimizer.sequencer.app import layout

    bridge.add("task", "A.calibrate")
    Runner(bridge.project).run()
    (name,) = (c["name"] for c in bridge.snapshot()["completed"])
    seen = str(layout.details(bridge.details("completed", name), {}, editing=True))
    assert "Edit copy" in seen and "Requeue" not in seen and "Editing" not in seen
    key = bridge.edit_copy(name)
    assert key.startswith("@hold/") and bridge.project.hold.ids()
    assert bridge.project.queue.ids() == []  # never in the run order


def test_a_reports_pictures_are_served_by_url_not_inline(bridge):
    """Review 2026-11-22: the full report sent every picture inline as
    base64, tens of MB for a long campaign."""
    from qdesignoptimizer.sequencer.app.queue_state import REPORT_URL
    from qdesignoptimizer.sequencer.app.server import build_app
    from qdesignoptimizer.sequencer.engine import report as rp

    bridge.add("task", "A.calibrate")
    Runner(bridge.project).run()
    app = build_app(project=bridge.project)
    gui = app.qdo_bridge  # the bridge the GUI's callbacks use
    keys = [r["key"] for r in rp.completed_runs(gui.workspace)]
    made = gui.make_report(keys, "")
    assert made["flow"].startswith(REPORT_URL + "/")
    assert "base64" not in json.dumps(made)
    client = app.server.test_client()
    image = client.get(made["flow"])
    assert image.status_code == 200 and image.data[:4] == b"\x89PNG"
    token = made["flow"].split("/")[2]
    assert client.get(f"{REPORT_URL}/{token}/../../secret.txt").status_code == 404
    assert client.get(f"{REPORT_URL}/nope/{rp.FLOW_PNG}").status_code == 404


def test_the_gui_serves_only_the_latest_reports(bridge, monkeypatch):
    from qdesignoptimizer.sequencer.app import queue_state
    from qdesignoptimizer.sequencer.engine import report as rp

    monkeypatch.setattr(queue_state, "REPORTS_SERVED", 2)
    bridge.add("task", "A.calibrate")
    Runner(bridge.project).run()
    keys = [r["key"] for r in rp.completed_runs(bridge.workspace)]
    tokens = [bridge.make_report(keys, "")["flow"].split("/")[2] for _ in range(3)]
    assert bridge.report_file(tokens[0], rp.FLOW_PNG) is None
    assert bridge.report_file(tokens[2], rp.FLOW_PNG) is not None
    assert len(bridge._report_folders) == 2
