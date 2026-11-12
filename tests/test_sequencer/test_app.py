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
    bridge.add("campaign", "full_A")
    bridge.add("task", "B.measure")
    snap = bridge.snapshot()
    assert [r["campaign"] for r in snap["rows"]] == ["c01", "c01", ""]
    assert all(r["problems"] == [] for r in snap["rows"])
    kinds = {c["kind"] for c in snap["catalog"]}
    assert kinds == {"task", "campaign"}
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
    bridge.create_optional(job_id, "design_variables.json")
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
    assert inputs == {"selected"}


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


def test_hold_toggles_from_the_gui(bridge):
    bridge.add("task", "A.calibrate")
    job_id = bridge.project.queue.ids()[0]
    assert "on hold" in bridge.toggle_hold(job_id)
    assert bridge.snapshot()["rows"][0]["hold"] is True
    assert "released" in bridge.toggle_hold(job_id)


def test_save_as_template_needs_a_backend_with_templates(bridge):
    bridge.add("task", "A.calibrate")
    job_id = bridge.project.queue.ids()[0]
    with pytest.raises(GuiError, match="no templates"):
        bridge.save_as_template("queue", job_id, "x")
