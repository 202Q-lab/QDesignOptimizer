"""Draft study sequences: put together, edited and launched as one."""

from __future__ import annotations

import pytest
import yaml

from qdesignoptimizer.sequencer.app import forms
from qdesignoptimizer.sequencer.app.queue_state import Bridge, GuiError
from qdesignoptimizer.sequencer.engine import state as st
from qdesignoptimizer.sequencer.engine.runner import Runner
from qdesignoptimizer.sequencer.engine.workspace import Workspace

from .helpers import make_project


@pytest.fixture
def bridge(tmp_path):
    return Bridge(project=make_project(tmp_path))


def test_a_draft_is_built_edited_and_launched_as_one_sequence(bridge):
    project = bridge.project
    key = bridge.new_draft(label="my run", from_sequence="full_A")
    view = bridge.draft_view(key)
    assert view["label"] == "my run"
    assert [s["task"] for s in view["studies"]] == [
        "A.calibrate",
        "A.measure",
    ]
    added = bridge.draft_add(key, "B.measure")
    assert added.startswith(key + "/")

    # Edited exactly like a queued study: forms, fields, item ops.
    bridge.save_form(added, "task.yaml", edits={forms.key(("value",)): "7"})
    bridge.save_job(added, {"notes": "third"})
    data = bridge.details("draft", added)
    assert (
        yaml.safe_load(
            next(f for f in data["files"] if f["name"] == "task.yaml")["text"]
        )["value"]
        == 7
    )

    # Reordered and trimmed; the runner never sees a draft.
    bridge.draft_move(added, -1)
    assert [s["task"] for s in bridge.draft_view(key)["studies"]] == [
        "A.calibrate",
        "B.measure",
        "A.measure",
    ]
    assert Runner(project).run().done == []
    first = bridge.draft_view(key)["studies"][0]["key"]
    bridge.draft_remove(first)

    text = bridge.launch_draft(key)
    assert "launched 2" in text
    ids = project.queue.ids()
    assert ids == ["s01-my_run-b_measure", "s01-my_run-a_measure"]
    job = project.job(ids[0])
    assert job.sequence == {"id": "s01", "name": "my run", "entry": 1}
    assert job.notes == "third" and job.save is True
    assert bridge.drafts() == []
    with pytest.raises(GuiError, match="no draft"):
        bridge.draft_view(key)


def test_a_draft_with_a_study_that_cannot_run_is_not_launched(bridge):
    key = bridge.new_draft(label="x")
    study = bridge.draft_add(key, "A.calibrate")
    bridge.save_file(study, "task.yaml", "nonsense: 1\n")
    assert bridge.draft_view(key)["studies"][0]["problems"]
    with pytest.raises(GuiError, match="fix first"):
        bridge.launch_draft(key)
    assert bridge.project.queue.ids() == []
    with pytest.raises(GuiError, match="no studies"):
        bridge.launch_draft(bridge.new_draft())


def test_a_draft_can_be_left_and_discarded(bridge):
    key = bridge.new_draft(label="later")
    bridge.draft_add(key, "A.calibrate")
    (listed,) = bridge.drafts()
    assert listed["key"] == key and listed["count"] == 1
    assert "discarded" in bridge.discard_draft(key)
    assert bridge.drafts() == []
    assert any(bridge.project.queue.removed_dir.iterdir())


def test_drafts_with_several_projects_open(tmp_path):
    a, b = make_project(tmp_path / "a"), make_project(tmp_path / "b")
    workspace = Workspace([a, b], state_dir=tmp_path / "state")
    bridge = Bridge(workspace=workspace)
    with pytest.raises(GuiError, match="pick the project"):
        bridge.new_draft(label="x")
    key = bridge.new_draft(project="b", label="on b")
    assert key.startswith("b/~")
    bridge.draft_add(key, "A.calibrate")
    bridge.launch_draft(key)
    assert workspace.ids() == ["b/s01-on_b-a_calibrate"]
    assert st.read_json(workspace.order_file) == ["b/s01-on_b-a_calibrate"]
