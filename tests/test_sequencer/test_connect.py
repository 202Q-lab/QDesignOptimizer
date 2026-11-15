"""The left side of the GUI (2026-11-15): connected project folders,
one project worked on at a time, campaigns picked to work in."""

from __future__ import annotations

import json

import pytest

from qdesignoptimizer.sequencer.app import layout
from qdesignoptimizer.sequencer.app.queue_state import (
    NO_PROJECT,
    Bridge,
    GuiError,
    repo_label,
)
from qdesignoptimizer.sequencer.engine.workspace import Workspace

from .helpers import make_project

ADAPTER = (
    "from pathlib import Path\n"
    "from qdesignoptimizer.sequencer.backends.fake import FakeBackend\n"
    "def make_backend():\n"
    "    return FakeBackend(Path(__file__).parent.parent)\n"
)


def _folder(tmp_path, name):
    folder = tmp_path / name
    make_project(folder)
    (folder / "simulations").mkdir()
    (folder / "simulations" / "sequencer.py").write_text(ADAPTER, "utf-8")
    return folder


def test_connect_and_disconnect_are_remembered(tmp_path):
    remember = tmp_path / "home" / "connected.json"
    bridge = Bridge([], remember=remember)
    try:
        assert bridge.snapshot()["empty"] == NO_PROJECT
        a = _folder(tmp_path, "chip_a")
        assert bridge.connect(str(a)) == "connected chip_a"
        snap = bridge.snapshot()
        assert [p["label"] for p in snap["projects"]] == ["chip_a"]
        assert snap["projects"][0]["title"] == str(a.resolve())
        assert json.loads(remember.read_text("utf-8")) == [str(a.resolve())]
        with pytest.raises(GuiError, match="connected already"):
            bridge.connect(str(a / "simulations" / "sequencer.py"))
        with pytest.raises(GuiError, match="does not exist"):
            bridge.connect(str(tmp_path / "nope"))
        b = _folder(tmp_path, "chip_b")
        bridge.connect(str(b))
        assert {p["label"] for p in bridge.snapshot()["projects"]} == {
            "chip_a",
            "chip_b",
        }
        assert "disconnected chip_a" in bridge.disconnect(str(a.resolve()))
        assert json.loads(remember.read_text("utf-8")) == [str(b.resolve())]
        bridge.disconnect(str(b.resolve()))
        assert bridge.snapshot()["empty"] == NO_PROJECT
    finally:
        bridge.close()


def test_a_project_is_named_relative_to_its_repository(tmp_path):
    (tmp_path / "repo" / ".git").mkdir(parents=True)
    folder = tmp_path / "repo" / "designs" / "chip"
    folder.mkdir(parents=True)
    assert repo_label(folder) == "designs/chip"
    assert repo_label(tmp_path / "loose") == "loose"


def test_the_left_side_shows_only_the_project_worked_on(tmp_path):
    a = make_project(tmp_path / "a")
    b = make_project(tmp_path / "b")
    bridge = Bridge(workspace=Workspace([a, b], state_dir=tmp_path / "state"))
    bridge.new_campaign("camp_a", project="a")
    bridge.new_campaign("camp_b", project="b")
    b.save_sequence("only_b", {"name": "Only b", "studies": ["A.calibrate"]})
    snap = bridge.snapshot()
    parts = layout.browser_parts(snap, None, "a")
    assert [o["value"] for o in parts["campaigns"]] == ["camp_a"]
    assert "Only b" not in str(parts["sequences"])
    parts = layout.browser_parts(snap, None, "b")
    assert [o["value"] for o in parts["campaigns"]] == ["camp_b"]
    # No "b / " in front: only one project is shown.
    assert "Only b" in str(parts["sequences"]) and "b / Only b" not in str(
        parts["sequences"]
    )


def test_new_study_starts_in_the_campaign_worked_in(project):
    project.new_campaign("tuneup_chip")
    bridge = Bridge(project=project)
    text = str(layout.new_study_view(bridge.snapshot(), "", "tuneup_chip"))
    assert "'value': 'tuneup_chip'" in text or "value='tuneup_chip'" in text
    assert "add to queue" in text
    template = str(layout.new_study_view(bridge.snapshot(), "", template=True))
    assert "Save as template" in template and "Start the template" in template


@pytest.fixture
def project(tmp_path):
    return make_project(tmp_path / "p")
