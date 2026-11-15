"""The GUI's left side (2026-11-15): + New study, study sequence
templates, campaigns and study templates in the editor."""

from __future__ import annotations

import pytest
import yaml

from qdesignoptimizer.sequencer.app import layout
from qdesignoptimizer.sequencer.app.queue_state import Bridge, GuiError
from qdesignoptimizer.sequencer.engine.queue import QueueError

from .helpers import make_project


@pytest.fixture
def project(tmp_path):
    return make_project(tmp_path / "proj")


@pytest.fixture
def bridge(project):
    return Bridge(project=project)


def _form(**over):
    form = {
        "name": "Calibrate both",
        "why_ordered": "A first: B reads what it wrote",
        "campaign": "",
        "defaults": {
            "settings": "seconds: 0",
            "checks": "converged",
            "save": True,
            "on_failure": "continue",
        },
        "studies": [
            {"task": "A.calibrate", "settings": "", "notes": "", "extra": {}},
            {
                "task": "B.calibrate",
                "settings": "seconds: 1",
                "notes": "slow",
                "extra": {},
            },
        ],
    }
    form.update(over)
    return form


def test_a_new_study_starts_on_hold_from_a_template(bridge, project):
    key = bridge.new_study(task="A.calibrate")
    assert key.startswith("@hold/")
    assert project.queue.ids() == []
    (held,) = project.hold.ids()
    assert key == f"@hold/{held}"
    assert bridge.details("hold", key)["job"]["task"] == "A.calibrate"
    with pytest.raises(GuiError, match="does not exist"):
        bridge.new_study(task="A.calibrate", campaign="nope")


def test_a_sequence_template_is_saved_read_back_and_queued(bridge, project):
    assert "saved study_sequences/both.yaml" == bridge.save_sequence(
        "both", _form(), overwrite=False
    )
    raw = yaml.safe_load((project.root / "study_sequences" / "both.yaml").read_text())
    assert raw["defaults"] == {
        "settings": {"seconds": 0},
        "checks": ["converged"],
        "save": True,
        "on_failure": "continue",
    }
    assert raw["studies"][1] == {
        "task": "B.calibrate",
        "settings": {"seconds": 1},
        "notes": "slow",
    }
    view = bridge.sequence_view("both")
    assert view["name"] == "Calibrate both" and not view["new"]
    assert [s["task"] for s in view["studies"]] == ["A.calibrate", "B.calibrate"]
    assert view["studies"][1]["settings"] == "seconds: 1"
    assert view["defaults"]["checks"] == "converged"

    # In the catalog at once, and queued like any study sequence.
    assert "queued 2" in bridge.add("sequence", "both")
    jobs = [project.job(i) for i in project.queue.ids()]
    assert [j.settings for j in jobs] == [{"seconds": 0}, {"seconds": 1}]
    assert all(j.on_failure == "continue" for j in jobs)


def test_saving_refuses_what_would_not_read_back(bridge):
    with pytest.raises(GuiError, match="unknown"):
        bridge.save_sequence(
            "bad",
            _form(
                studies=[
                    {"task": "Z.nothing", "settings": "", "notes": "", "extra": {}}
                ]
            ),
            overwrite=False,
        )
    with pytest.raises(GuiError, match="at least one"):
        bridge.save_sequence("empty", _form(studies=[]), overwrite=False)
    with pytest.raises(GuiError, match="name: value"):
        bridge.save_sequence(
            "x", _form(defaults={"settings": "[1, 2]"}), overwrite=False
        )
    with pytest.raises(GuiError, match="letters, digits"):
        bridge.save_sequence("has space", _form(), overwrite=False)
    with pytest.raises(GuiError, match="does not exist"):
        bridge.save_sequence("c", _form(campaign="nope"), overwrite=False)
    bridge.save_sequence("once", _form(), overwrite=False)
    with pytest.raises(GuiError, match="exists already"):
        bridge.save_sequence("once", _form(), overwrite=False)
    assert "saved" in bridge.save_sequence("once", _form(name="again"), overwrite=True)


def test_a_sequence_built_in_code_is_shown_but_saved_under_a_new_name(bridge):
    view = bridge.sequence_view("full_A")
    assert view["from_code"] is True
    assert [s["task"] for s in view["studies"]] == ["A.calibrate", "A.measure"]
    with pytest.raises(GuiError, match="adapter's code"):
        bridge.save_sequence("full_A", _form(), overwrite=False)
    layout.sequence_editor(view)  # draws


def test_per_study_keys_the_editor_does_not_show_are_kept(bridge, project):
    folder = project.root / "study_sequences"
    folder.mkdir()
    (folder / "kept.yaml").write_text(
        "name: kept\nstudies:\n- {task: A.calibrate, on_failure: continue, save: true}\n",
        "utf-8",
    )
    view = bridge.sequence_view("kept")
    assert view["studies"][0]["extra"] == {"on_failure": "continue", "save": True}
    form = _form(studies=view["studies"])
    bridge.save_sequence("kept", form, overwrite=True)
    raw = yaml.safe_load((folder / "kept.yaml").read_text())
    assert raw["studies"][0]["on_failure"] == "continue"


def test_the_browser_lists_campaigns_templates_and_sequences(bridge, project):
    project.new_campaign("tuneup_chip", "Can it all be tuned?")
    bridge.save_sequence("both", _form(), overwrite=False)
    snap = bridge.snapshot()
    sources = {
        c["key"]: c.get("source") for c in snap["catalog"] if c["kind"] == "sequence"
    }
    assert sources == {"full_A": "", "both": "study_sequences/both.yaml"}
    every = {"campaigns": True, "sequences": True, "templates": True}
    text = str(layout.browser(snap, {"kind": "sequence", "key": "both"}, every))
    for word in (
        "tuneup_chip",
        "Calibrate both",
        "A calibrate",
        "in the adapter's code",
    ):
        assert word in text
    # Templates show the name they are saved under: two may share a label.
    assert "both · 2 studies" in text and "full_A · 2 studies" in text
    # A closed section lists nothing (its state is kept in a store).
    closed = str(layout.browser(snap, None, {"sequences": False}))
    assert "Calibrate both" not in closed and "▸ Study sequence templates" in closed


def test_the_campaign_view_counts_its_studies(bridge, project):
    project.new_campaign("tuneup_chip", "Can it all be tuned?")
    project.add("A.calibrate", campaign="tuneup_chip")
    project.add("B.calibrate", campaign="tuneup_chip", hold=True)
    view = bridge.campaign_view("tuneup_chip")
    assert view["counts"] == {"queued": 1, "on hold": 1, "completed": 0}
    assert "Can it all be tuned?" in view["readme"]
    assert view["store"] == "campaigns/tuneup_chip/store.json"
    assert view["promote_to"] == "store.json"
    layout.campaign_view(view)
    with pytest.raises(GuiError, match="no campaign"):
        bridge.campaign_view("nope")


def test_the_template_view_and_new_study_views_draw(bridge):
    view = bridge.template_view("A.calibrate")
    assert view["label"] == "A calibrate"
    layout.template_view(view)
    layout.new_study_view(bridge.snapshot())
    layout.new_campaign_view()
    with pytest.raises(GuiError):
        bridge.template_view("Z.nothing")


def test_the_engine_refuses_a_sequence_where_none_can_be_kept(project, monkeypatch):
    monkeypatch.setattr(
        type(project.backend), "sequence_dir", property(lambda self: None)
    )
    with pytest.raises(QueueError, match="keeps no"):
        project.save_sequence("x", {"studies": ["A.calibrate"]})


def test_a_template_is_saved_through_a_worker_process(tmp_path):
    """The GUI reaches every project through a worker process; the
    folder a template is saved in must cross with it."""
    from qdesignoptimizer.sequencer.engine.workspace import open_workspace

    folder = tmp_path / "a"
    folder.mkdir()
    (folder / "sequencer.py").write_text(
        "from pathlib import Path\n"
        "from qdesignoptimizer.sequencer.backends.fake import FakeBackend\n"
        "def make_backend():\n"
        "    return FakeBackend(Path(__file__).parent)\n",
        "utf-8",
    )
    workspace = open_workspace([folder / "sequencer.py"], isolate=True)
    try:
        bridge = Bridge(workspace=workspace)
        assert type(bridge.project.backend).__name__ == "RemoteBackend"
        assert "saved" in bridge.save_sequence("w", _form(), overwrite=False)
        assert (folder / "study_sequences" / "w.yaml").is_file()
    finally:
        workspace.close()


# -- review of 2026-11-15 ---------------------------------------------------


def test_the_template_view_has_every_field_the_make_study_callback_reads(bridge):
    text = str(layout.template_view(bridge.template_view("A.calibrate")))
    for name in (
        "new-study-template",
        "new-study-name",
        "new-study-campaign",
        "new-study-task",
    ):
        assert name in text


def test_promote_in_the_campaign_view_names_project_and_campaign(bridge, project):
    project.new_campaign("tuneup_chip")
    text = str(layout.campaign_view(bridge.campaign_view("tuneup_chip")))
    assert "'index': 'proj::tuneup_chip'" in text


def test_an_unchanged_template_saves_back_as_it_was(bridge, project):
    folder = project.root / "study_sequences"
    folder.mkdir()
    (folder / "same.yaml").write_text(
        "name: same\nnotes: why this exists\n"
        "defaults: {checks: converged, notes: shared, save: true}\n"
        "studies: [A.calibrate]\n",
        "utf-8",
    )
    view = bridge.sequence_view("same")
    assert view["defaults"]["checks"] == "converged"
    form = {
        "name": view["name"],
        "why_ordered": view["why_ordered"],
        "campaign": view["campaign"],
        "defaults": view["defaults"],
        "studies": view["studies"],
        "keep": view["keep"],
    }
    bridge.save_sequence("same", form, overwrite=True)
    raw = yaml.safe_load((folder / "same.yaml").read_text())
    assert raw["notes"] == "why this exists"
    assert raw["defaults"]["checks"] == ["converged"]
    assert raw["defaults"]["notes"] == "shared"


def test_file_names_with_a_dash_save_and_a_newline_is_refused(bridge):
    assert "saved" in bridge.save_sequence(
        "tune-R1", _form(name="Dashed"), overwrite=False
    )
    with pytest.raises(GuiError, match="letters, digits"):
        bridge.save_sequence("abc\n", _form(), overwrite=False)


def test_a_file_named_like_a_code_sequence_is_not_shown_as_the_template(
    bridge, project
):
    folder = project.root / "study_sequences"
    folder.mkdir()
    (folder / "full_A.yaml").write_text(
        "name: shadow\nstudies: [B.calibrate]\n", "utf-8"
    )
    view = bridge.sequence_view("full_A")
    assert view["from_code"] is True
    assert [s["task"] for s in view["studies"]] == ["A.calibrate", "A.measure"]


def test_the_new_flows_work_with_several_projects_open(tmp_path):
    from qdesignoptimizer.sequencer.engine.workspace import Workspace

    a = make_project(tmp_path / "a")
    b = make_project(tmp_path / "b")
    bridge = Bridge(workspace=Workspace([a, b], state_dir=tmp_path / "state"))
    assert bridge.sequence_view("", "b")["project"] == "b"
    key = bridge.new_study(task="b/A.calibrate")
    assert key.startswith("b/@hold/") and b.hold.ids()
    bridge.new_campaign("camp", project="b")
    assert [c.key for c in b.campaigns()] == ["camp"] and a.campaigns() == []
    assert "saved" in bridge.save_sequence("s", _form(), overwrite=False, project="b")
    assert (tmp_path / "b" / "study_sequences" / "s.yaml").is_file()
    text = str(layout.new_study_view(bridge.snapshot(), "b"))
    assert "b/A.calibrate" in text and "a/A.calibrate" not in text


def test_an_untouched_sequence_editor_is_not_unsaved(bridge):
    """What the editor draws and what its fields then report compare
    equal, so moving away asks only after a real edit."""
    bridge.save_sequence("both", _form(), overwrite=False)
    view = bridge.sequence_view("both")
    drawn = layout.sequence_fields(view)
    live = layout.seq_state(
        {
            "key": "both",
            "name": view["name"],
            "why_ordered": view["why_ordered"],
            "campaign": None,
            "d_settings": view["defaults"]["settings"],
            "d_checks": view["defaults"]["checks"],
            "d_save": ["save"],
            "d_on_failure": view["defaults"]["on_failure"],
        },
        [dict(r) for r in view["studies"]],
    )
    assert live == drawn
    live["rows"][0][2] = "a note"
    assert live != drawn


def test_every_template_has_its_own_name(bridge):
    bridge.save_sequence("one", _form(name="Calibrate"), overwrite=False)
    with pytest.raises(GuiError, match="taken by 'one'"):
        bridge.save_sequence("two", _form(name="Calibrate"), overwrite=False)
    # The name of one built in code counts too.
    with pytest.raises(GuiError, match="taken by 'full_A'"):
        bridge.save_sequence(
            "copy", _form(name="Calibrate then measure A"), overwrite=False
        )
    # Saving a template again under its own name is fine.
    assert "saved" in bridge.save_sequence(
        "one", _form(name="Calibrate"), overwrite=True
    )


def test_a_name_is_checked_against_the_files_not_a_stale_catalog(project):
    """Review round 4: a worker's catalog may lag a save just made."""
    stale = project.backend.catalog()
    project.save_sequence("one", {"name": "Calibrate", "studies": ["A.calibrate"]})
    project.backend.catalog = lambda: stale
    with pytest.raises(QueueError, match="taken by 'one'"):
        project.save_sequence("two", {"name": "Calibrate", "studies": ["A.calibrate"]})


def test_a_new_template_never_replaces_one_saved_meanwhile(project):
    from qdesignoptimizer.sequencer.engine.sequence import SequenceError, save_sequence

    folder = project.sequence_dir
    save_sequence(folder, "one", {"studies": ["A.calibrate"]})
    with pytest.raises(SequenceError, match="exists already"):
        save_sequence(folder, "one", {"studies": ["A.calibrate"]})
    assert not list(folder.glob(".*.tmp"))


# -- review round 4 -----------------------------------------------------------


def test_keys_with_spaces_survive_a_queued_message(tmp_path):
    from qdesignoptimizer.sequencer.app.callbacks import _queued_key
    from qdesignoptimizer.sequencer.engine.workspace import Workspace

    a = make_project(tmp_path / "chip one")
    b = make_project(tmp_path / "b")
    bridge = Bridge(workspace=Workspace([a, b], state_dir=tmp_path / "state"))
    key = bridge.new_study(task="chip one/A.calibrate")
    queued = _queued_key(bridge.unpark(key))
    assert queued.startswith("chip one/") and bridge.details("queue", queued)
    assert _queued_key("queued x/1 -- warning, x/1: y") == "x/1"


def test_the_add_bar_keeps_campaigns_of_projects_apart(tmp_path):
    from qdesignoptimizer.sequencer.engine.workspace import Workspace

    a = make_project(tmp_path / "a")
    b = make_project(tmp_path / "b")
    bridge = Bridge(workspace=Workspace([a, b], state_dir=tmp_path / "state"))
    bridge.new_campaign("camp", project="a")
    bridge.new_campaign("camp", project="b")
    values = [o["value"] for o in layout.campaign_options(bridge.snapshot())]
    assert sorted(values) == ["a::camp", "b::camp"]
    bridge.add("task", "b/A.calibrate", "b::camp")
    assert [r["campaign"] for r in bridge.snapshot()["rows"]] == ["camp"]
    with pytest.raises(GuiError, match="is in a, not in b"):
        bridge.add("task", "b/A.calibrate", "a::camp")


def test_a_bad_settings_value_in_a_template_is_shown_not_crashed(project, bridge):
    (project.sequence_dir).mkdir(exist_ok=True)
    (project.sequence_dir / "odd.yaml").write_text(
        "version: 1\ndefaults: {settings: 3}\nstudies: [A.calibrate]\n", "utf-8"
    )
    with pytest.raises(GuiError):
        bridge.sequence_view("odd")
