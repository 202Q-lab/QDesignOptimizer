"""Campaigns: lab-log folders with their own store, nested."""

from __future__ import annotations

import json

import pytest
import yaml

from qdesignoptimizer.sequencer.app.queue_state import GuiError
from qdesignoptimizer.sequencer.engine import campaign as cp
from qdesignoptimizer.sequencer.engine import cli
from qdesignoptimizer.sequencer.engine import state as st
from qdesignoptimizer.sequencer.engine.backend import SequenceEntry, SequenceSpec
from qdesignoptimizer.sequencer.engine.queue import QueueError
from qdesignoptimizer.sequencer.engine.runner import Runner

from .helpers import make_project, set_task, store


@pytest.fixture
def project(tmp_path):
    return make_project(tmp_path / "proj")


def _store_of(project, key):
    return json.loads(project.campaign_store(key).read_text("utf-8"))


def test_keys_are_checked_and_normalised():
    assert cp.check_key("tuneup_chip\\q1_readout/") == "tuneup_chip/q1_readout"
    for bad in ("", "Tuneup", "2026-11-14_tuneup", "a b", "x/../y", "_x"):
        with pytest.raises(cp.CampaignError):
            cp.check_key(bad)


def test_a_new_campaign_gets_a_readme_and_its_own_store(project):
    folder = project.new_campaign("tuneup_chip", "Can every mode hit its target?")
    readme = (folder / cp.README).read_text("utf-8")
    assert readme.startswith("# Tuneup chip")
    assert "Can every mode hit its target?" in readme
    assert project.campaign_store("tuneup_chip") == folder / "store.json"
    assert _store_of(project, "tuneup_chip") == store(project)
    (only,) = project.campaigns()
    assert only.key == "tuneup_chip" and only.own_store and only.depth == 0


def test_a_sub_campaign_needs_its_parent_and_can_share_its_store(project):
    with pytest.raises(QueueError, match="make campaign 'tuneup_chip' first"):
        project.new_campaign("tuneup_chip/q1_readout")
    project.new_campaign("tuneup_chip")
    project.new_campaign("tuneup_chip/q1_readout", own_store=False)
    with pytest.raises(QueueError, match="exists already"):
        project.new_campaign("tuneup_chip")
    keys = [(c.key, c.own_store, c.parent) for c in project.campaigns()]
    assert keys == [
        ("tuneup_chip", True, ""),
        ("tuneup_chip/q1_readout", False, "tuneup_chip"),
    ]
    # No store of its own: the parent's.
    assert project.campaign_store("tuneup_chip/q1_readout") == project.campaign_store(
        "tuneup_chip"
    )


def test_a_sub_campaign_copies_its_parents_store_not_the_projects(project):
    project.new_campaign("tuneup_chip")
    path = project.campaign_store("tuneup_chip")
    path.write_text(json.dumps({"A_value": 7.0, "B_value": 0.0}), "utf-8")
    project.new_campaign("tuneup_chip/q1_readout")
    assert _store_of(project, "tuneup_chip/q1_readout")["A_value"] == 7.0


def test_a_campaign_study_writes_its_campaigns_store_only(project):
    project.new_campaign("tuneup_chip")
    job_id = project.add("A.calibrate", campaign="tuneup_chip")
    set_task(project, job_id, value=3.5)
    job = yaml.safe_load((project.queue.folder(job_id) / "job.yaml").read_text())
    assert job["campaign"] == "tuneup_chip"
    # The id says nothing of the campaign: solver paths stay short.
    assert "tuneup" not in job_id
    project.park(job_id)  # moving it keeps it
    project.unpark(job_id)
    assert project.job(job_id).campaign == "tuneup_chip"

    q = project.queue.folder(job_id)
    (q / "job.yaml").write_text(
        (q / "job.yaml").read_text("utf-8").replace("save: false", "save: true"),
        "utf-8",
    )
    Runner(project).run()
    assert _store_of(project, "tuneup_chip")["A_value"] == 3.5
    assert store(project)["A_value"] == 0.0  # the project's is untouched
    (done,) = project.queue.completed()
    record = json.loads((done / st.RESULT_FILE).read_text("utf-8"))
    assert record["status"] == st.DONE and record["campaign"] == "tuneup_chip"
    # It started from the campaign's values, and the write is recorded
    # next to the campaign's store.
    meta = json.loads(
        (project.campaign_store("tuneup_chip").with_name("store.meta.json")).read_text(
            "utf-8"
        )
    )
    assert meta["A_value"]["latest"]["campaign"] == "tuneup_chip"


def test_an_unknown_campaign_is_refused_when_queued_and_when_checked(project):
    with pytest.raises(QueueError, match="does not exist"):
        project.add("A.calibrate", campaign="nowhere")
    project.new_campaign("tuneup_chip")
    job_id = project.add("A.calibrate", campaign="tuneup_chip")
    (project.root / "campaigns" / "tuneup_chip" / cp.README).unlink()
    assert any("does not exist" in p for p in project.problems(job_id))


def test_a_sequence_queues_its_studies_in_its_campaign(project):
    project.new_campaign("mesh_sensitivity")
    spec = SequenceSpec(
        key="sweep",
        label="sweep",
        entries=(SequenceEntry("A.calibrate"), SequenceEntry("B.calibrate")),
        campaign="mesh_sensitivity",
    )
    ids = project.add_sequence(spec)
    assert [project.job(i).campaign for i in ids] == ["mesh_sensitivity"] * 2


def test_a_sequence_file_names_its_campaign(project, tmp_path):
    from qdesignoptimizer.sequencer.engine.sequence import load_sequence

    path = tmp_path / "tune.yaml"
    path.write_text("campaign: tuneup_chip\\q1\nstudies: [A.calibrate]\n", "utf-8")
    assert load_sequence(path).campaign == "tuneup_chip/q1"


def test_promote_copies_changed_values_one_level_up(project):
    project.new_campaign("tuneup_chip")
    project.new_campaign("tuneup_chip/q1_readout")
    sub = project.campaign_store("tuneup_chip/q1_readout")
    sub.write_text(json.dumps({"A_value": 2.0, "B_value": 5.0}), "utf-8")

    assert project.promote("tuneup_chip/q1_readout", ["A_value"]) == {"A_value": 2.0}
    assert _store_of(project, "tuneup_chip") == {"A_value": 2.0, "B_value": 0.0}
    assert store(project)["A_value"] == 0.0  # one level, not to the top

    assert project.promote("tuneup_chip/q1_readout") == {"B_value": 5.0}
    assert project.promote("tuneup_chip/q1_readout") == {}  # nothing left
    assert project.promote("tuneup_chip") == {"A_value": 2.0, "B_value": 5.0}
    assert store(project) == {"A_value": 2.0, "B_value": 5.0}
    meta = json.loads((project.root / "store.meta.json").read_text("utf-8"))
    assert meta["A_value"]["latest"]["promoted_from"] == "tuneup_chip"
    assert project.backend.value_sources("design_variables.json")["A_value"].startswith(
        "promoted from campaign tuneup_chip"
    )


def test_promote_refuses_a_campaign_without_its_own_store(project):
    project.new_campaign("tuneup_chip", own_store=False)
    with pytest.raises(QueueError, match="no store of its own"):
        project.promote("tuneup_chip")
    project.new_campaign("other")
    with pytest.raises(QueueError, match="not in campaign"):
        project.promote("other", ["nope"])


def test_the_cli_makes_lists_and_promotes(project, capsys):
    adapter = project.root / "sequencer.py"
    adapter.write_text(
        "from qdesignoptimizer.sequencer.backends.fake import FakeBackend\n"
        "from pathlib import Path\n"
        "def make_backend():\n"
        "    return FakeBackend(Path(__file__).parent)\n",
        "utf-8",
    )
    ref = str(adapter)
    assert cli.main([ref, "campaign", "new", "tuneup_chip", "--question", "Q?"]) == 0
    assert cli.main([ref, "campaign", "new", "tuneup_chip/q1", "--shared"]) == 0
    assert cli.main([ref, "campaign", "list"]) == 0
    out = capsys.readouterr().out
    assert "tuneup_chip  Tuneup chip  (store: campaigns/tuneup_chip/store.json)" in out
    assert "  tuneup_chip/q1  Q1  (store: campaigns/tuneup_chip/store.json)" in out
    assert cli.main([ref, "add", "A.calibrate", "--campaign", "tuneup_chip/q1"]) == 0
    assert cli.main([ref, "add", "A.calibrate", "--campaign", "nope"]) == 1
    assert cli.main([ref, "campaign", "promote", "tuneup_chip"]) == 0
    assert "nothing to promote" in capsys.readouterr().out


# -- the GUI --------------------------------------------------------------


def test_the_gui_lists_campaigns_and_tags_their_studies(project):
    from qdesignoptimizer.sequencer.app import layout
    from qdesignoptimizer.sequencer.app.queue_state import Bridge

    bridge = Bridge(project=project)
    assert "made campaign tuneup_chip" in bridge.new_campaign("tuneup_chip", "Q?")
    bridge.new_campaign("tuneup_chip/q1", own_store=False)
    snap = bridge.snapshot()
    assert [
        (c["key"], c["depth"], c["own_store"], c["store"]) for c in snap["campaigns"]
    ] == [
        ("tuneup_chip", 0, True, "campaigns/tuneup_chip/store.json"),
        ("tuneup_chip/q1", 1, False, "campaigns/tuneup_chip/store.json"),
    ]
    assert [o["value"] for o in layout.campaign_options(snap)] == [
        "tuneup_chip",
        "tuneup_chip/q1",
    ]
    assert "tuneup_chip" in str(layout.browser(snap, None))

    bridge.add("task", "A.calibrate", "tuneup_chip/q1")
    (row,) = bridge.snapshot()["rows"]
    assert row["campaign"] == "tuneup_chip/q1"
    assert "campaign tuneup_chip/q1" in str(layout.queue_table(bridge.snapshot(), None))
    with pytest.raises(GuiError, match="does not exist"):
        bridge.add("task", "A.calibrate", "nope")

    job_id = row["id"]
    queue = project.queue.folder(job_id)
    (queue / "job.yaml").write_text(
        (queue / "job.yaml").read_text("utf-8").replace("save: false", "save: true"),
        "utf-8",
    )
    Runner(project).run()
    (done,) = bridge.snapshot()["completed"]
    assert done["campaign"] == "tuneup_chip/q1"


def test_the_gui_moves_a_queued_study_into_or_out_of_a_campaign(project):
    from qdesignoptimizer.sequencer.app.queue_state import Bridge

    bridge = Bridge(project=project)
    project.new_campaign("tuneup_chip")
    job_id = project.add("A.calibrate")
    assert bridge.details("queue", job_id)["campaigns"] == ["tuneup_chip"]
    bridge.save_job(job_id, {"campaign": "tuneup_chip"})
    assert project.job(job_id).campaign == "tuneup_chip"
    with pytest.raises(GuiError, match="does not exist"):
        bridge.save_job(job_id, {"campaign": "nope"})
    bridge.save_job(job_id, {"campaign": ""})
    assert project.job(job_id).campaign == ""
    assert "campaign" not in (project.queue.folder(job_id) / "job.yaml").read_text()


def test_the_gui_promotes_a_campaign(project):
    from qdesignoptimizer.sequencer.app.queue_state import Bridge

    bridge = Bridge(project=project)
    project.new_campaign("tuneup_chip")
    assert bridge.promote("tuneup_chip").startswith("nothing to promote")
    project.campaign_store("tuneup_chip").write_text(
        json.dumps({"A_value": 4.0, "B_value": 0.0}), "utf-8"
    )
    assert bridge.promote("tuneup_chip") == "promoted to store.json: A_value"
    assert store(project)["A_value"] == 4.0
    project.new_campaign("shared", own_store=False)
    with pytest.raises(GuiError, match="no store of its own"):
        bridge.promote("shared")


def test_promote_carries_up_only_what_the_campaign_changed(project):
    """Review 2026-11-15: a campaign's copy of a value the parent changed
    since must not overwrite the parent's newer value."""
    project.new_campaign("tuneup_chip")
    path = project.campaign_store("tuneup_chip")
    # The project moves on after the campaign was made ...
    project.commit({"A_value": 5.0})
    # ... and the campaign tunes B only.
    data = json.loads(path.read_text("utf-8"))
    data["B_value"] = 2.0
    path.write_text(json.dumps(data), "utf-8")
    assert project.promote("tuneup_chip") == {"B_value": 2.0}
    assert store(project) == {"A_value": 5.0, "B_value": 2.0}
    # Promoted: the new starting point, so nothing is left to promote.
    assert project.promote("tuneup_chip") == {}
    # Asking for a name by name promotes it regardless.
    assert project.promote("tuneup_chip", ["A_value"]) == {"A_value": 0.0}


def test_a_failed_promote_is_promoted_again(project, monkeypatch):
    """Review round 4: the baseline moves only once the parent holds the
    values, so a commit that fails is not forgotten."""
    project.new_campaign("tuneup_chip")
    path = project.campaign_store("tuneup_chip")
    data = json.loads(path.read_text("utf-8"))
    data["B_value"] = 2.0
    path.write_text(json.dumps(data), "utf-8")

    def broken(*_args, **_kwargs):
        raise OSError("store locked")

    monkeypatch.setattr(project, "commit", broken)
    with pytest.raises(OSError):
        project.promote("tuneup_chip")
    monkeypatch.undo()
    assert project.promote("tuneup_chip") == {"B_value": 2.0}


def test_a_campaign_without_a_baseline_gets_one_at_promote(project):
    project.new_campaign("tuneup_chip")
    path = project.campaign_store("tuneup_chip")
    (path.parent / ".promoted.json").unlink()
    data = json.loads(path.read_text("utf-8"))
    data["B_value"] = 2.0
    path.write_text(json.dumps(data), "utf-8")
    assert project.promote("tuneup_chip") == {"B_value": 2.0}
    # From here on only what the campaign changes goes up.
    project.commit({"A_value": 5.0})
    assert project.promote("tuneup_chip") == {}
    assert store(project)["A_value"] == 5.0
