"""What an agent working next to the GUI needs (2026-11-16): who queued
a study and why, shown in the GUI; ``wait`` for results; a newly
connected project joining the order there was; On hold findable."""

from __future__ import annotations

import io
import json

import pytest

from qdesignoptimizer.sequencer.app import layout
from qdesignoptimizer.sequencer.app.queue_state import Bridge
from qdesignoptimizer.sequencer.engine import cli
from qdesignoptimizer.sequencer.engine import state as st
from qdesignoptimizer.sequencer.engine.backend import SequenceEntry, SequenceSpec
from qdesignoptimizer.sequencer.engine.runner import Runner
from qdesignoptimizer.sequencer.engine.workspace import Workspace

from .helpers import make_project, set_task

ADAPTER = (
    "from qdesignoptimizer.sequencer.backends.fake import FakeBackend\n"
    "from pathlib import Path\n"
    "def make_backend():\n"
    "    return FakeBackend(Path(__file__).parent)\n"
)


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    made = make_project(tmp_path / "p")
    (made.root / "sequencer.py").write_text(ADAPTER, "utf-8")
    return made


def _task(project):
    return project.backend.catalog().tasks[0].key


def _cli(project, *args):
    return cli.main([str(project.root / "sequencer.py"), *args])


# -- who queued it and why ----------------------------------------------------


def test_the_command_line_says_who_queued_it_and_why(project, capsys):
    assert (
        _cli(project, "add", _task(project), "--by", "claude", "--why", "kappa off")
        == 0
    )
    added = capsys.readouterr().out.split("queued ")[1].strip()
    job = project.job(added)
    assert (job.added_by, job.why) == ("claude", "kappa off")
    bridge = Bridge(project=project)
    snap = bridge.snapshot()
    (row,) = snap["rows"]
    assert (row["added_by"], row["why"]) == ("claude", "kappa off")
    shown = str(layout.queue_table(snap, {}))
    assert "by claude" in shown and "kappa off" in shown
    head = str(layout.details(bridge.details("queue", added), {}))
    assert "Added by claude" in head and "kappa off" in head
    # A GUI save keeps both.
    bridge.save_job(added, {"notes": "checked"})
    assert project.job(added).added_by == "claude"
    # The run record carries who queued it, for the completed list.
    Runner(project).run()
    (done,) = Bridge(project=project).snapshot()["completed"]
    assert done["added_by"] == "claude"


def test_the_default_is_the_agent_variable_or_the_command_line(
    project, capsys, monkeypatch
):
    assert _cli(project, "add", _task(project)) == 0
    added = capsys.readouterr().out.split("queued ")[1].strip()
    assert project.job(added).added_by in ("the command line", cli._AGENT)
    # A study made in the GUI has no tag.
    gui = project.add(_task(project))
    assert project.job(gui).added_by == ""
    rows = {r["id"]: r for r in Bridge(project=project).snapshot()["rows"]}
    assert "by " not in str(layout._by_tag(rows[gui]))


def test_a_sequence_marks_each_of_its_studies(project):
    task = _task(project)
    spec = SequenceSpec("s", "S", (SequenceEntry(task), SequenceEntry(task)))
    ids = project.add_sequence(spec, added_by="claude", why="retune")
    assert {(project.job(i).added_by, project.job(i).why) for i in ids} == {
        ("claude", "retune")
    }


# -- wait ------------------------------------------------------------------------


def test_wait_prints_the_result_and_its_folder(project, capsys):
    job_id = project.add(_task(project))
    Runner(project).run()
    assert _cli(project, "wait", job_id) == 0
    out = capsys.readouterr().out
    assert out.startswith(f"done {job_id}_")
    (folder,) = project.queue.completed()
    assert f"folder: {folder.resolve()}" in out


def test_wait_says_failed_held_and_unknown(project, capsys):
    bad = project.add(_task(project), on_failure="continue")
    set_task(project, bad, fail=True)
    held = project.add(_task(project), hold=True)
    Runner(project).run()
    assert _cli(project, "wait", bad) == 1
    assert capsys.readouterr().out.startswith(f"failed {bad}_")
    assert _cli(project, "wait", held) == 1
    assert "on hold" in capsys.readouterr().out
    assert _cli(project, "wait", "nothing-like-it") == 2


def test_wait_times_out_while_queued(project, capsys):
    job_id = project.add(_task(project))
    assert _cli(project, "wait", job_id, "--timeout", "0.05", "--every", "0.01") == 3
    out = capsys.readouterr().out
    assert "no runner is going" in out and f"still queued: {job_id}" in out


def test_wait_follows_a_requeued_copy(project, capsys):
    job_id = project.add(_task(project))
    Runner(project).run()
    (first,) = project.queue.completed()
    copy = project.queue.requeue(first, project.words)
    # The copy is queued: an old success does not end the wait.
    assert _cli(project, "wait", job_id, "--timeout", "0.05", "--every", "0.01") == 3
    Runner(project).run()
    capsys.readouterr()
    assert _cli(project, "wait", job_id) == 0
    assert capsys.readouterr().out.startswith(f"done {copy}_")


# -- a project connected later --------------------------------------------------


def test_a_project_connected_later_comes_after_the_order_there_was(tmp_path):
    a, b, c = (make_project(tmp_path / n) for n in ("a", "b", "c"))
    a1, a2 = a.add(_task(a)), a.add(_task(a))
    b1 = b.add(_task(b))
    two = Workspace([a, b])
    two.move(f"b/{b1}", to=1)
    assert two.ids() == [f"b/{b1}", f"a/{a1}", f"a/{a2}"]
    c1 = c.add(_task(c))
    three = Workspace([a, b, c])
    assert three.state_dir != two.state_dir
    assert three.ids() == [f"b/{b1}", f"a/{a1}", f"a/{a2}", f"c/{c1}"]
    # Disconnecting one keeps the rest of the order too.
    three.move(f"a/{a2}", to=1)
    again = Workspace([a, c])
    assert again.ids()[:2] == [f"a/{a2}", f"a/{a1}"]


# -- On hold findable -------------------------------------------------------------


def test_the_sections_show_their_counts_and_link_to_each_other(project):
    project.add(_task(project))
    project.add(_task(project), hold=True)
    project.add(_task(project), hold=True)
    queue, hold, completed, jump = layout.section_titles(
        Bridge(project=project).snapshot()
    )
    assert (queue, hold, completed) == ("Queue (1)", "On hold (2)", "Completed (0)")
    assert "#title-hold" in json.dumps(jump, default=str)


def test_the_guide_ships_with_the_package(capsys):
    assert cli.main(["guide"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("# Working the simulation sequencer as an agent")
    assert "wait <id>" in out and "--why" in out


def test_wait_names_files_inside_the_run_folder(tmp_path):
    inside = tmp_path / "run" / "plots" / "a.png"
    assert cli._under(tmp_path / "run", str(inside)) == "plots/a.png"
    assert cli._under(tmp_path / "run", str(tmp_path / "x.json")) == str(
        tmp_path / "x.json"
    )
