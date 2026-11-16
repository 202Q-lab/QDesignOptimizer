"""Moving a study in the run order (2026-11-16): ``Workspace.move``, the
command line's ``move`` (for an agent reordering the queue) and the
machine-wide lock every order write is made under."""

from __future__ import annotations

import threading

import pytest

from qdesignoptimizer.sequencer.app.queue_state import Bridge
from qdesignoptimizer.sequencer.engine import cli
from qdesignoptimizer.sequencer.engine import state as st
from qdesignoptimizer.sequencer.engine.queue import QueueError
from qdesignoptimizer.sequencer.engine.workspace import Workspace, user_home

from .helpers import make_project

ADAPTER = (
    "from qdesignoptimizer.sequencer.backends.fake import FakeBackend\n"
    "from pathlib import Path\n"
    "def make_backend():\n"
    "    return FakeBackend(Path(__file__).parent)\n"
)


@pytest.fixture
def project(tmp_path):
    return make_project(tmp_path / "p")


def _queue(project, n=4):
    task = project.backend.catalog().tasks[0].key
    return [project.add(task) for _ in range(n)]


def test_move_to_a_place_or_next_to_another(project):
    a, b, c, d = _queue(project)
    workspace = Workspace.of(project)
    assert workspace.move(d, to=1) == [d, a, b, c]
    assert workspace.move(d, to=99) == [a, b, c, d]  # past the end: last
    assert workspace.move(a, before=d) == [b, c, a, d]
    assert workspace.move(b, after=d) == [c, a, d, b]
    assert project.queue.ids() == [c, a, d, b]  # on disk


def test_move_refuses_what_is_not_queued(project):
    a, b = _queue(project, 2)
    workspace = Workspace.of(project)
    with pytest.raises(QueueError, match="is not queued"):
        workspace.move("nope", to=1)
    with pytest.raises(QueueError, match="is not queued"):
        workspace.move(a, before="nope")
    with pytest.raises(QueueError, match="next to itself"):
        workspace.move(a, after=a)
    with pytest.raises(QueueError, match="say where"):
        workspace.move(a)
    assert project.queue.ids() == [a, b]


def test_the_command_line_moves(project, capsys):
    a, b, c = _queue(project, 3)
    adapter = project.root / "sequencer.py"
    adapter.write_text(ADAPTER, "utf-8")
    ref = str(adapter)
    assert cli.main([ref, "move", c, "--first"]) == 0
    assert f"{c} is now 1 of 3" in capsys.readouterr().out
    assert cli.main([ref, "move", c, "--after", b]) == 0
    assert cli.main([ref, "move", a, "--last"]) == 0
    assert cli.main([ref, "move", b, "--to", "2"]) == 0
    assert project.queue.ids() == [c, b, a]
    assert cli.main([ref, "move", "nope", "--first"]) == 1
    assert "is not queued" in capsys.readouterr().err


def test_the_gui_arrows_move_through_the_same_path(project):
    a, b, c = _queue(project, 3)
    bridge = Bridge(project=project)
    bridge.move(c, -1)
    assert project.queue.ids() == [a, c, b]


def test_an_order_write_waits_for_another_process_s(project, monkeypatch):
    """The file lock is what keeps a command line move and a GUI move
    apart; here another holder is a second thread holding the file lock
    directly (another process would hold it the same way)."""
    a, b, c = _queue(project, 3)
    workspace = Workspace.of(project)
    held = threading.Event()
    release = threading.Event()

    def other():
        with st.file_lock(user_home() / "locks" / "order.lock"):
            held.set()
            release.wait(5)

    thread = threading.Thread(target=other)
    thread.start()
    held.wait(5)
    done = []
    mover = threading.Thread(target=lambda: done.append(workspace.move(c, to=1)))
    mover.start()
    mover.join(0.3)
    assert not done  # waiting for the lock
    release.set()
    thread.join(5)
    mover.join(5)
    assert done == [[c, a, b]]


def test_order_guard_is_reentrant(project):
    with st.order_guard():
        with st.order_guard():
            _queue(project, 1)  # add writes the order under the guard too
