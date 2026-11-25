"""Several projects, one queue: workers, merged order, one runner."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from qdesignoptimizer.sequencer.app.queue_state import Bridge, GuiError
from qdesignoptimizer.sequencer.engine import state as st
from qdesignoptimizer.sequencer.engine.cli import main as cli
from qdesignoptimizer.sequencer.engine.queue import QueueError
from qdesignoptimizer.sequencer.engine.runner import Runner, RunRefused
from qdesignoptimizer.sequencer.engine.workspace import (
    find_adapters,
    open_workspace,
    user_home,
)

from .helpers import make_project

# A design folder the way the real ones are: not a package, its adapter
# puts it on the import path, and its modules import each other by bare
# name. Both projects have a ``names`` module -- the collision workers
# exist for.
_ADAPTER = """
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from qdesignoptimizer.sequencer.backends.fake import FakeBackend


class Backend(FakeBackend):
    def generate(self, task):
        import names

        files, by = super().generate(task)
        files["task.yaml"] = f"value: {names.VALUE}\\n"
        return files, by


def make_backend():
    return Backend(HERE)
"""


def _design(root: Path, name: str, value: float) -> Path:
    folder = root / name
    (folder / "simulations").mkdir(parents=True)
    (folder / "names.py").write_text(f"VALUE = {value}\n")
    (folder / "simulations" / "sequencer.py").write_text(_ADAPTER)
    return folder


@pytest.fixture
def two(tmp_path):
    designs = tmp_path / "designs"
    alpha = _design(designs, "alpha", 1.5)
    beta = _design(designs, "beta", 2.5)
    bridge = Bridge([str(alpha), str(beta)])
    yield bridge, alpha, beta
    bridge.close()


def _value(folder: Path) -> float:
    data = yaml.safe_load((folder / "task.yaml").read_text())
    return data["value"]


def test_folders_are_searched_for_their_adapters(tmp_path):
    designs = tmp_path / "designs"
    alpha = _design(designs, "alpha", 1)
    _design(designs, "beta", 2)
    stray = designs / "beta" / "out" / "old"
    stray.mkdir(parents=True)
    (stray / "sequencer.py").write_text("def make_backend(): ...\n")
    assert find_adapters(alpha) == [
        str((alpha / "simulations" / "sequencer.py").resolve())
    ]
    found = find_adapters(designs)
    assert [Path(f).parents[1].name for f in found] == ["alpha", "beta"]
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(QueueError, match="no sequencer.py"):
        find_adapters(empty)


def test_each_project_runs_its_own_modules(two):
    bridge, alpha, beta = two
    assert bridge.workspace.projects.keys() == {"alpha", "beta"}
    a = bridge.add("task", "alpha/A.calibrate").split()[-1]
    b = bridge.add("task", "beta/A.calibrate").split()[-1]
    assert a.startswith("alpha/") and b.startswith("beta/")
    # In one process the second project would get the first one's
    # ``names`` (1.5 twice), silently.
    assert _value(alpha / "out" / "queue" / a.split("/")[1]) == 1.5
    assert _value(beta / "out" / "queue" / b.split("/")[1]) == 2.5


def test_one_merged_order_written_to_every_project(two):
    bridge, alpha, beta = two
    a = bridge.add("task", "alpha/A.calibrate").split()[-1]
    b = bridge.add("task", "beta/B.measure").split()[-1]
    a2 = bridge.add("task", "alpha/B.measure").split()[-1]
    snap = bridge.snapshot()
    assert snap["multi"] and [r["id"] for r in snap["rows"]] == [a, b, a2]
    assert [r["project"] for r in snap["rows"]] == ["alpha", "beta", "alpha"]
    assert any(c["key"] == "beta/A.calibrate" for c in snap["catalog"])
    bridge.reorder([a2, b, a])
    assert bridge.workspace.ids() == [a2, b, a]
    own = st.read_json(alpha / "out" / "queue" / "order.json")
    assert own == [a2.split("/")[1], a.split("/")[1]]
    bridge.move(b, -1)
    assert bridge.workspace.ids() == [b, a2, a]


def test_one_runner_runs_both_projects_in_the_shared_order(two):
    bridge, alpha, beta = two
    a = bridge.add("task", "alpha/A.calibrate").split()[-1]
    b = bridge.add("task", "beta/A.calibrate").split()[-1]
    bridge.reorder([b, a])
    final = Runner(bridge.workspace).run()
    assert final.status == st.FINISHED
    assert [d.split("/")[0] for d in final.done] == ["beta", "alpha"]
    assert len(list((alpha / "out" / "completed").iterdir())) == 1
    assert len(list((beta / "out" / "completed").iterdir())) == 1
    # Each study ran with its own project's value.
    (done_b,) = (beta / "out" / "completed").iterdir()
    (done_a,) = (alpha / "out" / "completed").iterdir()
    assert st.result_of(done_b)["outputs"]["value"] == 2.5
    assert st.result_of(done_a)["outputs"]["value"] == 1.5
    snap = bridge.snapshot()
    assert [c["project"] for c in snap["completed"]] == ["alpha", "beta"]
    assert snap["completed"][0]["name"].startswith("alpha/x-")
    state_dir = bridge.workspace.state_dir
    assert state_dir.parent == user_home() / "workspaces"
    assert (state_dir / st.LOG_FILE).exists()
    # Requeue and details work by key too.
    key = snap["completed"][1]["name"]
    assert bridge.details("completed", key)["result"]["status"] == "done"
    assert bridge.requeue(key).split()[-1].startswith("beta/")


def test_two_runners_never_solve_at_once(tmp_path):
    project = make_project(tmp_path / "p")
    project.add("A.calibrate")
    lock = st.acquire_lock(user_home())
    try:
        with pytest.raises(RunRefused, match="on this machine"):
            Runner(project).run()
        bridge = Bridge(project=project)
        bridge.workspace.refs = ["whatever.py"]
        with pytest.raises(GuiError, match="on this machine"):
            bridge.start()
        snap = bridge.snapshot()
        assert "another runner" in snap["runner"]["elsewhere"]
    finally:
        st.release_lock(lock)
    assert Runner(project).run().status == st.FINISHED


def test_cli_takes_several_projects(tmp_path, capsys):
    designs = tmp_path / "designs"
    alpha = _design(designs, "alpha", 1)
    beta = _design(designs, "beta", 2)
    assert cli([str(alpha), "-p", str(beta), "add", "beta/A.calibrate"]) == 0
    assert cli([str(alpha), "-p", str(beta), "add", "A.calibrate"]) == 1
    assert "name the project" in capsys.readouterr().err
    assert cli([str(designs), "list"]) == 0
    assert "beta/x-" in capsys.readouterr().out
