"""The simulation sequencer end to end, against the fake Ansys.

The tutorial's sequencer project (``tutorials/example_simulation_sequencer/
coupled_transmon_chip``) and the chip it builds on are copied into a
temporary folder; QDO's ``DesignAnalysis`` then runs for real with HFSS,
pyEPR and Q3D replaced by the recording fakes of ``fake_ansys``. Studies
and a study sequence are queued and run by the real runner, and the test
follows them through ``out/queue`` and ``out/completed`` and into the
store. Nothing is written outside ``tmp_path``.

The fake answers with the relations the targets assume (``fake_ansys``),
so this checks what the sequencer does with a study, not whether a
target's physics is right.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from qdesignoptimizer.sequencer.engine import state as st
from qdesignoptimizer.sequencer.engine.project import open_project
from qdesignoptimizer.sequencer.engine.queue import INPUTS, RESOLVED
from qdesignoptimizer.sequencer.engine.runner import Runner

from .example_harness import TUTORIAL_LOCAL_MODULES, TUTORIALS, _install_fakes
from .fake_ansys import SimulationSession

SEQUENCER_PROJECT = Path("example_simulation_sequencer") / "coupled_transmon_chip"
CHIP = Path("examples_coupled_transmon_chip")
#: What a run must never copy: results and the store of earlier runs.
_NOT_COPIED = shutil.ignore_patterns(
    "out", "__pycache__", "*.ipynb", "design_variables.json", "*.meta.json"
)
KAPPA_VARIABLE = "design_var_coupl_length_resonator_1_tee"


@pytest.fixture
def chip(tmp_path, monkeypatch):
    """The sequencer project in ``tmp_path``, run against the fakes.

    Returns ``(project, session, root)``: the opened project, the fake
    simulator's session (its ``events`` record every call) and the
    project's folder.
    """
    tutorials = tmp_path / "tutorials"
    shutil.copytree(TUTORIALS / CHIP, tutorials / CHIP, ignore=_NOT_COPIED)
    shutil.copy(TUTORIALS / CHIP / "design_variables.json", tutorials / CHIP)
    shutil.copytree(
        TUTORIALS / SEQUENCER_PROJECT,
        tutorials / SEQUENCER_PROJECT,
        ignore=_NOT_COPIED,
    )
    root = tutorials / SEQUENCER_PROJECT

    monkeypatch.setenv("QDO_SEQUENCER_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    # The adapter puts its folders on sys.path and imports the chip's
    # modules by bare name: undo both afterwards, so no other test sees
    # this copy's "design" or "names".
    monkeypatch.setattr(sys, "path", list(sys.path))
    for module in TUTORIAL_LOCAL_MODULES:
        monkeypatch.setitem(sys.modules, module, None)
        monkeypatch.delitem(sys.modules, module)

    session = SimulationSession()
    _install_fakes(session, monkeypatch)
    # A fresh chip model per study: each study starts off target and has
    # to converge, as against HFSS (the model is calibrated on the design
    # variables the study starts from).
    register = session.register_analysis

    def per_study(analysis):
        session.physics = None
        register(analysis)

    monkeypatch.setattr(session, "register_analysis", per_study)

    project = open_project(root / "simulations" / "sequencer.py")
    return project, session, root


def _store(root: Path) -> dict:
    return json.loads((root / "design_variables.json").read_text("utf-8"))


def _result(folder: Path) -> dict:
    return json.loads((folder / st.RESULT_FILE).read_text("utf-8"))


def _completed(project, job_id: str) -> Path:
    (folder,) = (p for p in project.queue.completed() if p.name.startswith(job_id))
    return folder


def test_a_study_and_a_sequence_move_through_the_queue_into_the_store(chip):
    project, session, root = chip
    start = _store(root)

    # -- queued: one study, then a study sequence of three ---------------
    single = project.add(
        "Q1.resonator_kappa",
        settings={"iterations": 3},
        checks=["converged"],
        save=True,
    )
    spec = project.backend.catalog().sequence("tune_Q1")
    sequence = project.add_sequence(spec)
    assert project.queue.ids() == [single, *sequence]
    assert [i.split("-", 2)[:2] for i in sequence] == [["s001", "tune_Q1"]] * 3
    for job_id in [single, *sequence]:
        folder = project.queue.folder(job_id)
        assert {"job.yaml", "ministudy.yaml", "opt_targets.yaml"} <= {
            p.name for p in folder.iterdir()
        }
        assert project.problems(job_id, deep=True) == []
    assert project.job(sequence[2]).settings == {"iterations": 4}
    assert project.job(sequence[0]).sequence == {
        "id": "s001",
        "name": "tune_Q1",
        "entry": 1,
    }

    # -- run --------------------------------------------------------------
    final = Runner(project).run()
    assert final.status == st.FINISHED, final.message
    assert final.failed == []
    assert project.queue.ids() == []
    assert not any(project.queue.queue_dir.glob(".claimed-*"))

    # -- every study is in completed/, with what it ran from --------------
    for job_id in [single, *sequence]:
        folder = _completed(project, job_id)
        result = _result(folder)
        assert result["status"] == st.DONE, result["message"]
        assert result["outputs"]["converged"] is True
        assert result["committed"], "a converged study with save writes back"
        assert (folder / INPUTS / "job.yaml").is_file()
        assert (folder / "job.yaml").is_file()  # kept after the run
        assert (folder / RESOLVED / "design_variables.json").is_file()
        assert (folder / "convergence.png").is_file()

    # -- chained: each study started from what the one before wrote back --
    single_wrote = _result(_completed(project, single))["committed"]
    assert single_wrote[KAPPA_VARIABLE] != start[KAPPA_VARIABLE]
    first_of_sequence = _completed(project, sequence[0])
    ran_with = json.loads(
        (first_of_sequence / RESOLVED / "design_variables.json").read_text("utf-8")
    )
    assert ran_with[KAPPA_VARIABLE] == single_wrote[KAPPA_VARIABLE]

    # -- the store holds the last write of each value, and who wrote it ---
    store = _store(root)
    meta = json.loads((root / "design_variables.meta.json").read_text("utf-8"))
    last_writer = {}
    for job_id in [single, *sequence]:
        for name, value in _result(_completed(project, job_id))["committed"].items():
            last_writer[name] = (job_id, value)
    for name, (job_id, value) in last_writer.items():
        assert store[name] == value
        assert meta[name]["latest"]["study"] == job_id
    assert meta[KAPPA_VARIABLE]["latest"]["sequence"] == "s001"

    # -- and the simulator was asked for both kinds of study --------------
    kinds = {event for event, _ in session.events}
    assert "render_qiskit_metal" in kinds
    assert len(kinds) > 1


def test_a_campaign_study_writes_the_campaigns_store_only(chip):
    project, _, root = chip
    start = _store(root)
    folder = project.new_campaign("tuneup_chip", "Can Q1's readout reach its kappa?")
    job_id = project.add(
        "Q1.resonator_kappa",
        checks=["converged"],
        save=True,
        campaign="tuneup_chip",
    )
    assert Runner(project).run().status == st.FINISHED
    campaign_store = json.loads((folder / "design_variables.json").read_text("utf-8"))
    assert campaign_store[KAPPA_VARIABLE] != start[KAPPA_VARIABLE]
    assert _store(root) == start  # the chip's own store is untouched
    assert _result(_completed(project, job_id))["campaign"] == "tuneup_chip"

    assert KAPPA_VARIABLE in project.promote("tuneup_chip")
    assert _store(root)[KAPPA_VARIABLE] == campaign_store[KAPPA_VARIABLE]


def test_a_study_off_target_writes_nothing_and_stops_the_rest(chip):
    project, _, root = chip
    start = _store(root)
    # One iteration starts off target and cannot converge in it.
    failing = project.add(
        "Q1.resonator_kappa",
        settings={"iterations": 1},
        checks=["converged"],
        save=True,
    )
    after = project.add("Q1.qubit_charge_line", checks=["converged"], save=True)
    final = Runner(project).run()
    assert final.status == st.ABORTED
    folder = _completed(project, failing)
    assert folder.name.endswith("_failed")
    result = _result(folder)
    assert result["status"] == st.FAILED
    assert result["checks"]["converged"]["ok"] is False
    assert result["committed"] == {}
    assert _store(root) == start
    assert project.queue.ids() == [after]  # on_failure: abort
