"""Reports of completed studies: which runs, brief and full, the table,
what fabrication gave, the command line."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
import yaml

from qdesignoptimizer.sequencer.backends import qdo_report
from qdesignoptimizer.sequencer.engine import cli
from qdesignoptimizer.sequencer.engine import report as rp
from qdesignoptimizer.sequencer.engine.runner import Runner
from qdesignoptimizer.sequencer.engine.workspace import Workspace

from .helpers import make_project

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


def _ran(project, n=1, requeue=False):
    task = project.backend.catalog().tasks[0].key
    for _ in range(n):
        project.add(task)
    Runner(project).run()
    if requeue:
        project.queue.requeue(project.queue.completed()[0], project.words)
        Runner(project).run()
    return Workspace([project])


def test_the_latest_run_of_each_study_is_chosen(project):
    workspace = _ran(project, n=2, requeue=True)
    runs = rp.completed_runs(workspace)
    assert len(runs) == 3
    chosen = rp.default_selection(runs)
    # the requeued copy stands for its study
    assert len(chosen) == 2
    assert any("_r" in key for key in chosen)


def test_a_study_id_picks_its_latest_run(project):
    workspace = _ran(project, requeue=True)
    first = rp.completed_runs(workspace)[-1]
    (key,) = rp.resolve(workspace, [first["study"]])
    assert "_r" in key
    assert rp.resolve(workspace, [first["key"]]) == [first["key"]]
    with pytest.raises(ValueError, match="no completed run"):
        rp.resolve(workspace, ["nothing-like-it"])


def test_brief_and_full_reports_and_their_files(project):
    workspace = _ran(project)
    runs = rp.collect(workspace, rp.default_selection(rp.completed_runs(workspace)))
    brief, full = rp.brief(runs), rp.full(runs)
    run_id = runs[0]["id"]
    assert run_id in brief and run_id in full
    assert brief.count("\n") < full.count("\n")
    assert "**Status:** done" in full
    folder = rp.write(project, runs, None)
    assert sorted(p.name for p in folder.iterdir()) == [
        "report.md",
        "report_brief.md",
        "study_flow.png",
    ]


def test_fabrication_is_kept_and_read_back(project):
    assert rp.load_fabrication(project) == {}
    values = rp.parse_fabrication(
        "qubit_1: 8.1 kOhm  # probe station\n\nqubit_2: 12 GHz"
    )
    rp.save_fabrication(project, values)
    assert rp.load_fabrication(project) == {"qubit_1": "8.1 kOhm", "qubit_2": "12 GHz"}
    with pytest.raises(ValueError, match="line 1"):
        rp.parse_fabrication("qubit_1 8.1 kOhm")


def test_tables_paste_as_markdown_and_as_cells():
    columns, rows = ["a", "b|c"], [[1, None], ["x\ny", "z"]]
    assert rp.markdown_table(columns, rows).splitlines() == [
        "| a | b\\|c |",
        "| --- | --- |",
        "| 1 |  |",
        "| x y | z |",
    ]
    assert rp.tsv_table(columns, rows) == "a\tb|c\n1\t\nx y\tz"


def test_the_command_line_writes_a_report_and_keeps_fabrication(project, capsys):
    _ran(project)
    ref = str(project.root / "sequencer.py")
    assert cli.main([ref, "report", "--fab", "qubit_1=9kOhm", "--show", "full"]) == 0
    out, err = capsys.readouterr()
    assert "# Simulation report" in out and "written to" in err
    assert rp.load_fabrication(project) == {"qubit_1": "9kOhm"}
    assert cli.main([ref, "report", "--fab", "qubit_1=", "--show", "none"]) == 0
    assert rp.load_fabrication(project) == {}
    assert cli.main([ref, "report", "nothing-like-it"]) == 1


# -- the QDO table ------------------------------------------------------------


def _qdo_run(tmp_path):
    folder = tmp_path / "x-1-q1_qubit_resonator_2026-11-14T17-12-32"
    folder.mkdir()
    (folder / "design_variables.json").write_text(
        json.dumps({"design_var_lj_qubit_1": "12.1nH"}), "utf-8"
    )
    (folder / "parameter_targets.yaml").write_text(
        yaml.safe_dump({"qubit_1_freq": 4e9}), "utf-8"
    )
    return {
        "id": "x-1-q1_qubit_resonator",
        "folder": str(folder),
        "outputs": {
            "qubit_1_freq": 4.055e9,
            "resonator_1_freq": 5.85e9,
            "resonator_1_kappa": 0.967e6,
            "qubit_1_to_qubit_1_nonlin": 214e6,
            "qubit_1_to_resonator_1_nonlin": 1.134e6,
            "resonator_1_to_resonator_1_nonlin": 1745.0,
        },
        "targets": [
            {
                "parameter": "qubit_1_freq",
                "design_var": "design_var_lj_qubit_1",
                "unit": "nH",
                "design_values": [12.1, 13.8, 14.26443177175191],
            }
        ],
    }


def test_a_qubit_row_with_its_fabricated_junction(tmp_path):
    made = qdo_report.table([_qdo_run(tmp_path)], {"qubit_1": "9 kOhm"})
    (row,) = made["rows"]
    cell = dict(zip(made["columns"], row))
    assert cell["Qubit"] == "qubit_1" and cell["Readout"] == "resonator_1"
    assert cell["f01 sim (GHz)"] == "4.0550"
    assert cell["f01 target (GHz)"] == "4.0000"
    assert cell["α (MHz)"] == "214.0"
    assert cell["χ (MHz)"] == "1.134"
    # the Lj of the last iteration, not the one it started with
    assert cell["Lj design (nH)"] == "14.26"
    assert cell["Ej design (GHz)"] == "11.459"
    # 9 kOhm: Ej = 15.60 GHz; f01 = (4.055 + 0.214) sqrt(15.60 / 11.46) - 0.214
    assert cell["Ej fab (GHz)"] == "15.604"
    assert cell["f01 from fab (GHz)"] == "4.7675"
    assert cell["Δf01 (MHz)"] == "713"


def test_without_fabrication_the_columns_are_left_to_fill_in(tmp_path):
    made = qdo_report.table([_qdo_run(tmp_path)], {})
    cell = dict(zip(made["columns"], made["rows"][0]))
    assert (
        cell["Fabricated"] == cell["Ej fab (GHz)"] == cell["f01 from fab (GHz)"] == ""
    )


def test_a_fabrication_value_that_does_not_read_is_noted(tmp_path):
    made = qdo_report.table([_qdo_run(tmp_path)], {"qubit_1": "nine"})
    assert any("qubit_1" in note for note in made["notes"])
    assert dict(zip(made["columns"], made["rows"][0]))["Ej fab (GHz)"] == ""


def test_a_mode_without_a_junction_is_not_a_qubit_row(tmp_path):
    run = _qdo_run(tmp_path)
    run["outputs"] = {"resonator_1_freq": 6e9, "resonator_1_kappa": 1e6}
    assert qdo_report.table([run], {})["rows"] == []


def _partition_run(tmp_path, aimed: list[str]):
    """Qubit 1 solved beside a coupler (also a junction) and a readout
    mode named without "resonator"; ``aimed``: the targets' parameters."""
    run = _qdo_run(tmp_path)
    (Path(run["folder"]) / "design_variables.json").write_text(
        json.dumps({"design_var_lj_qubit_1": "12.1nH", "design_var_lj_coupler": "4nH"}),
        "utf-8",
    )
    run["outputs"] = {
        "qubit_1_freq": 4.055e9,
        "coupler_freq": 5.3e9,
        "cavity_freq": 7.0e9,
        "qubit_1_to_qubit_1_nonlin": 214e6,
        "coupler_to_coupler_nonlin": 88e6,
        "cavity_to_cavity_nonlin": 12.0,
        "coupler_to_qubit_1_nonlin": 4.1e6,
        "cavity_to_qubit_1_nonlin": 0.15e6,
    }
    run["targets"] = [{"parameter": p} for p in aimed]
    return run


def _cells(made):
    return [dict(zip(made["columns"], row)) for row in made["rows"]]


def test_a_run_reports_only_the_qubits_it_tunes(tmp_path):
    run = _partition_run(
        tmp_path, ["qubit_1_freq", "qubit_1_to_qubit_1_nonlin", "cavity_freq"]
    )
    (cell,) = _cells(qdo_report.table([run], {}))
    # the coupler is in the model only to load qubit 1
    assert cell["Qubit"] == "qubit_1"


def test_a_cross_kerr_target_does_not_make_the_partner_tuned(tmp_path):
    run = _partition_run(tmp_path, ["coupler_freq", "coupler_to_qubit_1_nonlin"])
    assert [c["Qubit"] for c in _cells(qdo_report.table([run], {}))] == ["coupler"]


def test_a_run_without_mode_targets_reports_every_qubit(tmp_path):
    run = _partition_run(tmp_path, [])
    cells = _cells(qdo_report.table([run], {}))
    assert [c["Qubit"] for c in cells] == ["coupler", "qubit_1"]


def test_the_readout_is_the_linear_mode_whatever_its_name(tmp_path):
    run = _partition_run(tmp_path, ["qubit_1_freq"])
    (cell,) = _cells(qdo_report.table([run], {}))
    # the coupler couples more strongly, but it has a junction
    assert cell["Readout"] == "cavity" and cell["χ (MHz)"] == "0.150"


def test_without_a_linear_mode_there_is_no_readout(tmp_path):
    run = _partition_run(tmp_path, ["coupler_freq"])
    del run["outputs"]["cavity_freq"]
    (cell,) = _cells(qdo_report.table([run], {}))
    assert cell["Readout"] == cell["f_r sim (GHz)"] == cell["χ (MHz)"] == ""


def test_a_sequence_step_names_its_study_and_step(tmp_path):
    run = _qdo_run(tmp_path)
    run["task"] = "Q1.readout"
    run["sequence"] = {"id": "s001", "name": "tune", "entry": 4}
    (cell,) = _cells(qdo_report.table([run], {}))
    assert cell["Study"] == "Q1.readout · s001 step 4"
    run["sequence"] = {}
    (cell,) = _cells(qdo_report.table([run], {}))
    assert cell["Study"] == "Q1.readout"


def test_a_rerun_of_a_step_says_so(tmp_path):
    run = _qdo_run(tmp_path)
    run.update(
        task="Q1.readout",
        id="s001-tune-q1_readout_r2",
        sequence={"id": "s001", "entry": 4},
        requeued_from="s001-tune-q1_readout_2026-11-23T13-56-09_failed",
    )
    (cell,) = _cells(qdo_report.table([run], {}))
    assert cell["Study"] == "Q1.readout · s001 step 4, rerun 2"
    # a study's own name may end in _r2: only a requeued copy is a rerun
    run["requeued_from"] = ""
    (cell,) = _cells(qdo_report.table([run], {}))
    assert cell["Study"] == "Q1.readout · s001 step 4"


def test_values_read_with_their_units():
    assert qdo_report.value("qubit_1_freq", 4.055e9) == "4.055 GHz"
    assert qdo_report.value("resonator_1_kappa", 0.967e6) == "0.967 MHz"
    assert qdo_report.value("something", 3.0) == "3"
