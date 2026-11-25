"""A report about the design first: its name, picture and purpose, its
design variables with the study that set each as a side note, one row
per qubit, and the studies after."""

from __future__ import annotations

import base64
import io
import json

import pytest

from qdesignoptimizer.sequencer.app.layout import _split_at_studies
from qdesignoptimizer.sequencer.backends import qdo_report
from qdesignoptimizer.sequencer.backends._store import readme_summary
from qdesignoptimizer.sequencer.engine import report as rp
from qdesignoptimizer.sequencer.engine.runner import Runner
from qdesignoptimizer.sequencer.engine.workspace import Workspace

from .helpers import make_project
from .test_report import ADAPTER, _qdo_run

README = """# Two transmons and a coupler

Two qubits coupled through a tunable coupler,
tuned in partitions.

| A table | that is not the purpose |
| --- | --- |

More about it.
"""

# A 1x1 PNG.
PNG = base64.b64encode(
    bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
        "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082"
    )
).decode("ascii")


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    made = make_project(tmp_path / "p")
    (made.root / "sequencer.py").write_text(ADAPTER, "utf-8")
    return made


def _wrote(project, values: dict, meta: dict) -> None:
    project.backend.store_path.write_text(json.dumps(values), "utf-8")
    (project.root / "store.meta.json").write_text(json.dumps(meta), "utf-8")


def _latest(value, task="Q1.readout", **more):
    return {
        "latest": {
            "value": value,
            "task": task,
            "sequence": "s001",
            "written": "2026-11-23 14:19:52",
            **more,
        }
    }


def test_the_name_and_purpose_are_the_readme_heading_and_first_paragraph(tmp_path):
    path = tmp_path / "README.md"
    assert readme_summary(path) == ("", "")
    path.write_text(README, "utf-8")
    assert readme_summary(path) == (
        "Two transmons and a coupler",
        "Two qubits coupled through a tunable coupler, tuned in partitions.",
    )


def test_the_design_is_the_store_with_who_wrote_each_value(project):
    _wrote(
        project,
        {"lj_qubit_1": "17.3 nH", "cj_qubit_1": "0fF", "width": "140um"},
        {
            "lj_qubit_1": _latest("17.3 nH"),
            "width": _latest("135um"),  # edited by hand since
        },
    )
    design = project.backend.report_design("")
    by_name = {v["name"]: v for v in design["variables"]}
    assert by_name["lj_qubit_1"]["written_by"]["task"] == "Q1.readout"
    assert not by_name["lj_qubit_1"]["edited"]
    assert by_name["cj_qubit_1"]["written_by"] == {}
    assert by_name["width"]["edited"]


def test_the_variables_table_puts_results_first_and_the_study_aside():
    design = {
        "project": "p",
        "_backend": None,
        "variables": [
            {"name": "design_var_cj_qubit_1", "value": "0fF", "written_by": {}},
            {
                "name": "design_var_lj_qubit_1",
                "value": "17.293636761726173 nH",
                "edited": False,
                "written_by": _latest(
                    "17.293636761726173 nH",
                    target=[
                        {
                            "parameter": "qubit_1_freq",
                            "target": 4.16e9,
                            "achieved": 4.1545e9,
                            "within_tolerance": True,
                        }
                    ],
                )["latest"],
            },
            {
                "name": "design_var_coupl_length_qubit_1_resonator_1",
                "value": "50.659um",
                "edited": True,
                "written_by": _latest(
                    "48um",
                    target=[
                        {
                            "parameter": "qubit_1_to_resonator_1_nonlin",
                            "target": 0.17e6,
                            "achieved": 0.144e6,
                            "within_tolerance": False,
                        }
                    ],
                )["latest"],
            },
        ],
    }
    columns, rows = rp.variables_table(design)
    cells = [dict(zip(columns, row)) for row in rows]
    # tuned first, by what they tune; a value no study wrote last
    assert [c["Design variable"] for c in cells] == [
        "lj_qubit_1",
        "coupl_length_qubit_1_resonator_1",
        "cj_qubit_1",
    ]
    lj, chi, cj = cells
    assert lj["Value"] == "17.294 nH"
    assert lj["Tuned for"] == "qubit_1_freq"
    assert lj["Reached when set"] == "4.1545e+09 (-0.13 %)"
    assert lj["Set by"] == "*Q1.readout · s001, 11-23 14:19*"
    assert chi["Reached when set"].endswith("✗")
    assert chi["Set by"].startswith("*edited since;")
    assert cj["Value"] == "0 fF" and cj["Tuned for"] == ""
    assert cj["Set by"] == "*fixed: no study tunes it*"


def test_the_report_opens_with_the_design_and_ends_with_the_studies(project):
    (project.root / "README.md").write_text(README, "utf-8")
    project.backend.report_design_picture = lambda campaign, title: {
        "png": PNG,
        "caption": f"{title}, drawn",
    }
    project.add(project.backend.catalog().tasks[0].key)
    Runner(project).run()
    workspace = Workspace([project])
    runs = rp.collect(workspace, rp.default_selection(rp.completed_runs(workspace)))

    folder = rp.write(project, runs, None)

    assert (folder / rp.DESIGN_PNG).read_bytes()[:4] == b"\x89PNG"
    for name in ("report.md", "report_brief.md"):
        text = (folder / name).read_text("utf-8")
        lines = text.splitlines()
        assert lines[0] == "# Two transmons and a coupler"
        order = [
            text.index("![Two transmons and a coupler](design.png)"),
            text.index("*Two transmons and a coupler, drawn*"),
            text.index("Two qubits coupled through"),
            text.index("## Design variables"),
            text.index("## Studies"),
            text.index(runs[0]["id"]),
        ]
        assert order == sorted(order)
    full = (folder / "report.md").read_text("utf-8")
    assert f"### {runs[0]['id']}" in full and "**Status:** done" in full


def test_a_qubit_has_one_row_from_the_latest_study_that_wrote_it_back(tmp_path):
    def run(name, started, committed):
        made = _qdo_run(tmp_path / name)
        made.update(
            id=name, task=name, started=started, committed=committed, sequence={}
        )
        return made

    for name in ("old", "new_failed", "newer_failed"):
        (tmp_path / name).mkdir()
    runs = [
        run("new_failed", "2026-11-23 15:00", {}),
        run("old", "2026-11-23 13:00", {"design_var_lj_qubit_1": "14nH"}),
    ]
    made = qdo_report.table(runs, {})
    (row,) = made["rows"]
    assert dict(zip(made["columns"], row))["From study"] == "old"
    # none wrote back: the latest
    runs = [run("newer_failed", "2026-11-23 16:00", {}), runs[0]]
    (row,) = qdo_report.table(runs, {})["rows"]
    assert row[-1] == "newer_failed"


def test_the_study_column_is_a_side_note_in_the_report_not_in_the_paste():
    made = {"columns": ["Qubit", "From study"], "aside": ["From study"]}
    made["rows"] = [["qubit_1", "Q1.readout"], ["coupler", ""]]
    assert "| qubit_1 | *Q1.readout* |" in rp._table_md(made)
    assert "| coupler |  |" in rp._table_md(made)
    assert rp.tsv_table(made["columns"], made["rows"]).splitlines()[1] == (
        "qubit_1\tQ1.readout"
    )


def test_the_gui_shows_the_study_flow_at_the_head_of_the_studies():
    text = "# Chip\n\n## Design variables\n\n## Studies\n\nHow it got here.\n\n### s1\n"
    head, studies = _split_at_studies(text)
    assert head.endswith("How it got here.\n") and studies == "\n### s1\n"
    assert _split_at_studies("# no studies\n") == ("# no studies\n", "")
