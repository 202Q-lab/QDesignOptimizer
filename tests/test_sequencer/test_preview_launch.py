"""Launching in the background; the picture of a study's components."""

from __future__ import annotations

import base64
import time
from types import SimpleNamespace

import pytest

from qdesignoptimizer.sequencer.app.queue_state import Bridge, GuiError
from qdesignoptimizer.sequencer.backends import qdo_preview

from .helpers import make_project, set_task


@pytest.fixture
def bridge(tmp_path):
    return Bridge(project=make_project(tmp_path))


def _wait(bridge, key, seconds=30):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        status = bridge.launch_status(key)
        if status and status["state"] != "running":
            return status
        time.sleep(0.05)
    raise AssertionError("the launch did not finish")


# -- launching in the background -------------------------------------------


def test_a_launch_runs_in_the_background_and_reports_progress(bridge, monkeypatch):
    key = bridge.new_draft(label="bg", from_campaign="full_A")
    project = bridge.project
    real = project.problems
    seen = []

    def slow(job_id, deep=False, folder=None):
        seen.append(bridge.launch_status(key)["checked"])
        time.sleep(0.2)  # as a design build would take
        return real(job_id, deep=deep, folder=folder)

    monkeypatch.setattr(project, "problems", slow)
    started = time.monotonic()
    assert "launching" in bridge.start_launch(key)
    assert time.monotonic() - started < 0.2  # it did not wait
    assert bridge.launch_status(key)["state"] == "running"
    with pytest.raises(GuiError, match="already being launched"):
        bridge.start_launch(key)
    status = _wait(bridge, key)
    assert status["state"] == "done" and "launched 2" in status["message"]
    assert status["checked"] == status["total"] == 2
    assert seen == [0, 1]
    assert project.queue.ids() == [
        "c01-bg-a_calibrate",
        "c01-bg-a_measure",
    ]
    bridge.forget_launch(key)
    assert bridge.launch_status(key) is None


def test_a_launch_that_fails_in_the_background_says_why(bridge):
    key = bridge.new_draft(label="bad")
    study = bridge.draft_add(key, "A.calibrate")
    bridge.save_file(study, "task.yaml", "nonsense: 1\n")
    bridge.start_launch(key)
    status = _wait(bridge, key)
    assert status["state"] == "failed" and "fix first" in status["message"]
    assert bridge.project.queue.ids() == []
    assert bridge.draft_view(key)["studies"]  # still in the draft


# -- airbridges where the renderer puts them -------------------------------


def test_bonds_are_placed_as_the_ansys_renderer_places_them():
    from shapely.geometry import LineString

    line = LineString([(0.0, 0.0), (1.2, 0.0), (1.2, 0.3)])
    bonds = qdo_preview.bond_positions(
        [(line, 0.06)], size=1.8, threshold="500um", offset="0um"
    )
    # 1.2 mm // 0.5 mm = 2 bonds, at 1/3 and 2/3; the 0.3 mm leg: none.
    assert [round(b[0], 6) for b in bonds] == [0.4, 0.8]
    assert all(b[1] == 0 and (b[2], b[3]) == (-0.0, 1.0) for b in bonds)
    assert all(abs(b[4] - 0.108) < 1e-12 for b in bonds)
    shifted = qdo_preview.bond_positions(
        [(line, 0.06)], size=1.8, threshold=0.5, offset="10um"
    )
    assert round(shifted[0][0], 6) == 0.41


def test_bond_settings_follow_the_renderer_the_study_uses():
    design = SimpleNamespace(
        renderers={
            "q3d": SimpleNamespace(options={"wb_size": 1.8, "wb_threshold": "500um"})
        }
    )
    eigen = SimpleNamespace(
        modes=["resonator_a"],
        run_capacitance_studies_only=False,
        hfss_wire_bond_size=2,
        hfss_wire_bond_threshold="250um",
        hfss_wire_bond_offset="0um",
    )
    capacitance = SimpleNamespace(run_capacitance_studies_only=True)
    assert qdo_preview.bond_settings(design, eigen)[:2] == (2, "250um")
    assert qdo_preview.bond_settings(design, capacitance)[:2] == (
        1.8,
        "500um",
    )


def test_a_picture_shows_only_the_study_s_components_with_bridges():
    """A real qiskit-metal design: two CPWs, one in the study."""
    from qiskit_metal import designs
    from qiskit_metal.qlibrary.terminations.open_to_ground import (
        OpenToGround,
    )
    from qiskit_metal.qlibrary.tlines.straight_path import RouteStraight

    design = designs.DesignPlanar()
    for name, x in (("a", "0mm"), ("b", "2mm"), ("c", "0mm"), ("d", "2mm")):
        y = "0mm" if name in "ab" else "1mm"
        OpenToGround(
            design,
            f"open_{name}",
            options=dict(pos_x=x, pos_y=y, orientation="180" if name in "bd" else "0"),
        )
    for name, (start, end) in (("line_in", "ab"), ("line_out", "cd")):
        RouteStraight(
            design,
            name,
            options=dict(
                hfss_wire_bonds=True,
                pin_inputs=dict(
                    start_pin=dict(component=f"open_{start}", pin="open"),
                    end_pin=dict(component=f"open_{end}", pin="open"),
                ),
            ),
        )
    study = SimpleNamespace(
        qiskit_component_names=["line_in", "nosuch"],
        capacitance_matrix_studies=None,
        modes=["resonator_line"],
        run_capacitance_studies_only=False,
        hfss_wire_bond_size=1.8,
        hfss_wire_bond_threshold="500um",
        hfss_wire_bond_offset="0um",
    )
    picture = qdo_preview.render(design, study, title="line_in")
    assert picture["components"] == ["line_in"]
    assert picture["missing"] == ["nosuch"]
    assert picture["bridges"] >= 2  # a 2 mm line at 500 um
    assert base64.b64decode(picture["png"])[:4] == b"\x89PNG"
    assert "HFSS renderer" in picture["caption"]


def test_a_study_is_one_picture_per_solver_run():
    """HFSS draws the MiniStudy's components; each chained Q3D study
    draws its own, which may be more."""
    chained = [
        SimpleNamespace(qiskit_component_names=["a", "b", "c"]),
        SimpleNamespace(qiskit_component_names=["a", "d"]),
    ]
    eigen = SimpleNamespace(
        qiskit_component_names=["a", "b"],
        modes=["resonator_a"],
        capacitance_matrix_studies=chained,
        run_capacitance_studies_only=False,
    )
    runs = qdo_preview.runs(eigen)
    assert [r["solver"] for r in runs] == ["hfss", "q3d", "q3d"]
    assert [r["components"] for r in runs] == [
        ["a", "b"],
        ["a", "b", "c"],
        ["a", "d"],
    ]
    assert runs[1]["title"].startswith("Q3D capacitance 1: ")
    only = SimpleNamespace(
        qiskit_component_names=["a"],
        capacitance_matrix_studies=chained[:1],
        run_capacitance_studies_only=True,
    )
    assert [r["solver"] for r in qdo_preview.runs(only)] == ["q3d"]
    assert qdo_preview.runs(only)[0]["title"].startswith("Q3D capacitance: ")
    # No modes: QDO skips the eigenmode solve even when not told to.
    no_modes = SimpleNamespace(
        qiskit_component_names=["a"],
        modes=[],
        capacitance_matrix_studies=chained[:1],
        run_capacitance_studies_only=False,
    )
    assert [r["solver"] for r in qdo_preview.runs(no_modes)] == ["q3d"]
    design = SimpleNamespace(
        renderers={"q3d": SimpleNamespace(options={"wb_size": 1.8})}
    )
    # A chained study of an eigenmode MiniStudy is still Q3D's.
    assert qdo_preview.bond_settings(design, eigen, "q3d")[0] == 1.8


def test_components_own_bridges_are_drawn_on_the_hfss_run_only():
    bridge = [[0.0, 0.0, 0.0], [0.0, 0.1, 0.0]]
    design = SimpleNamespace(
        components={
            "coupler": SimpleNamespace(get_air_bridge_coordinates=lambda: [bridge]),
            "plain": SimpleNamespace(),
        }
    )
    study = SimpleNamespace(surface_properties=None)
    assert qdo_preview.custom_bridges(design, ["coupler", "plain", "gone"], study) == [
        ((0.0, 0.0), (0.0, 0.1))
    ]
    # QDO leaves them out of a surface-participation study.
    spr = SimpleNamespace(surface_properties={"x": 1})
    assert qdo_preview.custom_bridges(design, ["coupler"], spr) == []


def test_a_backend_without_a_design_says_so(tmp_path):
    bridge = Bridge(project=make_project(tmp_path))
    job_id = bridge.project.add("A.calibrate")
    with pytest.raises(GuiError, match="cannot draw"):
        bridge.preview("queue", job_id)
