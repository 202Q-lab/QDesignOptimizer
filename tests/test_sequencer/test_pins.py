"""Components and pins of a study: the design's components and how their
pins join, a study's runs read from its ministudy, a choice written back,
the pictures of a choice, and the GUI's bridge saving it."""

from __future__ import annotations

import base64

import pytest
import yaml

pytest.importorskip("qiskit_metal")

from qdesignoptimizer.sequencer.app.queue_state import Bridge  # noqa: E402
from qdesignoptimizer.sequencer.app.queue_state import GuiError
from qdesignoptimizer.sequencer.backends import qdo_pins  # noqa: E402
from qdesignoptimizer.sequencer.backends import qdo_preview
from qdesignoptimizer.sequencer.backends.qdo import QDOBackend  # noqa: E402
from qdesignoptimizer.sequencer.engine.project import Project  # noqa: E402

from .test_qdo_backend import TinyProject  # noqa: E402


def _chip():
    """Two CPWs, each between two open ends: line_in (open_a to open_b)
    and line_out (open_c to open_d)."""
    from qiskit_metal import designs
    from qiskit_metal.qlibrary.terminations.open_to_ground import OpenToGround
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
                pin_inputs=dict(
                    start_pin=dict(component=f"open_{start}", pin="open"),
                    end_pin=dict(component=f"open_{end}", pin="open"),
                ),
            ),
        )
    return design


MINISTUDY = {
    "class": "qdesignoptimizer.design_analysis_types:MiniStudy",
    "args": {
        "qiskit_component_names": ["line_in", "open_a"],
        "port_list": [["line_in", "end", 50]],
        "open_pins": [],
        "modes": ["resonator_x"],
        "capacitance_matrix_studies": [
            {
                "class": "qdesignoptimizer.sim_capacitance_matrix:CapacitanceMatrixStudy",
                "args": {
                    "qiskit_component_names": ["line_out"],
                    "open_pins": [["line_out", "start"]],
                },
            }
        ],
    },
}


def test_every_component_is_listed_with_what_its_pins_join():
    found = {c["name"]: c for c in qdo_pins.inventory(_chip())}
    assert sorted(found) == [
        "line_in",
        "line_out",
        "open_a",
        "open_b",
        "open_c",
        "open_d",
    ]
    assert found["open_a"]["class"] == "OpenToGround"
    joined = {p["name"]: p["joined"] for p in found["line_in"]["pins"]}
    assert joined == {"start": "open_a", "end": "open_b"}


def test_a_study_s_runs_are_read_from_its_ministudy():
    hfss, q3d = qdo_pins.runs(MINISTUDY)
    assert (hfss["title"], hfss["solver"], hfss["ports"]) == (
        "HFSS eigenmode",
        "hfss",
        True,
    )
    assert hfss["components"] == ["line_in", "open_a"]
    assert hfss["states"] == {"line_in/end": ["port", 50]}
    assert q3d["path"] == ["args", "capacitance_matrix_studies", 0, "args"]
    assert q3d["title"].startswith("Q3D capacitance") and not q3d["ports"]
    assert q3d["states"] == {"line_out/start": ["open", 50]}


def test_a_list_that_is_an_expression_is_not_edited_here():
    data = {"args": {"qiskit_component_names": "=v['names']", "modes": ["m"]}}
    (run,) = qdo_pins.runs(data)
    assert not run["editable"] and "form" in run["note"]


def test_a_choice_is_written_back_short_pins_listed_nowhere():
    states = {
        "line_in/end": ["port", "75"],
        "line_in/start": ["short", 50],
        "open_b/open": ["open", 50],
        "line_out/end": ["open", 50],  # not included: dropped
    }
    out = qdo_pins.apply(MINISTUDY, ["args"], ["line_in", "open_b"], states)
    args = out["args"]
    assert args["qiskit_component_names"] == ["line_in", "open_b"]
    assert args["port_list"] == [["line_in", "end", 75]]
    assert args["open_pins"] == [["open_b", "open"]]
    # the chained study is left as it was, and the input is not changed
    assert (
        args["capacitance_matrix_studies"]
        == MINISTUDY["args"]["capacitance_matrix_studies"]
    )
    assert MINISTUDY["args"]["qiskit_component_names"] == ["line_in", "open_a"]


def test_a_pin_joined_inside_the_model_is_no_end():
    inventory = qdo_pins.inventory(_chip())
    states = {"line_in/end": ["port", 50], "line_in/start": ["open", 50]}
    # open_b included: line_in's end joins it, its start (open_a) does not
    assert qdo_pins.loose_states(inventory, ["line_in", "open_b"], states) == {
        "line_in/start": ["open", 50]
    }


def test_a_capacitance_study_takes_no_ports_and_impedances_are_checked():
    path = ["args", "capacitance_matrix_studies", 0, "args"]
    with pytest.raises(ValueError, match="no ports"):
        qdo_pins.apply(MINISTUDY, path, ["line_out"], {"line_out/end": ["port", 50]})
    with pytest.raises(ValueError, match="positive"):
        qdo_pins.apply(MINISTUDY, ["args"], ["line_in"], {"line_in/end": ["port", 0]})
    with pytest.raises(ValueError, match="no longer"):
        qdo_pins.apply(
            MINISTUDY, ["args", "capacitance_matrix_studies", 3, "args"], [], {}
        )


def test_pins_are_marked_joined_inside_the_model_else_as_chosen():
    design = _chip()
    marks = qdo_pins.markers(
        design, ["line_in", "open_a"], {"line_in/end": ["open", 50]}
    )
    state = {(m["component"], m["pin"]): m["state"] for m in marks}
    assert state[("line_in", "start")] == "joined"  # open_a is included
    assert state[("open_a", "open")] == "joined"
    assert state[("line_in", "end")] == "open"  # open_b is not


def test_a_choice_is_drawn_with_the_rest_of_the_chip_and_its_pins():
    from types import SimpleNamespace

    design = _chip()
    study = SimpleNamespace(modes=["m"], run_capacitance_studies_only=False)
    pins = qdo_pins.markers(design, ["line_in"], {"line_in/end": ["port", 50]})
    for whole in (False, True):
        picture = qdo_preview.render(
            design,
            study,
            names=["line_in"],
            solver="hfss",
            bonds=(3, "300um", "0um", "test"),
            context=True,
            whole=whole,
            labels=whole,
            pins=pins,
        )
        assert base64.b64decode(picture["png"])[:4] == b"\x89PNG"
        assert picture["components"] == ["line_in"]


# -- the backend and the GUI's bridge, on a tiny project --------------------


@pytest.fixture
def tiny(tmp_path):
    qdo_project = TinyProject(tmp_path)
    qdo_project.components = {"res_x": {"start", "end"}, "feed": {"in", "out"}}
    project = Project(QDOBackend(qdo_project))
    return project, Bridge(project=project)


def test_the_bridge_lists_the_design_and_saves_a_choice(tiny):
    project, bridge = tiny
    job_id = project.add("X.freq")
    data = bridge.pin_view(job_id)
    assert [c["name"] for c in data["components"]] == ["feed", "res_x"]
    (run,) = data["runs"]
    assert run["components"] == ["res_x"]
    text = bridge.save_pins(
        job_id,
        0,
        ["res_x", "feed"],
        {"feed/in": ["port", 50], "res_x/end": ["open", 50]},
    )
    assert text.startswith("saved ministudy.yaml")
    saved = yaml.safe_load(
        (project.queue.folder(job_id) / "ministudy.yaml").read_text("utf-8")
    )
    assert saved["args"]["qiskit_component_names"] == ["res_x", "feed"]
    assert saved["args"]["port_list"] == [["feed", "in", 50]]
    assert saved["args"]["open_pins"] == [["res_x", "end"]]
    # what QDO sets after building is left alone
    assert saved["extra"] == {"grounded_pins": [["res_x", "end"]]}


def test_the_bridge_refuses_an_empty_choice_or_a_gone_run(tiny):
    project, bridge = tiny
    job_id = project.add("X.freq")
    with pytest.raises(GuiError, match="at least one"):
        bridge.save_pins(job_id, 0, [], {})
    with pytest.raises(GuiError, match="no longer"):
        bridge.save_pins(job_id, 5, ["res_x"], {})
