"""Components and pins of a study: the design's components and how their
pins join, the studies of a ministudy, a choice read from and written to
its fields, the pictures of a choice, and the GUI's form of it."""

from __future__ import annotations

import base64

import pytest
import yaml

pytest.importorskip("qiskit_metal")

from qdesignoptimizer.sequencer.app import form_layout, forms, layout  # noqa: E402
from qdesignoptimizer.sequencer.app.queue_state import Bridge  # noqa: E402
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
    assert not run["editable"] and "as text" in run["note"]


INVENTORY = [
    {
        "name": "line_in",
        "class": "RouteStraight",
        "pins": [
            {"name": "start", "joined": "open_a"},
            {"name": "end", "joined": "open_b"},
        ],
    },
    {
        "name": "open_a",
        "class": "OpenToGround",
        "pins": [{"name": "open", "joined": "line_in"}],
    },
    {
        "name": "open_b",
        "class": "OpenToGround",
        "pins": [{"name": "open", "joined": "line_in"}],
    },
]


def test_a_choice_is_read_from_a_study_s_fields_and_written_back():
    sel = layout.pins_from_fields(
        {
            "components": "[line_in, open_a]",
            "open": "[]",
            "ports": "[[line_in, end, 75]]",
        }
    )
    assert sel == {
        "components": ["line_in", "open_a"],
        "states": {"line_in/end": ["port", 75]},
    }
    sel["states"]["line_in/start"] = ["open", 50]  # joined to open_a: inside
    sel["states"]["open_b/open"] = ["open", 50]  # open_b not included
    assert layout.pins_to_fields(INVENTORY, sel) == {
        "components": "[line_in, open_a]",
        "open": "[]",
        "ports": "[[line_in, end, 75]]",
    }


def test_short_pins_are_listed_nowhere_and_a_bad_impedance_is_50_ohm():
    sel = {
        "components": ["line_in"],
        "states": {"line_in/end": ["port", "-3"], "line_in/start": ["open", 50]},
    }
    texts = layout.pins_to_fields(INVENTORY, sel)
    assert texts["ports"] == "[[line_in, end, 50]]"
    assert texts["open"] == "[[line_in, start]]"
    assert layout.pins_from_fields({"components": None})["components"] == []


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


def test_a_ministudy_is_edited_one_study_at_a_time(tiny):
    project, bridge = tiny
    job_id = project.add("X.freq")
    spec = bridge.file_spec(job_id, "ministudy.yaml")
    (study,) = spec["studies"]
    assert study["path"] == [] and study["solver"] == "hfss"
    assert study["fields"] == {
        "components": ["args", "qiskit_component_names"],
        "open": ["args", "open_pins"],
        "ports": ["args", "port_list"],
    }
    assert [c["name"] for c in bridge.pin_inventory(job_id)] == ["feed", "res_x"]
    # the form: a study picker, its box, the picked fields kept as text
    page = str(form_layout.file_form(spec, True))
    assert "study-pick" in page and "study-box" in page and "as text" in page
    # what the picker writes into the fields, the form's Save saves
    texts = layout.pins_to_fields(
        bridge.pin_inventory(job_id),
        {
            "components": ["res_x", "feed"],
            "states": {"feed/in": ["port", 50], "res_x/end": ["open", 50]},
        },
    )
    edits = {forms.key(study["fields"][role]): text for role, text in texts.items()}
    assert bridge.save_form(job_id, "ministudy.yaml", edits).startswith("saved")
    saved = yaml.safe_load(
        (project.queue.folder(job_id) / "ministudy.yaml").read_text("utf-8")
    )
    assert saved["args"]["qiskit_component_names"] == ["res_x", "feed"]
    assert saved["args"]["port_list"] == [["feed", "in", 50]]
    assert saved["args"]["open_pins"] == [["res_x", "end"]]


def test_a_capacitance_study_is_its_own_study_without_ports(tiny):
    project, _ = tiny
    studies = project.backend.form_studies("ministudy.yaml", MINISTUDY)
    assert [s["solver"] for s in studies] == ["hfss", "q3d"]
    assert studies[1]["path"] == ["args", "capacitance_matrix_studies", 0]
    assert set(studies[1]["fields"]) == {"components", "open"}
    assert project.backend.form_studies("opt_targets.yaml", MINISTUDY) == []


def test_a_cut_left_open_is_found_and_an_end_inside_is_not():
    from types import SimpleNamespace

    design = _chip()
    # line_in's end joins open_b, which the run leaves out: a cut
    assert qdo_pins.open_cuts(design, ["line_in"], [["line_in", "end"]]) == [
        ("line_in", "end", ["open_b"])
    ]
    # with open_b included the pin is a joint inside the model, not a cut
    assert qdo_pins.open_cuts(design, ["line_in", "open_b"], [["line_in", "end"]]) == []
    # a pin of a component the run does not include is not its business
    assert qdo_pins.open_cuts(design, ["line_in"], [["line_out", "start"]]) == []
    # a pin that is also a port ends in the port; a deliberate one is meant
    assert (
        qdo_pins.open_cuts(
            design, ["line_in"], [["line_in", "end"]], ports=[["line_in", "end", 50]]
        )
        == []
    )
    assert (
        qdo_pins.open_cuts(
            design, ["line_in"], [["line_in", "end"]], deliberate=[["line_in", "end"]]
        )
        == []
    )
    # a loose pin in a real netlist is an end of the chip: it may be open
    from qiskit_metal.qlibrary.terminations.open_to_ground import OpenToGround

    OpenToGround(design, "open_e", options=dict(pos_x="5mm", pos_y="5mm"))
    assert qdo_pins.open_cuts(design, ["open_e"], [["open_e", "open"]]) == []
    # a design with no netlist joins nothing: every pin is an end of the chip
    bare = SimpleNamespace(components={"res": SimpleNamespace(pins={"end": {}})})
    assert qdo_pins.open_cuts(bare, ["res"], [["res", "end"]]) == []


def test_the_deep_check_refuses_a_cut_left_open(tiny):
    project, _ = tiny
    project.backend.project.design = _chip
    job_id = project.add("X.freq")
    path = project.queue.folder(job_id) / "ministudy.yaml"
    text = path.read_text("utf-8").replace("res_x", "line_in")
    path.write_text(text, "utf-8")
    assert not any("cut left open" in p for p in project.problems(job_id, deep=True))
    path.write_text(
        text.replace("open_pins: []", "open_pins: [[line_in, end]]"), "utf-8"
    )
    (problem,) = (p for p in project.problems(job_id, deep=True) if "cut" in p)
    assert "line_in/end is a cut left open" in problem and "open_b" in problem
    # a capacitance-only study opens cuts on purpose (Q3D): not refused
    data = yaml.safe_load(path.read_text("utf-8"))
    data["args"]["run_capacitance_studies_only"] = True
    path.write_text(yaml.safe_dump(data), "utf-8")
    assert not any("cut" in p for p in project.problems(job_id, deep=True))
    # one meant open is listed as deliberate, and passes
    data["args"]["run_capacitance_studies_only"] = False
    data["extra"] = {
        **(data.get("extra") or {}),
        "deliberate_open_cuts": [["line_in", "end"]],
    }
    path.write_text(yaml.safe_dump(data), "utf-8")
    assert not any("cut" in p for p in project.problems(job_id, deep=True))


def test_the_pin_picker_warns_of_an_open_cut_in_hfss():
    sel = {"components": ["line_in"], "states": {"line_in/end": ["open", 50]}}
    hfss = str(layout.pins_summary(INVENTORY, sel, {"solver": "hfss", "fields": {}}))
    q3d = str(layout.pins_summary(INVENTORY, sel, {"solver": "q3d", "fields": {}}))
    assert "open cut(s): line_in/end" in hfss and "open cut" not in q3d
