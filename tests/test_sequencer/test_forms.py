"""Study files as forms: fields, nested studies, switch-off, requeue."""

from __future__ import annotations

import json

import pytest
import yaml

from qdesignoptimizer.sequencer.app import form_layout, forms
from qdesignoptimizer.sequencer.app.callbacks import _settings_from
from qdesignoptimizer.sequencer.app.queue_state import Bridge, GuiError
from qdesignoptimizer.sequencer.backends.qdo_files import active_targets, decode
from qdesignoptimizer.sequencer.engine.queue import RESOLVED
from qdesignoptimizer.sequencer.engine.runner import Runner

from .helpers import make_project

# A MiniStudy file the way a generator writes it, with a chained study.
MINISTUDY = {
    "class": "qdesignoptimizer.design_analysis_types:MiniStudy",
    "args": {
        "qiskit_component_names": ["res_1", "tee_1"],
        "modes": ["resonator_1"],
        "nbr_passes": 8,
        "delta_f": 0.001,
        "max_mesh_length_port": "3um",
        "build_fine_mesh": False,
        "surface_properties": None,
        "jj_setup": {},
        "capacitance_matrix_studies": [
            {
                "class": "qdesignoptimizer.sim_capacitance_matrix:"
                "ResonatorDecayIntoWaveguideStudy",
                "args": {
                    "mode": "resonator_1",
                    "nbr_passes": 8,
                    "percent_error": 0.5,
                    "open_pins": [["res_1", "start"], ["tee_1", "end"]],
                },
            }
        ],
    },
}
NESTED = ("args", "capacitance_matrix_studies", 0)


def _paths(section):
    return {f.path: f for f in forms.fields_of(section)}


def test_every_setting_is_a_field_nested_studies_included():
    form = forms.build(MINISTUDY)
    assert form.title == "MiniStudy"
    fields = _paths(form)
    assert fields[("args", "nbr_passes")].kind == "number"
    assert fields[("args", "build_fine_mesh")].kind == "bool"
    assert fields[("args", "qiskit_component_names")].value == "[res_1, tee_1]"
    nested = fields[NESTED + ("args", "percent_error")]
    assert nested.value == "0.5"
    (studies,) = form.children
    (study,) = studies.children
    assert study.item and study.title == "ResonatorDecayIntoWaveguideStudy"


def test_edits_keep_each_value_s_type():
    edits = {
        forms.key(("args", "nbr_passes")): "12",
        forms.key(("args", "delta_f")): "1e-3",
        forms.key(("args", "max_mesh_length_port")): "5um",
        forms.key(("args", "build_fine_mesh")): ["on"],
        forms.key(("args", "surface_properties")): "",
        forms.key(NESTED + ("args", "percent_error")): "0.2",
        forms.key(NESTED + ("args", "open_pins")): "[[res_1, end]]",
    }
    out = forms.apply(MINISTUDY, edits)
    args = out["args"]
    assert args["nbr_passes"] == 12 and args["delta_f"] == 0.001
    assert args["max_mesh_length_port"] == "5um"
    assert args["build_fine_mesh"] is True
    assert args["surface_properties"] is None
    nested = args["capacitance_matrix_studies"][0]["args"]
    assert nested["percent_error"] == 0.2
    assert nested["open_pins"] == [["res_1", "end"]]
    assert MINISTUDY["args"]["nbr_passes"] == 8  # a copy was edited
    with pytest.raises(forms.FormError, match="not a number"):
        forms.apply(MINISTUDY, {forms.key(("args", "nbr_passes")): "many"})
    with pytest.raises(forms.FormError, match="not a list"):
        forms.apply(MINISTUDY, {forms.key(("args", "modes")): "resonator_1"})


def test_nested_studies_duplicate_remove_and_switch_off():
    two = forms.operate(MINISTUDY, NESTED, "duplicate")
    assert len(two["args"]["capacitance_matrix_studies"]) == 2
    off = forms.operate(two, NESTED, "toggle")
    assert off["args"]["capacitance_matrix_studies"][0]["active"] is False
    # Kept in the file, left out of what is built.
    assert len(off["args"]["capacitance_matrix_studies"]) == 2
    assert decode([{"a": 1, "active": False}, {"b": 2}]) == [{"b": 2}]
    on = forms.operate(off, NESTED, "toggle")
    assert "active" not in on["args"]["capacitance_matrix_studies"][0]
    gone = forms.operate(two, NESTED, "remove")
    assert len(gone["args"]["capacitance_matrix_studies"]) == 1
    with pytest.raises(forms.FormError, match="only an item"):
        forms.operate(MINISTUDY, ("args", "nbr_passes"), "remove")


def test_opt_targets_are_items_that_can_be_switched_off():
    data = {
        "targets": [
            {"helper": "qdo:get_kappa", "args": {"resonator": "r1"}},
            {
                "target_param_type": "freq",
                "involved_modes": ["r1"],
                "design_var": "length_r1",
                "prop_to": "1 / v['length_r1']",
            },
        ]
    }
    form = forms.build(data)
    assert [c.title for c in form.children] == [
        "1. get_kappa",
        "2. freq via length_r1",
    ]
    assert _paths(form)[("targets", 1, "prop_to")].value == "1 / v['length_r1']"
    off = forms.operate(data, ("targets", 0), "toggle")
    assert len(active_targets(off["targets"])) == 1
    assert forms.build(off).note == "1 of 2 switched on"


def test_overrides_write_only_what_the_study_sets():
    project = {"a": 1.0, "b": "3um"}
    rows = forms.overrides({"b": "4um"}, project)
    assert [(r.name, r.own, r.value) for r in rows] == [
        ("a", False, "1.0"),
        ("b", True, "4um"),
    ]
    out = forms.apply_overrides(project, {"a": (True, "2.5"), "b": (False, "9um")})
    assert out == {"a": 2.5}


def test_settings_are_fields_not_json():
    assert _settings_from(
        {"setting:seconds": "5", "setting:passes": "", "notes": "x"}
    ) == {"seconds": 5}


def _component_values(tree) -> dict:
    """Every ff/fo input in a drawn pane, keyed as form_values keys."""
    found = {}

    def walk(node):
        if isinstance(node, (list, tuple)):
            for child in node:
                walk(child)
            return
        ident = getattr(node, "id", None)
        if isinstance(ident, dict) and ident.get("type") == "ff":
            found[f"{ident['file']}|ff|{ident['path']}"] = node.value
        if isinstance(ident, dict) and ident.get("type") == "fo":
            k = f"{ident['file']}|fo|{ident['key']}|{ident['part']}"
            found[k] = node.value
        children = getattr(node, "children", None)
        if children is not None:  # a component's len() can be 0
            walk(children)

    walk(tree)
    return found


def test_the_drawn_form_is_what_unsaved_is_measured_against():
    spec = {
        "name": "ministudy.yaml",
        "label": "MiniStudy",
        "data": MINISTUDY,
        "optional": False,
    }
    drawn = _component_values(form_layout.file_form(spec, True))
    assert drawn == form_layout.form_values(spec, True)
    spec = {
        "name": "design_variables.json",
        "label": "Design variables",
        "data": {"b": 2},
        "optional": True,
        "project": {"a": 1, "b": 0},
    }
    drawn = _component_values(form_layout.file_form(spec, True))
    assert drawn == form_layout.form_values(spec, True)


@pytest.fixture
def bridge(tmp_path):
    return Bridge(project=make_project(tmp_path))


def test_a_form_save_writes_the_file_keeping_its_header(bridge):
    bridge.add("task", "A.calibrate")
    job_id = bridge.project.queue.ids()[0]
    folder = bridge.project.queue.folder(job_id)
    before = (folder / "task.yaml").read_text(encoding="utf-8")
    text = bridge.save_form(job_id, "task.yaml", edits={forms.key(("value",)): "4.5"})
    assert "saved" in text
    after = (folder / "task.yaml").read_text(encoding="utf-8")
    assert yaml.safe_load(after)["value"] == 4.5
    assert after.splitlines()[0] == before.splitlines()[0]  # the header
    with pytest.raises(GuiError, match="not a number"):
        bridge.save_form(job_id, "task.yaml", edits={forms.key(("value",)): "lots"})


def test_overrides_create_and_delete_the_optional_file(bridge):
    bridge.add("task", "A.calibrate")
    job_id = bridge.project.queue.ids()[0]
    path = bridge.project.queue.folder(job_id) / "design_variables.json"
    bridge.save_form(
        job_id,
        "design_variables.json",
        overrides={"A_value": (True, "2"), "B_value": (False, "9")},
    )
    assert json.loads(path.read_text()) == {"A_value": 2}
    text = bridge.save_form(
        job_id, "design_variables.json", overrides={"A_value": (False, "2")}
    )
    assert "from the project" in text and not path.exists()


def test_requeue_fresh_pulls_again_exact_repeats_the_run(bridge):
    project = bridge.project
    bridge.add("task", "A.calibrate")
    Runner(project).run()
    (done,) = project.queue.completed()
    assert (done / RESOLVED / "design_variables.json").exists()
    ran_with = json.loads((done / RESOLVED / "design_variables.json").read_text())
    project.backend.commit({"A_value": 7.0})  # the project moves on

    fresh = bridge.requeue(done.name).split()[-1]
    exact = bridge.requeue(done.name, exact=True).split()[-1]
    fresh_dir = project.queue.folder(fresh)
    exact_dir = project.queue.folder(exact)
    assert not (fresh_dir / "design_variables.json").exists()
    assert json.loads((exact_dir / "design_variables.json").read_text()) == ran_with
    assert project.job(exact).raw["requeued_as"] == "exact copy"
    assert project.job(fresh).raw["requeued_as"] == "fresh"
    assert bridge.details("completed", done.name)["has_resolved"] is True
