"""Studies made from scratch, items added from a class, target templates."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from qdesignoptimizer.sequencer.app import forms
from qdesignoptimizer.sequencer.app.queue_state import Bridge, GuiError
from qdesignoptimizer.sequencer.engine import state as st
from qdesignoptimizer.sequencer.engine.job import is_custom
from qdesignoptimizer.sequencer.engine.runner import Runner

_DEMO = (
    Path(__file__).resolve().parents[2]
    / "tutorials"
    / "example_simulation_sequencer"
    / "demo"
)
CAPS = forms.key(("args", "capacitance_matrix_studies"))
TARGETS = forms.key(("targets",))


@pytest.fixture
def qdo(tmp_path):
    root = tmp_path / "chip_qdo"
    shutil.copytree(
        _DEMO / "chip_qdo",
        root,
        ignore=shutil.ignore_patterns(
            "out", "design_variables.json", "*.meta.json", "campaigns"
        ),
    )
    bridge = Bridge([str(root)])  # worker process, as the GUI runs it
    yield bridge, root
    bridge.close()


def _file(bridge, key, name):
    data = bridge.details("hold", key)
    return next(f for f in data["files"] if f["name"] == name)


def _choice(spec, path, label):
    return next(c["value"] for c in spec["addable"][path] if c["label"] == label)


def test_a_study_from_scratch_is_filled_in_saved_as_templates_and_run(qdo):
    bridge, root = qdo
    study = bridge.new_study(name="r1 kappa")
    assert study.startswith("@hold/")
    (held,) = bridge.snapshot()["held"]
    assert is_custom(held["task"])
    ministudy = _file(bridge, study, "ministudy.yaml")
    assert ministudy["data"]["args"]["qiskit_component_names"] == []
    assert ministudy["data"]["args"]["capacitance_matrix_studies"] is None
    assert any("lists no targets" in p for p in bridge._problems(study))

    # The MiniStudy's own fields, then a chained study from its class.
    path = lambda *p: forms.key(("args",) + p)  # noqa: E731
    bridge.save_form(
        study,
        "ministudy.yaml",
        edits={
            path("qiskit_component_names"): "[res_R1, tee_R1]",
            path("modes"): "[resonator_R1]",
            path("run_capacitance_studies_only"): ["on"],
        },
    )
    decay = _choice(ministudy, CAPS, "ResonatorDecayIntoWaveguideStudy")
    assert "added" in bridge.form_add(study, "ministudy.yaml", forms.unkey(CAPS), decay)
    cap = lambda f: path("capacitance_matrix_studies", 0, "args", f)  # noqa
    bridge.save_form(
        study,
        "ministudy.yaml",
        edits={
            cap("mode"): "resonator_R1",
            cap("mode_freq_GHz"): "5.2",
            cap("resonator_name"): "res_R1",
            cap("waveguide_name"): "tee_R1",
            cap("impedance_ohm"): "50",
            cap("qiskit_component_names"): "[res_R1, tee_R1]",
            cap("open_pins"): "[[res_R1, end]]",
        },
    )

    # A target from one of QDO's helpers.
    targets = _file(bridge, study, "opt_targets.yaml")
    kappa = _choice(targets, TARGETS, "get_opt_target_res_kappa_via_coupl_length")
    bridge.form_add(study, "opt_targets.yaml", ["targets"], kappa)
    entry = ("targets", 0, "args")
    bridge.save_form(
        study,
        "opt_targets.yaml",
        edits={
            forms.key(entry + ("resonator",)): "resonator_R1",
            forms.key(entry + ("resonator_coupled_identifier",)): "tee",
        },
    )
    assert bridge._deep_check(study) == []

    # The targets alone, as a template; then into another blank study.
    text = bridge.save_targets_template(study, "opt_targets.yaml", "r1_kappa_only")
    assert "opt_targets/r1_kappa_only.yaml" in text
    other = bridge.new_study(name="second")
    assert (
        "r1_kappa_only" in _file(bridge, other, "opt_targets.yaml")["target_templates"]
    )
    bridge.load_targets(other, "opt_targets.yaml", "r1_kappa_only")
    loaded = yaml.safe_load(_file(bridge, other, "opt_targets.yaml")["text"])
    assert loaded["targets"][0]["args"]["resonator"] == "resonator_R1"
    bridge.remove(other)
    # Saved once: the project has it now, so there is nothing to add.
    with pytest.raises(GuiError, match="already has every opt target"):
        bridge.save_targets_template(study, "opt_targets.yaml", "r1_kappa_only")

    # The whole study as a template: it is in the Add menu then.
    bridge.save_as_template("hold", study, "r1_from_scratch")
    keys = {c["key"] for c in bridge.snapshot()["catalog"]}
    assert "r1_from_scratch" in keys

    # Queued and run like any other.
    bridge.save_job(study, {"settings": {"iterations": 3}})
    bridge.unpark(study)
    project = bridge.project
    final = Runner(bridge.workspace).run()
    assert final.status == st.FINISHED, final.message
    (done,) = project.queue.completed()
    assert st.result_of(done)["committed"]  # the coupling length moved


def test_a_custom_study_gets_its_family_from_its_ministudy():
    from qdesignoptimizer.sequencer.backends.qdo import QDOBackend

    class Job:
        task, subject = "custom.x", "custom"

    spec = QDOBackend._custom_spec(None, Job(), {"args": {}})
    assert spec.family == "eigenmode"
    capacitance = {"args": {"run_capacitance_studies_only": True}}
    assert QDOBackend._custom_spec(None, Job(), capacitance).family == ("capacitance")


def test_a_study_whose_target_is_still_blank_does_not_run(qdo):
    """Passes the quick check (it has a target), fails the deep one (the
    target names no design variable): queued, the runner stops at it
    with the reason rather than solving it."""
    bridge, _ = qdo
    study = bridge.new_study(name="half")
    targets = _file(bridge, study, "opt_targets.yaml")
    kappa = _choice(targets, TARGETS, "get_opt_target_res_kappa_via_coupl_length")
    bridge.form_add(study, "opt_targets.yaml", ["targets"], kappa)
    scoped, local = bridge._study(study)
    assert scoped.problems(local) == []  # the quick check passes
    assert scoped.problems(local, deep=True)  # the deep one does not
    bridge.unpark(study)
    final = Runner(bridge.workspace).run()
    assert final.status == st.ERROR and "cannot run" in final.message
    assert bridge.project.queue.ids() == [local]
