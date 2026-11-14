"""The QDO backend on a tiny project made of real QDO objects (no Ansys).

Skipped when qdesignoptimizer is not importable.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

pytest.importorskip("qdesignoptimizer")

from qdesignoptimizer.sequencer.backends.qdo import QDOBackend  # noqa: E402
from qdesignoptimizer.sequencer.engine import state as st  # noqa: E402
from qdesignoptimizer.sequencer.engine.project import Project  # noqa: E402
from qdesignoptimizer.sequencer.engine.runner import Runner  # noqa: E402
from qdesignoptimizer.utils import optimization_targets as qdo_ot  # noqa

RES = "resonator_x"
LENGTH = "design_var_length_resonator_x"
WIDTH = "design_var_width_x"


# A ministudy template is the MiniStudy alone. Its file, X_freq.yaml,
# makes it the study "freq" of subsystem X.
_STUDY = """
class: qdesignoptimizer.design_analysis_types:MiniStudy
args:
  qiskit_component_names: [res_x]
  port_list: []
  open_pins: []
  modes: [resonator_x]
  # A non-linear junction: QDO runs EPR, so the nonlinearity is measured.
  jj_setup:
    jj_res_x:
      rect: JJ_rect_res_x
      line: JJ_line_res_x
      Lj_variable: design_var_lj_x
      Cj_variable: design_var_cj_x
extra:
  grounded_pins: [[res_x, end]]
"""

# The project's opt targets, each named. A study takes those of its
# modes: two here, and not the one of another resonator.
_TARGETS = """
targets:
- name: x_freq
  helper: qdesignoptimizer.utils.optimization_targets:get_opt_target_res_freq_via_length
  args: {resonator: resonator_x}
- name: other_freq
  helper: qdesignoptimizer.utils.optimization_targets:get_opt_target_res_freq_via_length
  args: {resonator: resonator_other}
- name: x_nonlin
  target_param_type: nonlinearity
  involved_modes: [resonator_x, resonator_x]
  design_var: design_var_width_x
  design_var_constraint: {larger_than: 1um, smaller_than: 50um}
  prop_to: "v['design_var_width_x'] ** (-1.17)"
"""


class TinyProject:
    """One ministudy template, ``X.freq``, and the project's opt
    targets: a resonator frequency by its length (QDO's helper) and an
    explicit target with ``prop_to`` as an expression."""

    def __init__(self, root: Path):
        self.root = root
        self.design_variables_file = root / "design_variables.json"
        self.design_variables_file.write_text(
            json.dumps({LENGTH: "5000um", WIDTH: "10um", "design_var_gap": "5um"})
        )
        (root / "ministudies").mkdir()
        (root / "opt_targets").mkdir()
        (root / "ministudies" / "X_freq.yaml").write_text(_STUDY)
        (root / "opt_targets" / "resonators.yaml").write_text(_TARGETS)
        self.subjects = {"X": ("Resonator X", "tiny")}
        self.solved = []
        self.components = {"res_x": {"start", "end"}}

    def settings_keys(self):
        return {"iterations", "strict_modes"}

    def default_settings(self, task):
        return {"iterations": 1}

    def modes_checked(self, settings):
        return settings.get("strict_modes", True)

    def design(self):
        return SimpleNamespace(
            components={
                name: SimpleNamespace(pins={pin: {} for pin in pins})
                for name, pins in self.components.items()
            }
        )

    def parameter_targets(self):
        return {
            "resonator_x_freq": 5e9,
            "resonator_x_kappa": 1e6,
            "resonator_x_to_resonator_x_nonlin": 200e6,
        }

    def before_step(self):
        pass

    def solve(
        self,
        study,
        targets,
        folder,
        settings,
        design_variables,
        parameter_targets,
    ):
        self.solved.append(
            (study, targets, dict(design_variables), dict(parameter_targets))
        )
        moved = dict(design_variables)
        moved[LENGTH] = "5100um"  # tuned by a target
        moved["design_var_gap"] = "6um"  # not tuned: must not be saved
        base = Path(folder) / "run"
        Path(str(base) + "_design_variables.json").write_text(json.dumps(moved))
        return SimpleNamespace(
            save_path=str(base),
            optimization_results=[
                {"system_optimized_params": {"resonator_x_freq": 5.01e9}}
            ],
        )

    def converged(self, analysis):
        return {"resonator_x_freq": (1e7, True)}


@pytest.fixture
def tiny(tmp_path):
    project = TinyProject(tmp_path)
    return project, Project(QDOBackend(project))


def _files(project, job_id):
    folder = project.queue.folder(job_id)
    return {p.name: p.read_text("utf-8") for p in folder.iterdir()}


def test_generate_writes_the_two_files(tiny):
    _, project = tiny
    job_id = project.add("X.freq", save=True)
    files = _files(project, job_id)
    study = yaml.safe_load(files["ministudy.yaml"])
    assert study["class"].endswith(":MiniStudy")
    assert study["extra"] == {"grounded_pins": [["res_x", "end"]]}
    targets = yaml.safe_load(files["opt_targets.yaml"])["targets"]
    # The project's targets of this study's mode, by name; not the one
    # of another resonator.
    assert [t["name"] for t in targets] == ["x_freq", "x_nonlin"]
    assert targets[0]["helper"].endswith(":get_opt_target_res_freq_via_length")
    assert targets[0]["args"] == {"resonator": RES}
    assert targets[1]["prop_to"] == f"v['{WIDTH}'] ** (-1.17)"
    assert project.problems(job_id, deep=True) == []
    assert "job.yaml" in files and "study.yaml" not in files


def test_a_bad_edit_is_caught_before_the_run(tiny):
    _, project = tiny
    job_id = project.add("X.freq")
    path = project.queue.folder(job_id) / "opt_targets.yaml"
    path.write_text(
        path.read_text("utf-8").replace(f"v['{WIDTH}']", "v['design_var_typo']"),
        "utf-8",
    )
    problems = project.problems(job_id, deep=True)
    assert any("design_var_typo" in p for p in problems)
    (project.queue.folder(job_id) / "parameter_targets.yaml").write_text(
        "resonator_x_frq: 5e9\n"
    )
    assert any("resonator_x_frq" in p for p in project.problems(job_id))


def test_a_run_writes_back_only_what_a_target_tunes(tiny):
    qdo_project, project = tiny
    job_id = project.add("X.freq", save=True)
    project.queue.write_file(
        job_id, "parameter_targets.yaml", "resonator_x_freq: 4.9e9\n"
    )
    project.queue.write_file(
        job_id, "design_variables.json", json.dumps({WIDTH: "12um"})
    )
    final = Runner(project).run()
    assert final.status == st.FINISHED
    (done,) = project.queue.completed()
    rec = json.loads((done / "result.json").read_text("utf-8"))
    assert rec["status"] == "done"
    assert rec["committed"] == {LENGTH: "5100um"}
    store = json.loads(qdo_project.design_variables_file.read_text())
    assert store == {LENGTH: "5100um", WIDTH: "10um", "design_var_gap": "5um"}
    # what ran: the project's, with the folder's values on top
    _, _, ran_vars, ran_params = qdo_project.solved[0]
    assert ran_vars[WIDTH] == "12um" and ran_vars["design_var_gap"] == "5um"
    assert ran_params == {
        "resonator_x_freq": 4.9e9,
        "resonator_x_kappa": 1e6,
        "resonator_x_to_resonator_x_nonlin": 200e6,
    }
    written = yaml.safe_load((done / "parameter_targets.yaml").read_text())
    assert written == ran_params


def test_the_rebuilt_objects_are_what_ran(tiny):
    qdo_project, project = tiny
    project.add("X.freq")
    Runner(project).run()
    study, targets, *_ = qdo_project.solved[0]
    assert study.grounded_pins == [["res_x", "end"]]
    assert targets[0].design_var == LENGTH
    assert targets[1].prop_to({}, {WIDTH: 2.0}) == pytest.approx(2.0**-1.17)


def test_mode_identity_switched_off_fails_the_study(tiny):
    qdo_project, project = tiny
    project.add(
        "X.freq",
        save=True,
        checks=["mode_identity"],
        settings={"strict_modes": False},
        on_failure="continue",
    )
    Runner(project).run()
    (done,) = project.queue.completed()
    rec = json.loads((done / "result.json").read_text("utf-8"))
    assert rec["status"] == "failed" and "switched off" in rec["message"]
    store = json.loads(qdo_project.design_variables_file.read_text())
    assert store[LENGTH] == "5000um"


def test_a_malformed_targets_file_is_a_problem_not_a_crash(tiny):
    _, project = tiny
    job_id = project.add("X.freq")
    project.queue.write_file(job_id, "opt_targets.yaml", "- a\n- b\n")
    problems = project.problems(job_id, deep=True)
    # Said plainly now, rather than as "cannot be checked: AttributeError".
    assert problems and "lists no targets" in problems[0]
    final = Runner(project).run()
    assert final.status == st.ERROR
    assert project.queue.ids() == [job_id]


# -- templates and names ---------------------------------------------------


def test_the_templates_are_the_catalog(tiny):
    qdo_project, project = tiny
    catalog = project.backend.catalog()
    (task,) = catalog.tasks
    assert task.key == "X.freq" and task.label == "Resonator X freq"
    assert task.family == "eigenmode"
    assert "ministudies/X_freq.yaml" in task.summary
    assert catalog.subjects[0].label == "Resonator X"


def test_a_misspelt_component_or_pin_is_caught(tiny):
    qdo_project, project = tiny
    job_id = project.add("X.freq")
    assert project.problems(job_id, deep=True) == []
    path = project.queue.folder(job_id) / "ministudy.yaml"
    text = path.read_text("utf-8")
    path.write_text(text.replace("- end", "- edn"), "utf-8")
    problems = project.problems(job_id, deep=True)
    assert any("has no pin 'edn'" in p and "did you mean end" in p for p in problems)
    path.write_text(text.replace("- res_x", "- res_y"), "utf-8")
    problems = project.problems(job_id, deep=True)
    assert any("no component 'res_y'" in p for p in problems)


def test_the_design_is_rebuilt_only_when_its_files_change(tiny, tmp_path):
    qdo_project, project = tiny
    calls = []
    original = qdo_project.design
    qdo_project.design = lambda: calls.append(1) or original()
    marker = tmp_path / "design.py"
    marker.write_text("x = 1")
    qdo_project.design_files = lambda: [marker]
    project.backend.design_names()
    project.backend.design_names()
    assert len(calls) == 1
    import os

    st_ = marker.stat()
    os.utime(marker, ns=(st_.st_atime_ns, st_.st_mtime_ns + 10**9))
    project.backend.design_names()
    assert len(calls) == 2


# -- templates with placeholders -------------------------------------------

# {r} is a value of subsystem X, so this makes one study for it: X.res.
_SKELETON = """
class: qdesignoptimizer.design_analysis_types:MiniStudy
args:
  qiskit_component_names: ["res_{r}"]
  port_list: []
  open_pins: []
  modes: ["resonator_{r}"]
  delta_f: "=p['resonator_{r}_kappa'] / 1e9"
extra:
  grounded_pins: [["res_{r}", end]]
"""

_SKELETON_TARGETS = """
targets:
- name: "{r}_kappa"
  helper: qdesignoptimizer.utils.optimization_targets:get_opt_target_res_kappa_via_coupl_length
  args: {resonator: "resonator_{r}", resonator_coupled_identifier: tee}
  tolerance: 0.05
"""


def _skeletons(
    qdo_project,
    ministudy=_SKELETON,
    targets=_SKELETON_TARGETS,
    stem="res",
):
    qdo_project.subjects = {"X": ("Resonator X", "tiny", {"r": "x"})}
    (qdo_project.root / "ministudies").mkdir(exist_ok=True)
    (qdo_project.root / "opt_targets").mkdir(exist_ok=True)
    (qdo_project.root / "ministudies" / f"{stem}.yaml").write_text(ministudy)
    (qdo_project.root / "opt_targets" / "res_kappa.yaml").write_text(targets)


def test_a_skeleton_is_in_the_catalog_and_queues_filled_in(tiny):
    qdo_project, project = tiny
    _skeletons(qdo_project)
    task = project.backend.catalog().task("X.res")
    assert task is not None and task.label == "Resonator X res"
    job_id = project.add("X.res")
    files = _files(project, job_id)
    study = yaml.safe_load(files["ministudy.yaml"])
    assert study["args"]["qiskit_component_names"] == ["res_x"]
    assert study["args"]["delta_f"] == pytest.approx(1e6 / 1e9)
    assert study["extra"]["grounded_pins"] == [["res_x", "end"]]
    targets = {
        t["name"]: t for t in yaml.safe_load(files["opt_targets.yaml"])["targets"]
    }
    assert set(targets) == {"x_freq", "x_nonlin", "x_kappa"}
    assert targets["x_kappa"]["args"]["resonator"] == "resonator_x"
    # An eigenmode study: the linewidth is a capacitance study's to
    # tune, so it is carried switched off and can be switched on.
    assert targets["x_kappa"]["active"] is False
    assert "active" not in targets["x_freq"]
    assert project.problems(job_id, deep=True) == []
    assert "ministudies/res.yaml" in project.job(job_id).generated_by
    # X_freq.yaml has no placeholders, but it is subsystem X's: it takes
    # the targets written for X as well.
    plain = project.add("X.freq")
    names = [
        t["name"]
        for t in yaml.safe_load(_files(project, plain)["opt_targets.yaml"])["targets"]
    ]
    assert sorted(names) == ["x_freq", "x_kappa", "x_nonlin"]


def test_skeleton_problems_are_reported(tiny):
    qdo_project, project = tiny
    _skeletons(qdo_project, ministudy=_SKELETON.replace('"res_{r}"]', '"res_{q}"]'))
    project.backend.catalog()
    assert any("no subsystem sets" in p for p in project.backend.catalog_problems)
    with pytest.raises(Exception, match="no study 'X.res'"):
        project.add("X.res")
    # Two templates that make the same study.
    _skeletons(qdo_project, stem="freq")
    project.backend.catalog()
    assert any("X.freq is also defined" in p for p in project.backend.catalog_problems)
    # A header is not a MiniStudy.
    _skeletons(qdo_project, ministudy="study: {name: res}\n" + _SKELETON)
    project.backend.catalog()
    assert any("the MiniStudy alone" in p for p in project.backend.catalog_problems)


_MORE_TARGETS = """
targets:
- name: x_freq
  helper: qdesignoptimizer.utils.optimization_targets:get_opt_target_res_freq_via_length
  args: {resonator: resonator_x}
- helper: qdesignoptimizer.utils.optimization_targets:get_opt_target_res_freq_via_length
  args: {resonator: resonator_x}
"""


def _target_names(project, job_id):
    data = yaml.safe_load(_files(project, job_id)["opt_targets.yaml"])
    return [t["name"] for t in data["targets"]]


def test_opt_target_names_are_unique_in_the_project(tiny):
    qdo_project, project = tiny
    assert project.check_templates()["X.freq"] == []
    (qdo_project.root / "opt_targets" / "more.yaml").write_text(_MORE_TARGETS)
    project.backend.catalog()
    problems = project.backend.catalog_problems
    assert any(
        "target name 'x_freq' is also used in opt_targets/more.yaml" in p
        and "unique in the project" in p
        for p in problems
    )
    assert any("target 2 needs a name" in p for p in problems)
    # check-templates says so too, though they belong to no one study.
    assert problems == project.check_templates()["(templates)"]
    # Neither is used: the study still has each target once.
    assert _target_names(project, project.add("X.freq")) == [
        "x_freq",
        "x_nonlin",
    ]


def test_a_single_opt_targets_file_is_read_too(tiny):
    qdo_project, project = tiny
    (qdo_project.root / "opt_targets" / "resonators.yaml").rename(
        qdo_project.root / "opt_targets.yaml"
    )
    assert _target_names(project, project.add("X.freq")) == [
        "x_freq",
        "x_nonlin",
    ]


def _edit_targets(project, job_id, change):
    path = project.queue.folder(job_id) / "opt_targets.yaml"
    data = yaml.safe_load(path.read_text("utf-8"))
    change(data["targets"])
    path.write_text(yaml.safe_dump(data), "utf-8")


def test_a_target_is_judged_by_its_own_tolerance(tiny):
    qdo_project, project = tiny
    qdo_project.converged = lambda analysis: {}  # the hook says nothing

    def loose(targets):
        targets[0]["tolerance"] = {"abs": 2.0e7}  # solved 10 MHz off
        targets[1]["active"] = False

    job_id = project.add("X.freq", save=True, checks=["converged"])
    _edit_targets(project, job_id, loose)
    Runner(project).run()
    (done,) = project.queue.completed()
    rec = json.loads((done / "result.json").read_text("utf-8"))
    assert rec["status"] == "done" and rec["outputs"]["converged"] is True
    # The switched-off target stays in the record, and did not run.
    kept = yaml.safe_load((done / "opt_targets.yaml").read_text("utf-8"))
    assert kept["targets"][1]["active"] is False
    built = qdo_project.solved[0][1]
    assert [t.design_var for t in built] == [LENGTH]

    def tight(targets):
        targets[0]["tolerance"] = 0.001  # relative: 5 MHz
        targets[1]["active"] = False

    again = project.add(
        "X.freq", save=True, checks=["converged"], on_failure="continue"
    )
    _edit_targets(project, again, tight)
    Runner(project).run()
    (failed,) = (d for d in project.queue.completed() if d.name.endswith("_failed"))
    rec = json.loads((failed / "result.json").read_text("utf-8"))
    assert "off target: resonator_x_freq" in rec["message"]


def test_a_target_nobody_judges_is_not_converged(tiny):
    qdo_project, project = tiny
    qdo_project.converged = None  # the hook is optional

    def only_one_judged(targets):
        targets[0]["tolerance"] = {"abs": 2.0e7}  # x_nonlin sets none

    job_id = project.add(
        "X.freq", save=True, checks=["converged"], on_failure="continue"
    )
    _edit_targets(project, job_id, only_one_judged)
    Runner(project).run()
    (done,) = project.queue.completed()
    rec = json.loads((done / "result.json").read_text("utf-8"))
    assert rec["status"] == "failed"
    assert "not judged" in rec["message"]
    assert rec["outputs"]["unjudged"] == ["resonator_x_to_resonator_x_nonlin"]
    # Not written back: an unjudged study is not a tuned one.
    store = json.loads(qdo_project.design_variables_file.read_text())
    assert store[LENGTH] == "5000um"


def test_saving_a_family_study_adds_no_target_twice(tiny):
    qdo_project, project = tiny
    _skeletons(qdo_project)
    job_id = project.add("X.res")
    paths = project.backend.save_template(
        project.queue.folder(job_id), project.job(job_id), "copy"
    )
    # x_kappa is the project's "{r}_kappa": nothing new for opt_targets/.
    assert [p.name for p in paths] == ["X_copy.yaml"]
    names = _target_names(project, project.add("X.copy"))
    assert sorted(names) == ["x_freq", "x_kappa", "x_nonlin"]


def test_two_targets_with_one_name_once_filled_in_are_refused(tiny):
    qdo_project, project = tiny
    _skeletons(
        qdo_project,
        targets=_SKELETON_TARGETS.replace('"{r}_kappa"', '"{r}_freq"'),
    )
    with pytest.raises(Exception, match="called 'x_freq' once filled in"):
        project.add("X.res")


def test_a_bad_tolerance_is_a_problem(tiny):
    _, project = tiny
    job_id = project.add("X.freq")

    def bad(targets):
        targets[0]["tolerance"] = "close"

    _edit_targets(project, job_id, bad)
    assert any("tolerance" in p for p in project.problems(job_id, deep=True))


def test_save_as_template_round_trips(tiny):
    from qdesignoptimizer.sequencer.app.queue_state import Bridge

    qdo_project, project = tiny
    job_id = project.add("X.freq")
    bridge = Bridge(project=project)
    text = bridge.save_as_template("queue", job_id, "freq_tpl")
    assert "ministudies" in text and "Add menu" in text
    # Kept with its subsystem, and nothing written to opt_targets/: the
    # study's targets are the project's already.
    assert (qdo_project.root / "ministudies" / "X_freq_tpl.yaml").exists()
    assert not (qdo_project.root / "opt_targets" / "X_freq_tpl.yaml").exists()
    assert project.backend.catalog().task("X.freq_tpl") is not None
    copy = project.add("X.freq_tpl")
    a = yaml.safe_load(_files(project, job_id)["ministudy.yaml"])
    b = yaml.safe_load(_files(project, copy)["ministudy.yaml"])
    assert a == b
    with pytest.raises(Exception, match="already exist"):
        bridge.save_as_template("queue", job_id, "freq_tpl")


# -- safeguards for a templates-only project --------------------------------


def test_a_template_can_take_names_from_the_project(tiny):
    qdo_project, project = tiny
    qdo_project.template_values = lambda: {"res": "res_x"}
    _skeletons(
        qdo_project,
        ministudy=_SKELETON.replace('["res_{r}"]', '["{res}"]'),
    )
    job_id = project.add("X.res")
    study = yaml.safe_load(
        (project.queue.folder(job_id) / "ministudy.yaml").read_text()
    )
    assert study["args"]["qiskit_component_names"] == ["res_x"]


def test_a_misspelt_mode_is_caught_with_a_suggestion(tiny):
    _, project = tiny
    job_id = project.add("X.freq")
    path = project.queue.folder(job_id) / "ministudy.yaml"
    path.write_text(path.read_text().replace("- resonator_x", "- resonatr_x"))
    problems = project.problems(job_id)  # the quick check
    assert any(
        "'resonatr_x' is the mode of no parameter target" in p
        and "did you mean resonator_x" in p
        for p in problems
    )


def test_a_mode_name_cut_short_is_caught(tiny):
    """``resonator`` is the start of ``resonator_x_freq``, yet no mode."""
    _, project = tiny
    job_id = project.add("X.freq")
    path = project.queue.folder(job_id) / "ministudy.yaml"
    path.write_text(path.read_text().replace("- resonator_x", "- resonator"))
    assert any(
        "'resonator' is the mode of no parameter target" in p
        for p in project.problems(job_id)
    )


def test_modes_are_read_exactly_off_qdo_parameter_names():
    from qdesignoptimizer.sequencer.backends.qdo import _param_modes

    assert _param_modes(
        [
            "resonator_a_freq",
            "resonator_a_kappa",
            "qubit_b_charge_line_limited_t1",
            "qubit_b_to_resonator_a_nonlin",
            "qubit_c_in_jj1_participation_ratio",
            "pad_to_ground_capacitance",
        ]
    ) == {"resonator_a", "qubit_b", "qubit_c"}


def test_check_templates_names_a_broken_one_and_queues_nothing(tiny):
    qdo_project, project = tiny
    _skeletons(qdo_project, ministudy=_SKELETON.replace('["res_{r}"]', '["res_y"]'))
    found = project.check_templates()
    assert found["X.freq"] == []
    assert any("no component 'res_y'" in p for p in found["X.res"])
    assert project.queue.ids() == []
