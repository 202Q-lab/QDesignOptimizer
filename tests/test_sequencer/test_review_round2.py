"""The smaller findings of the port's second review (2026-11-12)."""

import math
from pathlib import Path

import numpy as np

from qdesignoptimizer.sequencer.backends import qdo_files as qf
from qdesignoptimizer.sequencer.backends.qdo import (
    _deviation_line,
    _same_value,
    _unit_problems,
    judge_targets,
)
from qdesignoptimizer.sequencer.backends.qdo_templates import fill, split_unit
from qdesignoptimizer.sequencer.engine import state as st

_FREQ = {
    "name": "f",
    "helper": "qdesignoptimizer.utils.optimization_targets:"
    "get_opt_target_res_freq_via_length",
    "args": {"resonator": "resonator_1"},
}


def test_template_expressions_read_design_variables_without_their_unit():
    v = {"design_var_width_qubit_1": "400um", "plain": 3}
    assert fill("=v['design_var_width_qubit_1'] * 2", {}, {}, v) == 800.0
    assert fill("=v['plain'] + 1", {}, {}, v) == 4.0
    assert split_unit("-5.0 um") == (-5.0, "um")
    assert split_unit("12.1nH") == (12.1, "nH")
    assert split_unit("abc") is None


def test_a_study_override_must_keep_the_projects_unit():
    store = {"w": "400um", "lj": "12.1nH", "name": "abc"}
    assert _unit_problems({"w": "410um", "name": "x"}, store) == []
    problems = _unit_problems({"w": 400, "lj": "12.1pH"}, store)
    assert len(problems) == 2
    assert "'none'" in problems[1] and "'um'" in problems[1]  # sorted: w last
    assert "nH" in problems[0]


def test_a_value_in_another_format_is_not_a_move():
    assert _same_value("-5um", "-5.0 um")
    assert not _same_value("-5um", "-6um")
    assert not _same_value("5um", "5nm")


def test_numpy_measurements_are_judged_and_abs_deviations_shown_as_such():
    entries = [
        {**_FREQ, "tolerance": {"abs": 1e6}},
        {
            "name": "chi",
            "target_param_type": "nonlinearity",
            "involved_modes": ["qubit_1", "resonator_1"],
            "design_var": "x",
            "design_var_constraint": {},
            "prop_to": "1",
            "tolerance": {"abs": 0.1e6},
        },
    ]
    measured = {
        "resonator_1_freq": np.float32(6.0005e9),
        "qubit_1_to_resonator_1_nonlin": np.float64(1.05e6),
    }
    params = {"resonator_1_freq": 6e9, "qubit_1_to_resonator_1_nonlin": 1.1e6}
    absolute: set = set()
    judged = judge_targets(entries, measured, params, absolute)
    assert judged["resonator_1_freq"][1] is True
    dev, ok = judged["qubit_1_to_resonator_1_nonlin"]
    assert ok and math.isclose(dev, -0.05e6)
    line = _deviation_line(judged, absolute)
    assert "-5e+04" in line and "%" not in line


def test_a_relative_tolerance_on_a_zero_target_says_why():
    judged = judge_targets(
        [{**_FREQ, "tolerance": 0.1}],
        {"resonator_1_freq": 1.0},
        {"resonator_1_freq": 0},
    )
    assert "give {abs: ...}" in _deviation_line(judged)


def test_numbers_written_yaml_1_1_style_reach_qdo_as_numbers():
    study = qf.load_object(
        {
            "class": "qdesignoptimizer.design_analysis_types:MiniStudy",
            "args": {
                "qiskit_component_names": ["1", "Q"],
                "port_list": [["Q", "2", 50]],
                "open_pins": [],
                "modes": [],
                "jj_setup": {},
                "design_name": "2025",
                "delta_f": "1e-3",
                "nbr_passes": 15,
            },
        }
    )
    assert study.delta_f == 1e-3 and study.nbr_passes == 15
    # Digits that are names stay names (third review).
    assert study.qiskit_component_names == ["1", "Q"]
    assert study.port_list == [["Q", "2", 50]]
    assert study.design_name == "2025"


def test_runner_state_waits_like_a_result_file(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(st, "write_json", lambda p, d, s=1.0: seen.append(s))
    st.RunnerState(status=st.IDLE).save(tmp_path)
    assert seen == [30.0]


def test_a_launched_draft_is_ordered_before_its_studies_appear(tmp_path, monkeypatch):
    """The order is written before the first rename: a runner looking at
    the queue between two renames sees the study sequence in study sequence order,
    not the unlisted folders by name (b_measure before a_measure here)."""
    from qdesignoptimizer.sequencer.app.queue_state import Bridge
    from qdesignoptimizer.sequencer.engine import project as project_module

    from .helpers import make_project

    bridge = Bridge(project=make_project(tmp_path))
    key = bridge.new_draft(label="run", from_sequence="full_A")
    added = bridge.draft_add(key, "B.measure")
    first = bridge.draft_view(key)["studies"][0]["key"]
    bridge.draft_remove(first)
    bridge.draft_move(added, -1)  # study sequence order: B.measure, A.measure
    seen = []
    real = project_module._rename

    def rename(source, dest):
        real(source, dest)
        if Path(dest).parent == bridge.project.queue.queue_dir:
            seen.append(list(bridge.project.queue.ids()))

    monkeypatch.setattr(project_module, "_rename", rename)
    bridge.launch_draft(key)
    # Right after the last rename, before anything else runs.
    assert seen[-1] == ["s01-run-b_measure", "s01-run-a_measure"]
