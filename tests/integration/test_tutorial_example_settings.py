"""Readable checks that the notebook settings reach the simulators as intended.

The snapshot tests catch *any* change; these tests spell out the most important
expectations so that a failure points directly at the broken setting.
"""

import json

import pytest

from .conftest import EXAMPLE_CASES
from .example_harness import events, notebook_code_cells

ANALYZE = "hfss.eigenmode.analyze"
Q3D_RUN = "q3d.run"
EPR = "epr.analyze_all_variations"


def _values(payloads, key):
    return [payload[key] for payload in payloads]


@pytest.mark.parametrize(
    "name",
    [
        case.name
        for case in EXAMPLE_CASES
        if case.name != "tunable_coupler__eigenmode_two_qubit_resonator_partitioned"
    ],
)
def test_simulated_variables_equal_stored_design_variables(name, run_example):
    """Each simulation must use exactly the design variables stored for that iteration."""
    run = run_example(name)
    simulated = [p["hfss_variables"] for p in events(run.session, ANALYZE)] or [
        p["design_variables"] for p in events(run.session, Q3D_RUN)
    ]
    stored = [r["design_variables"] for r in run.optimization_results]
    assert simulated == stored


@pytest.mark.parametrize(
    "name",
    [
        case.name
        for case in EXAMPLE_CASES
        if any(
            "overwrite_parameters()" in cell
            and not any(marker in cell for marker in case.skip_cells_containing)
            for cell in notebook_code_cells(case.notebook)
        )
    ],
)
def test_overwrite_parameters_writes_final_design_variables(name, run_example):
    run = run_example(name)
    written = json.loads((run.workdir / "design_variables.json").read_text())
    final = run.optimization_results[-1]["design_variables"]
    for design_var in {target.design_var for target in run.analysis.opt_targets}:
        assert written[design_var] == final[design_var]


@pytest.mark.parametrize(
    "name",
    [
        "coupled_transmon__eigenmode_single_qubit_resonator",
        "coupled_transmon__eigenmode_single_qubit_resonator_exponent_study",
        "coupled_transmon__eigenmode_single_qubit_resonator_pratio",
        "coupled_transmon__eigenmode_two_qubit_resonator",
        "tunable_coupler__eigenmode_two_qubit_resonator",
        "tunable_coupler__eigenmode_two_qubit_resonator_model_inaccuracy",
        "tunable_coupler__eigenmode_two_qubit_resonator_model_inaccuracy_qb_res",
        "tunable_coupler__eigenmode_two_qubit_resonator_partitioned",
    ],
)
def test_every_eigenmode_simulation_is_followed_by_one_epr_analysis(name, run_example):
    run = run_example(name)
    sequence = [event for event, _ in run.session.events if event in (ANALYZE, EPR)]
    assert sequence == [ANALYZE, EPR] * (len(sequence) // 2)
    assert not events(run.session, Q3D_RUN)


@pytest.mark.parametrize(
    "name",
    [
        "coupled_transmon__capacitance_target_resonator",
        "coupled_transmon__charge_line_decay_single_qubit",
        "coupled_transmon__kappa_target_resonator",
    ],
)
def test_capacitance_only_examples_never_run_hfss_or_epr(name, run_example):
    run = run_example(name)
    assert not events(run.session, ANALYZE)
    assert not events(run.session, EPR)
    assert len(events(run.session, Q3D_RUN)) == len(run.optimization_results) == 10


# ------------------------------------------------------- examples_coupled_transmon_chip
def test_single_qubit_resonator(run_example):
    run = run_example("coupled_transmon__eigenmode_single_qubit_resonator")
    analyses = events(run.session, ANALYZE)
    assert _values(analyses, "passes") == [10] * 10
    assert set(_values(analyses, "delta_f")) == {0.001}
    assert set(_values(analyses, "n_modes")) == {2}
    assert [list(j) for j in _values(analyses, "junctions")] == [
        ["jj_name_qubit_1_"]
    ] * 10
    assert _values(events(run.session, EPR), "cos_trunc") == [8] * 10
    assert _values(events(run.session, EPR), "fock_trunc") == [7] * 10

    renders = events(run.session, "hfss.renderer.render_design")
    assert {tuple(r["selection"]) for r in renders} == {
        ("name_qubit_1_", "name_resonator_1_", "name_tee_1_")
    }
    assert renders[0]["port_list"] == [
        ["name_tee_1_", "prime_end", 50],
        ["name_tee_1_", "prime_start", 50],
    ]
    # build_fine_mesh=True + MeshingMap(CoupledLineTee) refines the tee every iteration
    meshes = events(run.session, "hfss.modeler.mesh_length")
    assert (
        meshes
        == [
            {
                "name": "fine_mesh",
                "objects": ["prime_cpw_name_tee_1_", "second_cpw_name_tee_1_"],
                "MaxLength": "5um",
                "RefineInside": True,
            }
        ]
        * 10
    )


def test_single_qubit_resonator_exponent_study(run_example):
    run = run_example(
        "coupled_transmon__eigenmode_single_qubit_resonator_exponent_study"
    )
    assert _values(events(run.session, ANALYZE), "passes") == [12] * 10
    assert not events(run.session, "hfss.modeler.mesh_length")  # build_fine_mesh=False
    assert all(t.target_param_type != "kappa" for t in run.analysis.opt_targets)


def test_single_qubit_resonator_pratio(run_example):
    run = run_example("coupled_transmon__eigenmode_single_qubit_resonator_pratio")
    analyses = events(run.session, ANALYZE)
    assert _values(analyses, "passes") == [7]
    assert not events(run.session, "hfss.modeler.mesh_length")

    interfaces = ["substrate_air", "metal_substrate", "underside_air", "metal_air"]
    group_renames = [
        call["args"][0][1][2][1][2]
        for call in events(run.session, "hfss.oEditor.ChangeProperty")
        if call["args"][0][1][0] == "NAME:Attributes"
    ]
    assert group_renames == [
        "substrate_air",
        "metal_substrate",
        "metal_air",
        "underside_air",
    ]
    assert set(run.session.pinfo.dissipative["dielectric_surfaces"]) == {
        f"{interface}_object" for interface in interfaces
    }

    # every interface is queried for both modes, once by optimize_target and once
    # by the notebook's get_surface_p_ratio_df() cell
    surface_queries = [
        (call["name"], call["mode"]) for call in events(run.session, "epr.get_Qsurface")
    ]
    assert surface_queries == 2 * [
        (f"{interface}_object", mode) for interface in interfaces for mode in (0, 1)
    ]


def test_two_qubit_resonator(run_example):
    run = run_example("coupled_transmon__eigenmode_two_qubit_resonator")
    analyses = events(run.session, ANALYZE)
    # notebook increases the passes successively from 8 to 15
    assert _values(analyses, "passes") == [8, 9, 10, 11, 12, 13, 14, 15, 15, 15]
    assert set(_values(analyses, "delta_f")) == {0.001}
    assert set(_values(analyses, "n_modes")) == {5}
    assert sorted(analyses[0]["junctions"]) == ["jj_name_qubit_1_", "jj_name_qubit_2_"]
    assert set(_values(events(run.session, EPR), "cos_trunc")) == {6}
    assert set(_values(events(run.session, EPR), "fock_trunc")) == {5}
    assert len(events(run.session, "hfss.renderer.render_design")[0]["port_list"]) == 4


def test_capacitance_target_resonator(run_example):
    run = run_example("coupled_transmon__capacitance_target_resonator")
    runs = events(run.session, Q3D_RUN)
    # update_nbr_passes_capacitance_ministudies(10) overrides the ministudy's 8 passes
    assert _values(runs, "max_passes") == [10] * 10
    assert set(_values(runs, "percent_error")) == {0.5}
    assert runs[0]["components"] == ["name_resonator_1_", "name_tee_1_"]
    assert runs[0]["renderer_options"] == {
        "x_buffer_width_mm": 2,
        "y_buffer_width_mm": 2,
    }


def test_charge_line_decay_single_qubit(run_example):
    run = run_example("coupled_transmon__charge_line_decay_single_qubit")
    runs = events(run.session, Q3D_RUN)
    assert _values(runs, "max_passes") == [10] * 10
    assert runs[0]["components"] == ["name_qubit_1_", "name_charge_line_1_"]
    assert runs[0]["open_terminations"] == [
        ["name_qubit_1_", "readout"],
        ["name_qubit_1_", "coupler"],
        ["name_charge_line_1_", "start"],
        ["name_charge_line_1_", "end"],
    ]
    renders = events(run.session, "render_qiskit_metal")
    assert len(renders) == 10


def test_kappa_target_resonator(run_example):
    run = run_example("coupled_transmon__kappa_target_resonator")
    runs = events(run.session, Q3D_RUN)
    assert _values(runs, "max_passes") == [8] * 10
    # render kwargs of the ResonatorDecayIntoWaveguideStudy reach the render function
    assert (
        events(run.session, "render_qiskit_metal")
        == [{"kwargs": {"capacitance_or_surface_p_ratio": True}}] * 10
    )


# --------------------------------------------- examples_coupled_transmons_tunable_coupler_chip
def test_tunable_coupler_two_qubit_resonator(run_example):
    run = run_example("tunable_coupler__eigenmode_two_qubit_resonator")
    analyses = events(run.session, ANALYZE)
    assert _values(analyses, "passes") == [17] * 10
    assert set(_values(analyses, "n_modes")) == {5}
    assert sorted(analyses[0]["junctions"]) == [
        "jj_name_coupler_1to2_",
        "jj_name_qubit_1_",
        "jj_name_qubit_2_",
    ]
    assert events(run.session, "hfss.renderer.render_design")[0]["port_list"] == []


def test_tunable_coupler_model_inaccuracy(run_example):
    run = run_example("tunable_coupler__eigenmode_two_qubit_resonator_model_inaccuracy")
    analyses = events(run.session, ANALYZE)
    assert _values(analyses, "passes") == [12] * 10
    assert set(_values(analyses, "n_modes")) == {3}


def test_tunable_coupler_model_inaccuracy_qb_res(run_example):
    run = run_example(
        "tunable_coupler__eigenmode_two_qubit_resonator_model_inaccuracy_qb_res"
    )
    analyses = events(run.session, ANALYZE)
    assert _values(analyses, "passes") == [12] * 10
    assert set(_values(analyses, "n_modes")) == {2}


def test_tunable_coupler_partitioned(run_example):
    run = run_example("tunable_coupler__eigenmode_two_qubit_resonator_partitioned")
    analyses = events(run.session, ANALYZE)
    # 11 iterations x 3 partitions, each with 3 modes
    assert len(analyses) == 33
    assert set(_values(analyses, "passes")) == {15}
    assert set(_values(analyses, "n_modes")) == {3}
    selections = [
        tuple(r["selection"])
        for r in events(run.session, "hfss.renderer.render_design")
    ]
    assert selections[:3] == [
        ("name_qubit_1_", "name_resonator_1_", "name_tee_1_", "name_coupler_1to2_"),
        ("name_qubit_2_", "name_resonator_2_", "name_tee_2_", "name_coupler_1to2_"),
        ("name_qubit_1_", "name_qubit_2_", "name_coupler_1to2_"),
    ]
    assert len(run.optimization_results) == 11


@pytest.mark.parametrize(
    "name",
    [
        case.name
        for case in EXAMPLE_CASES
        if case.name != "tunable_coupler__eigenmode_two_qubit_resonator_partitioned"
    ],
)
def test_reported_parameters_equal_simulated_ground_truth(name, run_example):
    """Every targeted parameter must be what the simulator produced for that design.

    Catches unit conversions, mode/junction index mix-ups and wrong decay formulas,
    since the fake computes its values independently of the library.
    """
    run = run_example(name)
    physics = run.session.physics
    for iteration, result in enumerate(run.optimization_results):
        for parameter in physics.targets:
            reported = result["system_optimized_params"][parameter]
            expected = physics.value(parameter, result["design_variables"])
            assert reported == pytest.approx(expected, rel=1e-9), (iteration, parameter)
