"""DesignAnalysis features that the tutorial notebooks do not exercise.

These tests build the coupled-transmon tutorial chip for real and drive
``DesignAnalysis`` directly with non-default settings, against the same faked
simulators as the notebook tests.
"""

import importlib
import inspect
import json
from dataclasses import asdict
from unittest.mock import MagicMock

import pytest
from pyEPR._config_default import config as pyepr_config
from qiskit_metal.qlibrary.couplers.coupled_line_tee import CoupledLineTee

from qdesignoptimizer.design_analysis import DesignAnalysis
from qdesignoptimizer.design_analysis_types import (
    DesignAnalysisState,
    InterfaceProperties,
    Interfaces,
    MeshingMap,
    MiniStudy,
    SurfaceProperties,
)
from qdesignoptimizer.sim_capacitance_matrix import CapacitanceMatrixStudy
from qdesignoptimizer.utils.chip_generation import create_chip_base
from qdesignoptimizer.utils.names_design_variables import junction_setup
from qdesignoptimizer.utils.names_parameters import (
    CHARGE_LINE_LIMITED_T1,
    param,
    param_participation_ratio,
)
from qdesignoptimizer.utils.optimization_targets import (
    get_opt_target_res_freq_via_length,
)

from .conftest import COUPLED_TRANSMON
from .example_harness import events, prepare_example
from .fake_ansys import parse_value_with_unit

ANALYZE = "hfss.eigenmode.analyze"


class Chip:
    """The coupled-transmon tutorial chip with its helper modules loaded."""

    def __init__(self, session, workdir):
        self.session = session
        self.workdir = workdir
        modules = (
            "names",
            "design",
            "mini_studies",
            "optimization_targets",
            "parameter_targets",
        )
        loaded = {name: importlib.import_module(name) for name in modules}
        self.n = loaded["names"]
        self.d = loaded["design"]
        self.ms = loaded["mini_studies"]
        self.ot = loaded["optimization_targets"]
        self.pt = loaded["parameter_targets"]

    def analysis(self, mini_study, opt_targets, render_kwargs=None, **kwargs):
        design, gui = create_chip_base(
            self.n.CHIP_NAME, self.d.chip_type, open_gui=True
        )
        self.d.render_qiskit_metal_design(design, gui)
        fixed_kwargs = render_kwargs or {}

        def render(design, **kw):
            self.d.render_qiskit_metal_design(design, gui, **{**kw, **fixed_kwargs})

        state = DesignAnalysisState(design, render, self.pt.PARAM_TARGETS)
        kwargs.setdefault("save_path", str(self.workdir / "out" / "run"))
        kwargs.setdefault("update_design_variables", False)
        return DesignAnalysis(
            state, mini_study=mini_study, opt_targets=opt_targets, **kwargs
        )

    def qb_res_targets(self):
        return self.ot.get_opt_targets_2qubits_resonator_coupler(
            groups=[1],
            opt_target_qubit_freq=True,
            opt_target_qubit_anharm=True,
            opt_target_resonator_freq=True,
            opt_target_resonator_kappa=True,
            opt_target_resonator_qubit_chi=True,
        )

    def qb_res_mini_study(self, **overrides):
        mini_study = self.ms.get_mini_study_qb_res(group=1)
        mini_study.build_fine_mesh = False
        for key, value in overrides.items():
            setattr(mini_study, key, value)
        return mini_study


@pytest.fixture
def chip(tmp_path, monkeypatch):
    session = prepare_example(COUPLED_TRANSMON, tmp_path, monkeypatch)
    return Chip(session, tmp_path)


def run(analysis, iterations):
    for _ in range(iterations):
        analysis.optimize_target({}, {})
    return analysis.optimization_results


def value_of(design_variable: str) -> float:
    return parse_value_with_unit(design_variable)[0]


def assert_params_close(actual: dict, expected: dict):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if value is None:
            assert actual[key] is None, key
        else:
            assert actual[key] == pytest.approx(value, rel=1e-9), key


# ---------------------------------------------------------------------------- modes
def test_mode_order_in_mini_study_does_not_change_results(chip):
    """Results are mapped to modes by target frequency, not by MiniStudy.modes order."""
    results = {}
    for order in ("ascending", "descending"):
        chip.session.reset()
        mini_study = chip.qb_res_mini_study()
        if order == "descending":
            mini_study.modes = list(reversed(mini_study.modes))
        analysis = chip.analysis(mini_study, chip.qb_res_targets())
        results[order] = run(analysis, 3)[-1]["system_optimized_params"]

    assert_params_close(results["descending"], results["ascending"])
    n = chip.n
    assert results["descending"][param(n.QUBIT_1, n.FREQ)] == pytest.approx(
        4e9, rel=1e-3
    )
    assert results["descending"][param(n.RESONATOR_1, n.FREQ)] == pytest.approx(
        6e9, rel=1e-3
    )


def test_participation_ratios_are_assigned_per_mode_and_junction(chip):
    n = chip.n
    analysis = chip.analysis(chip.qb_res_mini_study(), chip.qb_res_targets())
    run(analysis, 1)
    params = analysis.system_optimized_params
    # fake: p(mode i, junction j) = 0.1 (i + 1) + 0.013 (j + 1)^2, modes sorted by frequency
    jj = "jj_name_qubit_1_"
    assert params[param_participation_ratio(n.QUBIT_1, jj)] == pytest.approx(0.113)
    assert params[param_participation_ratio(n.RESONATOR_1, jj)] == pytest.approx(0.213)


def test_linear_junctions_are_rendered_but_excluded_from_epr(chip):
    n = chip.n
    jj_setup = {**junction_setup(n.QUBIT_1), **junction_setup(n.QUBIT_2, "linear")}
    analysis = chip.analysis(
        chip.qb_res_mini_study(jj_setup=jj_setup), chip.qb_res_targets()
    )
    run(analysis, 1)

    assert sorted(events(chip.session, ANALYZE)[0]["junctions"]) == [
        "jj_name_qubit_1_",
        "jj_name_qubit_2_",
    ]
    assert list(events(chip.session, "epr.do_EPR_analysis")[0]["junctions"]) == [
        "jj_name_qubit_1_"
    ]
    assert not any(
        "jj_name_qubit_2_" in key for key in analysis.system_optimized_params
    )
    # the full junction setup is restored after the EPR analysis
    assert analysis.eig_solver.setup.junctions == jj_setup


# ------------------------------------------------------------------- settings plumbing
def test_non_default_mini_study_settings_reach_hfss_and_pyepr(chip):
    n, d = chip.n, chip.d
    tee = n.name_tee(1)
    mini_study = MiniStudy(
        qiskit_component_names=[
            n.name_mode(n.QUBIT_1),
            n.name_mode(n.RESONATOR_1),
            tee,
        ],
        port_list=[(tee, "prime_end", 50), (tee, "prime_start", 25)],
        open_pins=[(n.name_mode(n.QUBIT_1), "coupler")],
        modes=[n.QUBIT_1, n.RESONATOR_1],
        jj_setup=junction_setup(n.QUBIT_1),
        design_name="custom_study",
        nbr_passes=5,
        delta_f=0.02,
        x_buffer_width_mm=0.7,
        y_buffer_width_mm=0.9,
        max_mesh_length_port="7um",
        max_mesh_length_lines_to_ports="4um",
        hfss_wire_bond_size=5,
        hfss_wire_bond_offset="1um",
        hfss_wire_bond_threshold="200um",
        build_fine_mesh=True,
        cos_trunc=9,
        fock_trunc=4,
        render_qiskit_metal_eigenmode_kw_args={"capacitance_or_surface_p_ratio": True},
    )
    analysis = chip.analysis(
        mini_study,
        chip.qb_res_targets(),
        meshing_map=[MeshingMap(CoupledLineTee, d.CoupledLineTee_mesh_names)],
        minimization_tol=1e-9,
    )
    run(analysis, 2)
    session = chip.session

    assert analysis.anmod_optimizer.minimization_tol == 1e-9

    assert events(session, "hfss.renderer.activate_ansys_design") == [
        {"design_name": "custom_study", "solution_type": "eigenmode"}
    ]
    for analyze in events(session, ANALYZE):
        assert (analyze["passes"], analyze["delta_f"], analyze["n_modes"]) == (
            5,
            0.02,
            2,
        )
        assert analyze["setup_name"] == "Resonator_setup"
        assert analyze["sweep_variable"] == "dummy"
        assert analyze["renderer_options"] == {
            "x_buffer_width_mm": 0.7,
            "y_buffer_width_mm": 0.9,
            "max_mesh_length_port": "7um",
            "wb_size": 5,
            "wb_threshold": "200um",
            "wb_offset": "1um",
        }
    for render in events(session, "hfss.renderer.render_design"):
        assert render["port_list"] == [[tee, "prime_end", 50], [tee, "prime_start", 25]]
        assert render["open_pins"] == [[n.name_mode(n.QUBIT_1), "coupler"]]
        assert render["wire_bonds"] == {
            "wb_size": 5,
            "wb_threshold": "200um",
            "wb_offset": "1um",
        }
    assert (
        events(session, "render_qiskit_metal")
        == [{"kwargs": {"capacitance_or_surface_p_ratio": True}}] * 2
    )
    assert (
        events(session, "hfss.modeler.mesh_length")
        == [
            {
                "name": "fine_mesh",
                "objects": [f"prime_cpw_{tee}", f"second_cpw_{tee}"],
                "MaxLength": "4um",
                "RefineInside": True,
            }
        ]
        * 2
    )
    assert (
        events(session, "epr.analyze_all_variations")
        == [{"cos_trunc": 9, "fock_trunc": 4}] * 2
    )


def test_fine_mesh_requires_ports(chip):
    analysis = chip.analysis(
        chip.qb_res_mini_study(build_fine_mesh=True, port_list=[]),
        chip.qb_res_targets(),
        meshing_map=[MeshingMap(CoupledLineTee, chip.d.CoupledLineTee_mesh_names)],
    )
    run(analysis, 1)
    assert not events(chip.session, "hfss.modeler.mesh_length")


def test_non_default_capacitance_study_settings_reach_q3d(chip):
    n = chip.n
    components = [n.name_mode(n.RESONATOR_1), n.name_tee(1)]
    open_pins = [(n.name_mode(n.RESONATOR_1), "end"), (n.name_tee(1), "prime_end")]
    study = CapacitanceMatrixStudy(
        qiskit_component_names=components,
        open_pins=open_pins,
        x_buffer_width_mm=1.5,
        y_buffer_width_mm=2.5,
        percent_error=0.2,
        nbr_passes=6,
        render_qiskit_metal_kwargs={"capacitance_or_surface_p_ratio": True},
    )
    mini_study = MiniStudy(
        qiskit_component_names=components,
        port_list=[],
        open_pins=[],
        modes=[],
        capacitance_matrix_studies=[study],
    )
    analysis = chip.analysis(mini_study, chip.ot.get_opt_target_capacitance(group=1))
    run(analysis, 3)

    q3d_runs = events(chip.session, "q3d.run")
    assert len(q3d_runs) == 3
    for q3d_run in q3d_runs:
        assert q3d_run["components"] == components
        assert q3d_run["open_terminations"] == [list(pin) for pin in open_pins]
        assert (q3d_run["max_passes"], q3d_run["percent_error"]) == (6, 0.2)
        assert q3d_run["renderer_options"] == {
            "x_buffer_width_mm": 1.5,
            "y_buffer_width_mm": 2.5,
        }
    assert (
        events(chip.session, "render_qiskit_metal")
        == [{"kwargs": {"capacitance_or_surface_p_ratio": True}}] * 3
    )
    assert not events(chip.session, ANALYZE)


# ------------------------------------------------------------------------ optimizer
def test_adjustment_rate_moves_design_variables_part_of_the_way(chip):
    design_var = chip.n.design_var_length(chip.n.RESONATOR_1)
    proposed = {}
    for rate in (1.0, 0.5):
        chip.session.reset()
        analysis = chip.analysis(
            chip.qb_res_mini_study(adjustment_rate=rate), chip.qb_res_targets()
        )
        proposed[rate] = value_of(run(analysis, 2)[1]["design_variables"][design_var])

    initial = 7500.0
    assert proposed[1.0] != pytest.approx(initial)
    assert proposed[0.5] == pytest.approx(0.5 * proposed[1.0] + 0.5 * initial)


@pytest.mark.parametrize(
    "constraint, expected",
    [
        ({"larger_than": "5um", "smaller_than": "7600um"}, 7600.0),
        ({"larger_than": "8000um", "smaller_than": "15000um"}, 8000.0),
    ],
)
def test_design_variable_constraints_are_enforced(chip, constraint, expected):
    """The +5% frequency offset of the fake asks for a ~7875um resonator."""
    n = chip.n
    target = get_opt_target_res_freq_via_length(
        n.RESONATOR_1, design_var_constraint=constraint
    )
    analysis = chip.analysis(chip.qb_res_mini_study(), [target])
    results = run(analysis, 3)
    for result in results[1:]:
        assert value_of(result["design_variables"][target.design_var]) == pytest.approx(
            expected, rel=1e-9
        )


def test_resuming_from_stored_state_reproduces_the_next_iteration(chip):
    """optimize_target(design_vars, params) must continue exactly where a run left off."""
    first = chip.analysis(chip.qb_res_mini_study(), chip.qb_res_targets())
    results = run(first, 3)

    resumed = chip.analysis(chip.qb_res_mini_study(), chip.qb_res_targets())
    resumed.optimize_target(
        dict(results[1]["design_variables"]),
        dict(results[1]["system_optimized_params"]),
    )
    resumed_result = resumed.optimization_results[0]

    for target in first.opt_targets:
        assert value_of(resumed_result["design_variables"][target.design_var]) == (
            pytest.approx(value_of(results[2]["design_variables"][target.design_var]))
        )
    assert_params_close(
        resumed_result["system_optimized_params"], results[2]["system_optimized_params"]
    )
    assert resumed.state.system_optimized_params is resumed.system_optimized_params


# ----------------------------------------------------------- surface participation
def test_surface_properties_are_mapped_per_interface(chip):
    interfaces = Interfaces(
        substrate_air=InterfaceProperties(eps_r=10.0, th=2e-6, tan_delta_surf=1e-3),
        metal_substrate=InterfaceProperties(eps_r=11.0, th=3e-6, tan_delta_surf=2e-3),
        underside_air=InterfaceProperties(eps_r=12.0, th=4e-6, tan_delta_surf=3e-3),
        metal_air=InterfaceProperties(eps_r=13.0, th=5e-6, tan_delta_surf=4e-3),
    )
    surface_properties = SurfaceProperties(
        interfaces=interfaces, sheet_material="Niobium", sheet_thickness=0.0002
    )
    mini_study = chip.ms.get_mini_study_qb_res(
        group=1, surface_properties=surface_properties
    )
    mini_study.build_fine_mesh = False
    analysis = chip.analysis(
        mini_study,
        chip.qb_res_targets(),
        render_kwargs={"capacitance_or_surface_p_ratio": True},
    )
    run(analysis, 1)
    session = chip.session

    expected_surfaces = {
        f"{name}_object": asdict(getattr(interfaces, name))
        for name in interfaces.keys()
    }
    assert events(session, ANALYZE)[0]["dissipative"] == {
        "dielectric_surfaces": expected_surfaces
    }

    oeditor = {
        event: events(session, f"hfss.oEditor.{event}")
        for event in ("ThickenSheet", "ChangeProperty")
    }
    assert oeditor["ThickenSheet"][0]["args"][1] == [
        "NAME:SheetThickenParameters",
        "Thickness:=",
        "0.0002mm",
        "BothSides:=",
        True,
    ]
    material_changes = [
        call["args"][0][1]
        for call in oeditor["ChangeProperty"]
        if call["args"][0][1][0] == "NAME:Geometry3DAttributeTab"
    ]
    assert material_changes == [
        [
            "NAME:Geometry3DAttributeTab",
            ["NAME:PropServers", "metal_air_object"],
            ["NAME:ChangedProps", ["NAME:Material", "Value:=", '"Niobium"']],
        ]
    ]
    assert events(session, "hfss.modeler.translate")[0] == {
        "name": "pad_top_metal",
        "vector": ["0mm", "0mm", "0.0001mm"],
    }

    # fake: Q_surface = (1 + mode)(1 + len(name)) / (eps_r * th * tan_delta)
    p_ratio = analysis.surface_p_ratio
    for name in interfaces.keys():
        props = getattr(interfaces, name)
        for mode in (0, 1):
            q_surface = (
                (1 + mode)
                * (1 + len(f"{name}_object"))
                / (props.eps_r * props.th * props.tan_delta_surf)
            )
            assert p_ratio[name][mode] == pytest.approx(1 / q_surface)
    for mode in (0, 1):
        assert p_ratio["dielectric"][mode] == pytest.approx(
            1 / (2e6 * (1 + mode) * pyepr_config.dissipation.tan_delta_sapp)
        )
        assert p_ratio["Junction(inductive energy)"][mode] == pytest.approx(
            0.9 / (1 + mode)
        )

    p_ratio_df = analysis.get_surface_p_ratio_df()
    assert list(p_ratio_df.index) == list(interfaces.keys()) + [
        "Junction(inductive energy)",
        "dielectric",
    ]
    assert list(p_ratio_df.columns) == ["mode 0", "mode 1"]
    assert p_ratio_df.index.name == "contribution"


# ------------------------------------------------------------------ decay studies
def test_grounded_charge_line_decay_study(chip):
    n = chip.n
    mini_study = chip.ms.get_mini_study_qb_charge_line(group=1)
    study = mini_study.capacitance_matrix_studies[0]
    study.mode_capacitance_name = "pad_top_" + n.name_mode(n.QUBIT_1)  # grounded island
    analysis = chip.analysis(
        mini_study, chip.ot.get_opt_targets_qb_charge_line(group=1)
    )
    results = run(analysis, 4)

    t1 = param(n.QUBIT_1, CHARGE_LINE_LIMITED_T1)
    for result in results:
        # the library's T1 must equal the fake's independently computed ground truth
        assert result["system_optimized_params"][t1] == pytest.approx(
            chip.session.physics.value(t1, result["design_variables"]), rel=1e-9
        )
    assert results[-1]["system_optimized_params"][t1] == pytest.approx(20e-3, rel=1e-4)


# ----------------------------------------------------------- defaults & side effects
def test_library_defaults_reach_the_simulators(chip):
    """Pin the default settings of MiniStudy, CapacitanceMatrixStudy and DesignAnalysis."""
    n = chip.n
    tee = n.name_tee(1)
    mini_study = MiniStudy(
        qiskit_component_names=[
            n.name_mode(n.QUBIT_1),
            n.name_mode(n.RESONATOR_1),
            tee,
        ],
        port_list=[(tee, "prime_end", 50)],
        open_pins=[],
        modes=[n.QUBIT_1, n.RESONATOR_1],
        jj_setup=junction_setup(n.QUBIT_1),
    )
    analysis = chip.analysis(
        mini_study,
        chip.qb_res_targets(),
        meshing_map=[MeshingMap(CoupledLineTee, chip.d.CoupledLineTee_mesh_names)],
        save_path="analysis_result",
        update_design_variables=False,
    )
    run(analysis, 1)
    session = chip.session

    assert analysis.minimization_tol == 1e-12
    assert analysis.anmod_optimizer.minimization_tol == 1e-12
    assert analysis.anmod_optimizer.adjustment_rate == 1.0
    assert (
        events(session, "hfss.renderer.activate_ansys_design")[0]["design_name"]
        == "mini_study"
    )
    analyze = events(session, ANALYZE)[0]
    assert (analyze["passes"], analyze["delta_f"]) == (10, 0.1)
    assert analyze["renderer_options"] == {
        "x_buffer_width_mm": 0.5,
        "y_buffer_width_mm": 0.5,
        "max_mesh_length_port": "3um",
        "wb_size": 3,
        "wb_threshold": "300um",
        "wb_offset": "0um",
    }
    assert events(session, "epr.analyze_all_variations") == [
        {"cos_trunc": 8, "fock_trunc": 7}
    ]
    assert not events(session, "hfss.modeler.mesh_length")  # build_fine_mesh=False
    assert (chip.workdir / "analysis_result.npy").exists()

    session.reset()  # new chip model for the capacitance targets
    study = CapacitanceMatrixStudy(
        qiskit_component_names=[n.name_mode(n.RESONATOR_1), tee]
    )
    cap_mini_study = MiniStudy(
        qiskit_component_names=study.qiskit_component_names,
        port_list=[],
        open_pins=[],
        modes=[],
        capacitance_matrix_studies=[study],
    )
    run(chip.analysis(cap_mini_study, chip.ot.get_opt_target_capacitance(group=1)), 1)
    q3d_run = events(session, "q3d.run")[0]
    assert (q3d_run["max_passes"], q3d_run["percent_error"]) == (10, 0.5)
    assert q3d_run["renderer_options"] == {
        "x_buffer_width_mm": 2,
        "y_buffer_width_mm": 2,
    }
    assert q3d_run["open_terminations"] == []


def test_default_surface_properties(chip):
    mini_study = chip.ms.get_mini_study_qb_res(
        group=1, surface_properties=SurfaceProperties()
    )
    mini_study.build_fine_mesh = False
    analysis = chip.analysis(
        mini_study,
        chip.qb_res_targets(),
        render_kwargs={"capacitance_or_surface_p_ratio": True},
    )
    run(analysis, 1)
    session = chip.session
    surfaces = events(session, ANALYZE)[0]["dissipative"]["dielectric_surfaces"]
    assert set(surfaces) == {
        "substrate_air_object",
        "metal_substrate_object",
        "underside_air_object",
        "metal_air_object",
    }
    for props in surfaces.values():
        assert props == {"eps_r": 11.4, "th": 1e-6, "tan_delta_surf": 1.0}
    assert asdict(InterfaceProperties()) == {
        "eps_r": 11.4,
        "th": 1e-6,
        "tan_delta_surf": 1.0,
    }
    thicken = events(session, "hfss.oEditor.ThickenSheet")[0]["args"][1]
    assert thicken[2] == "0.00015mm"
    material = [
        call["args"][0][1][2][1][2]
        for call in events(session, "hfss.oEditor.ChangeProperty")
        if call["args"][0][1][0] == "NAME:Geometry3DAttributeTab"
    ]
    assert material == ['"Aluminum"']


def test_update_design_variables_rewrites_design_variables_json(chip):
    """The default update_design_variables=True overwrites design_variables.json each iteration."""
    design_variables_file = chip.workdir / "design_variables.json"
    analysis = chip.analysis(
        chip.qb_res_mini_study(),
        chip.qb_res_targets(),
        update_design_variables=True,
    )
    for _ in range(3):
        analysis.optimize_target({}, {})
        written = json.loads(design_variables_file.read_text())
        latest = analysis.optimization_results[-1]["design_variables"]
        for target in analysis.opt_targets:
            assert written[target.design_var] == latest[target.design_var]
    initial = json.loads((COUPLED_TRANSMON / "design_variables.json").read_text())
    assert written.keys() == initial.keys()
    assert written != initial


def test_update_design_variables_is_the_default():
    signature = inspect.signature(DesignAnalysis.__init__)
    assert signature.parameters["update_design_variables"].default is True
    assert signature.parameters["save_path"].default == "analysis_result"
    assert signature.parameters["minimization_tol"].default == 1e-12


def test_progress_figures_are_saved(chip):
    plot_settings = importlib.import_module("plot_settings")
    analysis = chip.analysis(
        chip.qb_res_mini_study(),
        chip.qb_res_targets(),
        plot_settings=plot_settings.PLOT_SETTINGS,
    )
    analysis.optimize_target({}, {}, save_figures=True)
    analysis.optimize_target({}, {}, save_figures=True)
    saved = sorted(path.name for path in (chip.workdir / "out").glob("*.png"))
    assert saved == sorted(
        f"run_optimization_plot_{name}.png" for name in plot_settings.PLOT_SETTINGS
    )


def test_screenshot_names_follow_the_run_number(chip):
    analysis = chip.analysis(chip.qb_res_mini_study(), chip.qb_res_targets())
    gui = MagicMock()
    analysis.screenshot(gui, run=0)
    analysis.screenshot(gui)
    names = [call.kwargs["name"] for call in gui.screenshot.call_args_list]
    assert names == [analysis.save_path + "_1", analysis.save_path]

    gui.screenshot.side_effect = RuntimeError("no display")
    analysis.screenshot(gui, run=1)  # failures are logged, not raised


# ------------------------------------------------------------------- validation
def test_unknown_design_variable_is_rejected(chip):
    target = get_opt_target_res_freq_via_length(chip.n.RESONATOR_1)
    target.design_var = "design_var_does_not_exist"
    with pytest.raises(AssertionError, match="not found in design variables"):
        chip.analysis(chip.qb_res_mini_study(), [target])


def test_target_for_mode_that_is_not_simulated_is_rejected(chip):
    targets = chip.ot.get_opt_targets_2qubits_resonator_coupler(
        groups=[2], opt_target_resonator_freq=True
    )
    with pytest.raises(AssertionError, match="not found in modes"):
        chip.analysis(chip.qb_res_mini_study(), targets)
    # allowed when the analysis is one partition of a larger system
    chip.analysis(
        chip.qb_res_mini_study(), targets, is_part_of_partitioned_optimization=True
    )


def test_charge_line_target_requires_capacitance_study(chip):
    with pytest.raises(AssertionError, match="capacitance_matrix_studies"):
        chip.analysis(
            chip.qb_res_mini_study(), chip.ot.get_opt_targets_qb_charge_line(group=1)
        )


def test_capacitance_target_requires_target_value(chip):
    targets = chip.ot.get_opt_target_capacitance(group=2)  # no target value for group 2
    with pytest.raises(AssertionError, match="requires"):
        chip.analysis(chip.ms.get_mini_study_resonator_capacitance(group=2), targets)


def test_duplicate_design_variables_are_rejected(chip):
    targets = chip.qb_res_targets()
    duplicate = get_opt_target_res_freq_via_length(chip.n.RESONATOR_1)
    with pytest.raises(ValueError, match="unique design_var"):
        chip.analysis(chip.qb_res_mini_study(), targets + [duplicate])
