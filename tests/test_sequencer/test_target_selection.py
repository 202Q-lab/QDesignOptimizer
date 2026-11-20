"""What a study file may name, and which opt targets a study switches on.

Second review of the port (2026-11-12): a ``Class.method`` reference got
past the allowlist; a MiniStudy chaining a capacitance study under an
eigenmode solve switched on targets it never measures, and two that tune
one variable; capacitance targets were taken by no study at all.
"""

import pytest

from qdesignoptimizer.design_analysis_types import MiniStudy
from qdesignoptimizer.sequencer.backends import qdo_files as qf
from qdesignoptimizer.sequencer.backends.qdo import study_targets
from qdesignoptimizer.sim_capacitance_matrix import (
    CapacitanceMatrixStudy,
    ResonatorDecayIntoWaveguideStudy,
)

_OT = "qdesignoptimizer.utils.optimization_targets"


@pytest.mark.parametrize(
    "entry",
    [
        # A method of a class in an allowed package runs anything.
        {
            "helper": "qiskit_metal.renderers.renderer_ansys.ansys_renderer"
            ":QAnsysRenderer.open_ansys",
            "args": {"self": None, "path": "C:/Windows/System32"},
        },
        # QDO functions that are no opt-target helpers.
        {"helper": "qdesignoptimizer.utils.utils:close_ansys", "args": {}},
        {"helper": "qdesignoptimizer.utils.plotting:plot_optimization_results"},
        # A name the helper module merely imports.
        {"helper": f"{_OT}:OptTarget", "args": {}},
    ],
)
def test_a_helper_entry_names_only_qdo_opt_target_helpers(entry):
    with pytest.raises(qf.CodecError):
        qf.data_to_target(entry)


def test_class_and_ref_entries_are_limited_to_their_modules():
    with pytest.raises(qf.CodecError, match="may not name"):
        qf.load_object(
            {"class": "qiskit_metal.designs.design_planar:DesignPlanar", "args": {}}
        )
    with pytest.raises(qf.CodecError, match="may not name"):
        qf.decode({"ref": "qdesignoptimizer.utils.utils:close_ansys"})
    with pytest.raises(qf.CodecError, match="top-level"):
        qf.decode({"ref": "qdesignoptimizer.utils.names_parameters:x.y"})
    fn = qf.decode(
        {"ref": "qdesignoptimizer.utils.names_design_variables:design_var_lj"}
    )
    assert fn("qubit_1") == "design_var_lj_qubit_1"


def _kappa_study(components=("name_resonator_1_", "name_tee_1_")):
    return ResonatorDecayIntoWaveguideStudy(
        mode="resonator_1",
        mode_freq_GHz=6.0,
        resonator_name="second_cpw_name_tee_1_",
        waveguide_name="prime_cpw_name_tee_1_",
        impedance_ohm=50,
        qiskit_component_names=list(components),
        open_pins=[],
        resonator_type="lambda_4",
    )


def _entries():
    kappa = {
        "helper": f"{_OT}:get_opt_target_res_kappa_via_coupl_length",
        "args": {"resonator": "resonator_1", "resonator_coupled_identifier": "tee"},
    }
    return [
        {
            "name": "freq",
            "helper": f"{_OT}:get_opt_target_res_freq_via_length",
            "args": {"resonator": "resonator_1"},
        },
        {"name": "kappa", **kappa},
        {"name": "kappa_eigenmode", "simulation": "eigenmode", **kappa},
        {
            "name": "t1",
            "target_param_type": "charge_line_limited_t1",
            "involved_modes": ["resonator_1"],
            "design_var": "design_var_cl_pos_x_qubit_1",
            "design_var_constraint": {},
            "prop_to": "v['design_var_cl_pos_x_qubit_1']",
        },
        {
            "name": "tee_capacitance",
            "target_param_type": "capacitance",
            "involved_modes": ["prime_cpw_name_tee_1_", "second_cpw_name_tee_1_"],
            "design_var": "design_var_length_resonator_1_capacitance",
            "design_var_constraint": {},
            "prop_to": "1",
        },
    ]


def _switched(study) -> dict:
    return {e["name"]: qf.is_active(e) for e in study_targets(study, _entries())}


def test_a_mixed_study_switches_on_only_what_it_measures_once():
    """Eigenmode solve plus a chained linewidth study: the frequency from
    the eigenmode, the linewidth from the capacitance study (QDO lets it
    overwrite), no charge-line T1 (nothing measures it), the tee
    capacitance (its conductors are in the chained study's matrix)."""
    study = MiniStudy(
        qiskit_component_names=["name_resonator_1_", "name_tee_1_"],
        port_list=[],
        open_pins=[],
        modes=["resonator_1"],
        jj_setup={},
        capacitance_matrix_studies=[_kappa_study()],
    )
    assert _switched(study) == {
        "freq": True,
        "kappa": True,
        "kappa_eigenmode": False,
        "t1": False,
        "tee_capacitance": True,
    }


def test_an_eigenmode_study_takes_its_kappa_from_the_ports():
    study = MiniStudy(
        qiskit_component_names=["name_resonator_1_", "name_tee_1_"],
        port_list=[["name_tee_1_", "prime_end", 50]],
        open_pins=[],
        modes=["resonator_1"],
        jj_setup={},
    )
    switched = _switched(study)
    assert switched["kappa_eigenmode"] and not switched["kappa"]
    assert "tee_capacitance" not in switched  # no capacitance matrix


def test_a_capacitance_target_is_taken_by_a_plain_capacitance_study():
    study = MiniStudy(
        qiskit_component_names=["name_resonator_1_", "name_tee_1_"],
        port_list=[],
        open_pins=[],
        modes=[],
        jj_setup={},
        capacitance_matrix_studies=[
            CapacitanceMatrixStudy(
                qiskit_component_names=["name_resonator_1_", "name_tee_1_"],
                open_pins=[],
            )
        ],
    )
    assert _switched(study) == {"tee_capacitance": True}
    other = MiniStudy(
        qiskit_component_names=["name_qubit_1_"],
        port_list=[],
        open_pins=[],
        modes=[],
        jj_setup={},
        capacitance_matrix_studies=[
            CapacitanceMatrixStudy(
                qiskit_component_names=["name_qubit_1_"], open_pins=[]
            )
        ],
    )
    assert _switched(other) == {}


_JJ = {
    "jj_name_qubit_1_": {
        "rect": "JJ_rect_Lj_name_qubit_1__rect_jj",
        "line": "JJ_Lj_name_qubit_1__rect_jj_",
        "Lj_variable": "design_var_lj_qubit_1",
        "Cj_variable": "design_var_cj_qubit_1",
    }
}
_CHI = {
    "name": "chi",
    "helper": f"{_OT}:get_opt_target_res_qub_chi_via_coupl_length",
    "args": {"qubit": "qubit_1", "resonator": "resonator_1"},
}


def _qb_res(jj_setup, ports):
    return MiniStudy(
        qiskit_component_names=["name_qubit_1_", "name_resonator_1_", "name_tee_1_"],
        port_list=[["name_tee_1_", "prime_end", 50]] if ports else [],
        open_pins=[],
        modes=["qubit_1", "resonator_1"],
        jj_setup=jj_setup,
    )


def test_nonlinearities_need_a_non_linear_junction_and_kappa_needs_ports():
    """Third review: QDO runs EPR only for a non-linear junction, and its
    eigenmode kappa (f/Q) is a coupling only through ports."""
    entries = _entries() + [_CHI]
    on = {
        e["name"]: qf.is_active(e) for e in study_targets(_qb_res(_JJ, True), entries)
    }
    assert on["chi"] and on["kappa_eigenmode"]
    on = {
        e["name"]: qf.is_active(e) for e in study_targets(_qb_res({}, False), entries)
    }
    assert not on["chi"] and not on["kappa_eigenmode"] and on["freq"]
    linear = {k: {**v, "junction_type": "linear"} for k, v in _JJ.items()}
    on = {
        e["name"]: qf.is_active(e)
        for e in study_targets(_qb_res(linear, True), entries)
    }
    assert not on["chi"]


def test_a_target_without_a_target_value_starts_switched_off():
    study = _qb_res(_JJ, True)
    params = {"resonator_1_freq": 6e9}  # no kappa, no chi
    on = {
        e["name"]: qf.is_active(e)
        for e in study_targets(study, _entries() + [_CHI], params)
    }
    assert on["freq"] and not on["kappa_eigenmode"] and not on["chi"]


def test_a_conductor_belongs_to_the_longest_component_it_ends_with():
    from qdesignoptimizer.sequencer.backends.qdo import measures_capacitance

    study = MiniStudy(
        qiskit_component_names=["res_1"],
        port_list=[],
        open_pins=[],
        modes=[],
        jj_setup={},
        capacitance_matrix_studies=[
            CapacitanceMatrixStudy(qiskit_component_names=["res_1"], open_pins=[])
        ],
    )
    conductors = ["pad_ext_res_1", "ground_main_plane"]
    assert measures_capacitance(study, conductors)  # nothing else known
    assert not measures_capacitance(study, conductors, {"res_1", "ext_res_1"})
    assert measures_capacitance(study, ["pad_res_1"], {"res_1", "ext_res_1"})


def test_an_eigenmode_mode_needs_a_target_frequency():
    from qdesignoptimizer.sequencer.backends.qdo import _mode_problems

    ministudy = {"args": {"modes": ["resonator_1"]}}
    known = {"resonator_1_kappa"}
    (problem,) = _mode_problems(ministudy, {}, {}, known, {})
    assert "resonator_1_freq" in problem
    capacitance_only = {
        "args": {"modes": ["resonator_1"], "run_capacitance_studies_only": True}
    }
    assert _mode_problems(capacitance_only, {}, {}, known, {}) == []


def test_a_value_written_as_a_ref_is_one_that_reads_back():
    with pytest.raises(qf.CodecError, match="may not name"):
        qf.encode(MiniStudy)  # a class: not a ref: entry
    from qdesignoptimizer.utils.names_design_variables import design_var_lj

    ref = qf.encode(design_var_lj)
    assert qf.decode(ref) is design_var_lj
