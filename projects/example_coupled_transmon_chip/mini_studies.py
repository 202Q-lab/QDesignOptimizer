import numpy as np
import names as n
import parameter_targets as pt

from qdesignoptimizer.design_analysis_types import MiniStudy
from qdesignoptimizer.sim_capacitance_matrix import ModeDecayIntoChargeLineStudy
from qdesignoptimizer.utils.names_design_variables import junction_setup
from qdesignoptimizer.utils.names_parameters import (FREQ, param)

CONVERGENCE = dict(nbr_passes=7, delta_f=0.03)


def get_mini_study_qb_res(group: int):
    qubit = [n.QUBIT_1, n.QUBIT_2][group - 1]
    resonator = [n.RESONATOR_1, n.RESONATOR_2][group - 1]

    return MiniStudy(
        qiskit_component_names=[
            n.name_mode(qubit),
            n.name_mode(resonator),
            n.name_tee(group),
        ],
        port_list=[
            (n.name_tee(group), "prime_end", 50),
            (n.name_tee(group), "prime_start", 50),
        ],
        open_pins=[],
        modes=[qubit, resonator],
        jj_setup={**junction_setup(qubit)},
        design_name="get_mini_study_qb_res",
        adjustment_rate=1,
        build_fine_mesh=True,
        **CONVERGENCE
    )


def get_mini_study_2qb_resonator_coupler():
    all_comps = []
    all_ports = []
    all_modes = []
    all_jjs = {}
    for group in [n.GROUP_1, n.GROUP_2]:
        qubit = [n.QUBIT_1, n.QUBIT_2][group - 1]
        resonator = [n.RESONATOR_1, n.RESONATOR_2][group - 1]
        all_comps.extend([n.name_mode(qubit), n.name_mode(resonator), n.name_tee(group)])
        all_ports.extend(
            [
                (n.name_tee(group), "prime_end", 50),
                (n.name_tee(group), "prime_start", 50),
            ]
        )
        all_modes.extend([qubit, resonator])
        all_jjs.update(junction_setup(qubit))

    all_comps.append(n.name_mode(n.COUPLER_12))
    all_modes.append(n.COUPLER_12)

    all_mode_freq = []
    for i in range(len(all_modes)):
        all_mode_freq.append(pt.PARAM_TARGETS[param(all_modes[i], FREQ)])
    all_modes_sorted = [all_modes[i] for i in np.argsort(all_mode_freq)]

    return MiniStudy(
        qiskit_component_names=all_comps,
        port_list=all_ports,
        open_pins=[],
        modes=all_modes_sorted,
        jj_setup=all_jjs,
        design_name="get_mini_study_2qb_resonator_coupler",
        adjustment_rate=1,
        build_fine_mesh=False,
        **CONVERGENCE
    )

def get_mini_study_qb_charge_line(group: int):
    qubit = [n.QUBIT_1, n.QUBIT_2][group - 1]
    qiskit_component_names = [
        n.name_mode(qubit),
        n.name_charge_line(group),
    ]
    charge_decay_study = ModeDecayIntoChargeLineStudy(
        open_pins=[
            (n.name_mode(qubit), "readout"),
            (n.name_charge_line(group), "start"),
        ],
        mode_capacitance_name=[
            "pad_bot_NAME_QB0",
            "pad_top_NAME_QB0",
        ],  # These names must be found from the model list in Ansys
        charge_line_capacitance_name="trace_NAME_CHARGE_LINE_0",
        charge_line_impedance_Ohm=50,
        qiskit_component_names=qiskit_component_names,
        freq_GHz=pt.PARAM_TARGETS[param(n.QUBIT_1, FREQ)]* 1e-9,  # not updated dynamically at the moment
        ground_plane_capacitance_name="ground_main_plane",
        nbr_passes=8,
    )
    return MiniStudy(
        qiskit_component_names=qiskit_component_names,
        port_list=[],
        open_pins=[],
        modes=[],  # No mode frequencies to run only capacitance studies and not eigenmode/epr
        jj_setup={**junction_setup(n.name_mode(qubit))},
        design_name="get_mini_study_qb_charge_line",
        adjustment_rate=0.8,
        capacitance_matrix_studies=[charge_decay_study],
        **CONVERGENCE
    )
