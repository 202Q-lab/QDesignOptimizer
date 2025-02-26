import names as n
import numpy as np
import parameter_targets as pt

from qdesignoptimizer.design_analysis_types import MiniStudy
from qdesignoptimizer.sim_capacitance_matrix import ModeDecayIntoChargeLineStudy
from qdesignoptimizer.utils.names_design_variables import junction_setup

CONVERGENCE = dict(nbr_passes=7, delta_f=0.03)


def get_mini_study_qb_res(nbr: int):
    qubit = [n.QUBIT_1, n.QUBIT_2][nbr - 1]
    resonator = [n.RESONATOR_1, n.RESONATOR_2][nbr - 1]

    return MiniStudy(
        qiskit_component_names=[
            n.name_mode(qubit),
            n.name_mode(resonator),
            n.name_tee(nbr),
        ],
        port_list=[
            (n.name_tee(nbr), "prime_end", 50),
            (n.name_tee(nbr), "prime_start", 50),
        ],
        open_pins=[],
        modes=[qubit, resonator],
        jj_setup={**junction_setup(qubit)},
        design_name="get_mini_study_qb_res",
        adjustment_rate=1,
        build_fine_mesh=False,
        **CONVERGENCE
    )


def get_mini_study_2qb_resonator_coupler():
    all_comps = []
    all_ports = []
    all_modes = []
    all_jjs = {}
    for nbr in [n.NBR_1, n.NBR_2]:
        qubit = [n.QUBIT_1, n.QUBIT_2][nbr - 1]
        resonator = [n.RESONATOR_1, n.RESONATOR_2][nbr - 1]
        all_comps.extend([n.name_mode(qubit), n.name_mode(resonator), n.name_tee(nbr)])
        all_ports.extend(
            [
                (n.name_tee(nbr), "prime_end", 50),
                (n.name_tee(nbr), "prime_start", 50),
            ]
        )
        all_modes.extend([qubit, resonator])
        all_jjs.update(junction_setup(qubit))

    all_comps.append(n.name_mode(n.COUPLER_12))
    all_modes.append(n.COUPLER_12)

    all_mode_freq = []
    for i in range(len(all_modes)):
        all_mode_freq.append(pt.PARAM_TARGETS[all_modes[i][0]][all_modes[i][1]])
    all_modes_sorted = [all_modes[i] for i in np.argsort(all_mode_freq)]

    return MiniStudy(
        qiskit_component_names=all_comps,
        port_list=all_ports,
        open_pins=[],
        modes=all_modes_sorted,
        jj_setup=all_jjs,
        design_name="get_mini_study_2qb_resonator_coupler",
        adjustment_rate=1,
        **CONVERGENCE
    )

def get_mini_study_qb_charge_line(branch: int):
    qiskit_component_names = [
        u.name_qb(branch),
        u.name_charge_line(branch),
        u.name_otg_chargeline(branch),
    ]
    charge_decay_study = ModeDecayIntoChargeLineStudy(
        str(branch),
        dc.QUBIT_FREQ,
        open_pins=[
            (u.name_qb(branch), "readout"),
            (u.name_charge_line(branch), "start"),
        ],
        mode_capacitance_name=[
            "pad_bot_NAME_QB0",
            "pad_top_NAME_QB0",
        ],  # These names must be found from the model list in Ansys
        charge_line_capacitance_name="trace_NAME_CHARGE_LINE_0",
        charge_line_impedance_Ohm=50,
        qiskit_component_names=qiskit_component_names,
        freq_GHz=pt.PARAM_TARGETS[str(branch)][dc.QUBIT_FREQ]
        * 1e-9,  # not updated dynamically at the moment
        ground_plane_capacitance_name="ground_main_plane",
        nbr_passes=8,
    )
    return MiniStudy(
        qiskit_component_names=qiskit_component_names,
        port_list=[],
        open_pins=[],
        mode_freqs=[],  # No mode frequencies to run only capacitance studies and not eigenmode/epr
        jj_setup={**junction_setup(u.name_qb(branch))},
        design_name="get_mini_study_qb_charge_line",
        adjustment_rate=0.8,
        capacitance_matrix_studies=[charge_decay_study],
        **CONVERGENCE
    )
