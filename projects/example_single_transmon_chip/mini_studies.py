import design_variable_names as u
import parameter_targets as pt

import qdesignoptimizer.utils.constants as dc
from qdesignoptimizer.design_analysis_types import MiniStudy
from qdesignoptimizer.sim_capacitance_matrix import ModeDecayIntoChargeLineStudy
from qdesignoptimizer.utils.utils_design_variable_names import junction_setup

CONVERGENCE = dict(nbr_passes=7, delta_f=0.03)


def get_mini_study_qb_res(branch: int):
    return MiniStudy(
        qiskit_component_names=[
            u.name_qb(branch),
            u.name_res(branch),
            u.name_tee(branch),
        ],
        port_list=[
            (u.name_tee(branch), "prime_end", 50),
            (u.name_tee(branch), "prime_start", 50),
        ],
        open_pins=[],
        mode_freqs=[
            (str(branch), dc.mode_freq(dc.QUBIT)),
            (str(branch), dc.mode_freq(dc.RESONATOR)),
        ],
        jj_setup={**junction_setup(u.name_qb(branch))},
        hfss_wire_bond_size=2,
        hfss_wire_bond_offset="0um",
        hfss_wire_bond_threshold="300um",
        design_name="get_mini_study_qb_res",
        adjustment_rate=0.8,
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
