import names as n

from qdesignoptimizer.design_analysis_types import MiniStudy
from qdesignoptimizer.utils.names_design_variables import junction_setup


def get_mini_study_resonator_only(group: int):
    resonator = [n.RESONATOR_1, n.RESONATOR_2][group - 1]

    return MiniStudy(
        qiskit_component_names=[
            n.name_mode(resonator),
            n.name_tee(group),
        ],
        port_list=[
            (n.name_tee(group), "prime_end", 50),
            (n.name_tee(group), "prime_start", 50),
        ],
        open_pins=[],
        modes=[resonator],
        design_name="mini_study_resonator_only",
        jj_setup=None,
        build_fine_mesh=True,
        adjustment_rate=1,
    )


def get_mini_study_resonator_qubit(group: int):
    resonator = [n.RESONATOR_1, n.RESONATOR_2][group - 1]
    qubit = [n.QUBIT_1, n.QUBIT_2][group - 1]

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
        design_name="mini_study_qubit_resonator",
        jj_setup=junction_setup(qubit),
        build_fine_mesh=True,
        adjustment_rate=0.9,
    )
