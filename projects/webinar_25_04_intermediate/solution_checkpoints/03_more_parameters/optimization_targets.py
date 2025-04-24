import names as n

from qdesignoptimizer.design_analysis_types import OptTarget
from qdesignoptimizer.utils.optimization_targets import (
    get_opt_target_qubit_anharmonicity_via_capacitance_width,
    get_opt_target_qubit_freq_via_lj,
    get_opt_target_res_freq_via_length,
    get_opt_target_res_qub_chi_via_coupl_length,
)


def get_opt_target_resonator_qubit(group: int) -> list[OptTarget]:

    resonator = [n.RESONATOR_1, n.RESONATOR_2][group - 1]
    qubit = [n.QUBIT_1, n.QUBIT_2][group - 1]
    opt_targets = [
        get_opt_target_qubit_freq_via_lj(
            qubit,
            design_var_qubit_lj=n.design_var_lj,
            design_var_qubit_width=n.design_var_width,
        ),
        get_opt_target_res_freq_via_length(resonator),
        get_opt_target_res_qub_chi_via_coupl_length(qubit, resonator),
        get_opt_target_qubit_anharmonicity_via_capacitance_width(qubit),
    ]
    return opt_targets
