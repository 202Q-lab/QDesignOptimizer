import names as n

from qdesignoptimizer.design_analysis_types import OptTarget


def get_opt_target_res_freq_via_length(group: int) -> list[OptTarget]:

    resonator = [n.RESONATOR_1, n.RESONATOR_2][group - 1]
    opt_targets = [
        OptTarget(
            target_param_type=n.FREQ,
            involved_modes=[resonator],
            design_var=n.design_var_length(resonator),
            design_var_constraint={"larger_than": "500um", "smaller_than": "15000um"},
            prop_to=lambda p, v: 1 / v[n.design_var_length(resonator)],
            independent_target=True,
        )
    ]
    return opt_targets
