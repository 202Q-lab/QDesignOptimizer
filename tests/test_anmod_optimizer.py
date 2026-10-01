import pytest

from qdesignoptimizer.anmod_optimizer import ANModOptimizer
from qdesignoptimizer.design_analysis_types import OptTarget
from qdesignoptimizer.utils.names_parameters import NONLIN


def _toy_target(mode_name: str, design_var: str, group: str) -> OptTarget:
    return OptTarget(
        target_param_type=NONLIN,
        involved_modes=[mode_name, mode_name],
        design_var=design_var,
        design_var_constraint={"larger_than": "0", "smaller_than": "20"},
        prop_to=lambda p, v, dv=design_var: v[dv],
        independent_target=group,
    )


def test_unique_design_vars_accepted():
    ANModOptimizer(
        [_toy_target("modeA", "x", "group"), _toy_target("modeB", "y", "group")],
        {},
    )


@pytest.mark.parametrize("group_b", ["group", "other_group"])
def test_duplicate_design_var_raises(group_b):
    with pytest.raises(ValueError, match="unique design_var"):
        ANModOptimizer(
            [_toy_target("modeA", "x", "group"), _toy_target("modeB", "x", group_b)],
            {},
        )
