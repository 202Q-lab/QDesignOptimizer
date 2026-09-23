import pytest

from qdesignoptimizer.anmod_optimizer import ANModOptimizer
from qdesignoptimizer.design_analysis_types import OptTarget
from qdesignoptimizer.utils.names_parameters import NONLIN, param_nonlin


def _toy_target(mode_name: str, prop_to_design_var: str, group: str) -> OptTarget:
    return OptTarget(
        target_param_type=NONLIN,
        involved_modes=[mode_name, mode_name],
        design_var=prop_to_design_var,
        design_var_constraint={"larger_than": "0", "smaller_than": "20"},
        prop_to=lambda p, v, dv=prop_to_design_var: v[dv],
        independent_target=group,
    )


class TestSharedDesignVar:
    """Two DIFFERENT target quantities sharing the SAME design_var, within
    one independent_target group, should be jointly (least-squares) fit
    against that one shared coordinate -- not silently collapse to whichever
    target happens to be last in the group's own list order.
    """

    def test_two_targets_sharing_one_design_var_are_jointly_fit(self):
        # Toy system: paramA = 2*x, paramB = 3*x, one shared design var x.
        # paramA=10 wants x=5; paramB=18 wants x=6. Neither is individually
        # satisfiable at the same time, so a genuine joint least-squares fit
        # lands strictly BETWEEN 5 and 6 (analytically x = 1002/185 ~ 5.4098:
        # minimizing (2x/10-1)^2 + (3x/18-1)^2 = (x/5-1)^2 + (x/6-1)^2).
        target_a = _toy_target("modeA", "x", "shared_group")
        target_b = _toy_target("modeB", "x", "shared_group")

        system_target_params = {
            param_nonlin("modeA", "modeA"): 10.0,
            param_nonlin("modeB", "modeB"): 18.0,
        }
        # Current state: x=3 -> paramA=6, paramB=9.
        system_optimized_params = {
            param_nonlin("modeA", "modeA"): 6.0,
            param_nonlin("modeB", "modeB"): 9.0,
        }

        anmod = ANModOptimizer([target_a, target_b], system_target_params)
        updated, _results = anmod.calculate_target_design_var(
            system_optimized_params, {"x": "3"}
        )

        x_updated = float(updated["x"].split()[0])
        assert 5.0 < x_updated < 6.0
        assert x_updated == pytest.approx(5.409836, abs=1e-4)

    def test_shared_design_var_with_mismatched_bounds_raises(self):
        target_a = _toy_target("modeA", "x", "shared_group")
        target_b = _toy_target("modeB", "x", "shared_group")
        target_b.design_var_constraint = {"larger_than": "0", "smaller_than": "50"}

        system_target_params = {
            param_nonlin("modeA", "modeA"): 10.0,
            param_nonlin("modeB", "modeB"): 18.0,
        }
        system_optimized_params = {
            param_nonlin("modeA", "modeA"): 6.0,
            param_nonlin("modeB", "modeB"): 9.0,
        }

        anmod = ANModOptimizer([target_a, target_b], system_target_params)
        with pytest.raises(ValueError, match="DIFFERENT design_var_constraint bounds"):
            anmod.calculate_target_design_var(system_optimized_params, {"x": "3"})
