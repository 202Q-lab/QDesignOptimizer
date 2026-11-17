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


def _linear_target(mode_name, design_var, constraint, independent_target=True):
    """param(mode) is proportional to the design variable."""
    return OptTarget(
        target_param_type=NONLIN,
        involved_modes=[mode_name, mode_name],
        design_var=design_var,
        design_var_constraint=constraint,
        prop_to=lambda p, v, dv=design_var: v[dv],
        independent_target=independent_target,
    )


WIDE = {"larger_than": "0.1um", "smaller_than": "20um"}


def _optimize(targets, current, targets_values, design_vars, adjustment_rate=1.0):
    optimizer = ANModOptimizer(targets, targets_values, adjustment_rate=adjustment_rate)
    return optimizer.calculate_target_design_var(current, design_vars)


def test_proportional_target_is_solved_in_one_step():
    # param = 2 at x = 1 um and param ∝ x, so reaching 4 needs x = 2 um
    updated, minimization = _optimize(
        [_linear_target("a", "x", WIDE)],
        {"a_to_a_nonlin": 2.0},
        {"a_to_a_nonlin": 4.0},
        {"x": "1 um", "unrelated": "3 um"},
    )
    assert set(updated) == {"x"}
    value, unit = updated["x"].split()
    assert (float(value), unit) == (pytest.approx(2.0, rel=1e-6), "um")
    assert len(minimization) == 1
    assert minimization[0]["targets_to_minimize_for"] == ["x"]
    assert minimization[0]["final_cost"] == pytest.approx(0, abs=1e-10)


def test_adjustment_rate_moves_part_of_the_way():
    updated, _ = _optimize(
        [_linear_target("a", "x", WIDE)],
        {"a_to_a_nonlin": 2.0},
        {"a_to_a_nonlin": 4.0},
        {"x": "1 um"},
        adjustment_rate=0.25,
    )
    assert float(updated["x"].split()[0]) == pytest.approx(1.25, rel=1e-6)


@pytest.mark.parametrize(
    "current_value, expected_x",
    [(2.0, 1.5), (8.0, 0.75)],  # optimum would be 2.0 resp. 0.5
)
def test_design_variable_is_clipped_to_constraint(current_value, expected_x):
    constraint = {"larger_than": "0.75um", "smaller_than": "1.5um"}
    optimizer = ANModOptimizer(
        [_linear_target("a", "x", constraint)], {"a_to_a_nonlin": 4.0}
    )
    # bypass the bounded minimizer to test the clipping itself
    constrained = optimizer._constrain_design_value(
        "1 um", f"{4.0 / current_value} um", constraint
    )
    assert float(constrained.split()[0]) == pytest.approx(expected_x)


def test_constraint_units_must_match_design_variable():
    optimizer = ANModOptimizer([], {})
    with pytest.raises(AssertionError, match="Units"):
        optimizer._constrain_design_value(
            "1 um", "2 um", {"larger_than": "1nH", "smaller_than": "5nH"}
        )


def test_targets_are_grouped_by_independence():
    independent = _linear_target("a", "x", WIDE, independent_target=True)
    group_1a = _linear_target("b", "y", WIDE, independent_target="group_1")
    group_2 = _linear_target("c", "z", WIDE, independent_target="group_2")
    group_1b = _linear_target("d", "w", WIDE, independent_target="group_1")
    dependent_1 = _linear_target("e", "u", WIDE, independent_target=False)
    dependent_2 = _linear_target("f", "t", WIDE, independent_target=False)

    groups = ANModOptimizer.group_targets(
        [dependent_1, group_1a, independent, group_2, group_1b, dependent_2]
    )
    assert groups == [
        [independent],
        [group_1a, group_1b],
        [group_2],
        [dependent_1, dependent_2],
    ]


def test_invalid_independent_target_raises():
    with pytest.raises(ValueError, match="independent_target"):
        ANModOptimizer.group_targets(
            [_linear_target("a", "x", WIDE, independent_target=1.5)]
        )


@pytest.mark.parametrize(
    "x, constraint",
    [
        ("4961 um", {"larger_than": "5um", "smaller_than": "15000um"}),
        ("-4961 um", {"larger_than": "-15000um", "smaller_than": "-5um"}),
    ],
)
def test_a_small_remaining_error_on_a_large_variable_is_closed(x, constraint):
    # 0.2 % short, on a variable of thousands of um with wide bounds: in raw
    # units L-BFGS-B stopped here without moving (a resonator 12 MHz low).
    updated, minimization = _optimize(
        [_linear_target("a", "x", constraint)],
        {"a_to_a_nonlin": 0.998},
        {"a_to_a_nonlin": 1.0},
        {"x": x},
    )
    value = float(updated["x"].split()[0])
    assert value == pytest.approx(float(x.split()[0]) / 0.998, rel=1e-7)
    assert minimization[0]["final_cost"] == pytest.approx(0, abs=1e-12)
