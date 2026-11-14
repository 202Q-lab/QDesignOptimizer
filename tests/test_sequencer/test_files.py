"""Expressions and the object <-> file codec."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qdesignoptimizer.sequencer.backends import qdo_files as qf
from qdesignoptimizer.sequencer.engine.expressions import Expression, ExpressionError

# -- expressions --------------------------------------------------------


def test_expression_evaluates_and_lists_its_keys():
    e = Expression("v['L'] ** -1.17 * abs(p['f'] - p['g']) / sqrt(4) + pi")
    assert e.keys == {"p": {"f", "g"}, "v": {"L"}}
    want = 2.0**-1.17 * abs(5 - 7) / 2 + math.pi
    assert math.isclose(e({"f": 5, "g": 7}, {"L": 2.0}), want)


@pytest.mark.parametrize(
    "text",
    [
        "os.system('x')",
        "__import__('os')",
        "q['a']",
        "v[0]",
        "x + 1",
        "v['a'] if 1 else 2",
        "max(v['a'], key=1)",
        "'s'",
    ],
)
def test_expression_refuses_anything_else(text):
    with pytest.raises(ExpressionError):
        Expression(text)


def test_expression_power_overflows_instead_of_hanging():
    with pytest.raises(OverflowError):
        Expression("10 ** 10 ** 10")({}, {})


def test_expression_refuses_unit_strings():
    with pytest.raises(ExpressionError, match="not a number"):
        Expression("v['L'] * 2")({}, {"L": "120um"})


# -- generic objects ----------------------------------------------------


@pytest.fixture(autouse=True)
def _allow_this_module(monkeypatch):
    """The classes below live here, outside the allowlist."""
    monkeypatch.setattr(
        qf, "ALLOWED_MODULES", qf.ALLOWED_MODULES | {__name__.split(".")[0]}
    )


class Inner:
    def __init__(self, mode, size=2):
        self.mode = mode
        self.size = size
        self.render = None


class Outer:
    def __init__(self, names, inner=None, ports=(), flag=False):
        self.names = names
        self.inner = inner
        self.ports = list(ports)
        self.flag = flag
        self.callback = print  # a callable: left out


class Lossy:
    def __init__(self, a):
        self.b = a * 2  # does not keep ``a``


def test_objects_round_trip_with_attributes_set_later():
    obj = Outer(["x", "y"], Inner("m", 3), [("c", "pin", 50)], flag=np.bool_(True))
    obj.grounded = [["c", "pin2"]]
    data = qf.dump_object(obj)
    assert data["extra"] == {"grounded": [["c", "pin2"]]}
    back = qf.load_object(data)
    assert back.inner.size == 3 and back.grounded == [["c", "pin2"]]
    assert qf.dump_object(back) == data


def test_an_object_that_drops_its_argument_is_refused():
    with pytest.raises(qf.CodecError, match="does not keep"):
        qf.dump_object(Lossy(1))


# -- lambdas -> expressions ---------------------------------------------


def test_a_study_file_cannot_name_any_function(monkeypatch):
    """Review 2026-11-10: a file's class:/ref:/helper: is imported and
    called on every check, so it may only name allowed packages."""
    monkeypatch.setattr(qf, "ALLOWED_MODULES", {"qdesignoptimizer"})
    with pytest.raises(qf.CodecError, match="may not name"):
        qf.load_object({"class": "os.path:join", "args": {}})
    with pytest.raises(qf.CodecError, match="may not name"):
        qf.decode({"ref": "os:system"})
    with pytest.raises(qf.CodecError, match="may not name"):
        qf.data_to_target({"helper": "subprocess:run", "args": {}})


def test_to_file_data_goes_through_the_yaml_text():
    obj = Outer(["x"], Inner("m"), [("c", "pin", 50)])
    data = qf.to_file_data(obj)
    back = qf.load_object(data)
    assert back.ports == [["c", "pin", 50]]  # the documented tuple->list


@pytest.mark.parametrize(
    "ref",
    [
        "qdesignoptimizer.design_analysis:os.system",
        "qdesignoptimizer.utils.optimization_targets:np.save",
        "qdesignoptimizer.design_analysis:__builtins__",
        "qdesignoptimizer.design_analysis:os",
    ],
)
def test_the_allowlist_cannot_be_walked_out_of(ref):
    """Review round 3: attribute chains walked from an allowed module
    into os / numpy."""
    pytest.importorskip("qdesignoptimizer")
    monkeypatch_free = {
        "qdesignoptimizer",
        "qiskit_metal",
        "quantum_metal_contrib",
    }
    old = set(qf.ALLOWED_MODULES)
    qf.ALLOWED_MODULES.clear()
    qf.ALLOWED_MODULES.update(monkeypatch_free)
    try:
        with pytest.raises(qf.CodecError):
            qf.resolve_ref(ref)
    finally:
        qf.ALLOWED_MODULES.clear()
        qf.ALLOWED_MODULES.update(old)
