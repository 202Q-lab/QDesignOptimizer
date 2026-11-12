"""Numeric expressions over two mappings, ``p`` and ``v``.

The file form of a QDO ``prop_to``: ``"v['design_var_length_x'] ** -1.17
* abs(p['qubit_freq'] - p['resonator_freq'])"``. Parsed once, checked
against a whitelist, evaluated without ``eval``:

* numbers (always as float, so ``10 ** 10 ** 10`` overflows at once
  instead of hanging on an integer power);
* ``p[<string>]`` and ``v[<string>]``, the only subscripts;
* ``+ - * / ** %``, unary ``-`` / ``+``;
* the functions ``abs sqrt exp log log10 min max`` and the constants
  ``pi`` and ``e``.

Nothing else: no attributes, no other names, no other calls.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any, Callable, Mapping

_BIN = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos}
FUNCTIONS: dict[str, Callable] = {
    "abs": abs,
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "min": min,
    "max": max,
}
CONSTANTS = {"pi": math.pi, "e": math.e}
MAPPINGS = ("p", "v")


class ExpressionError(ValueError):
    """An expression is malformed, or reads a key that is not there."""


class Expression:
    """A parsed, whitelisted expression of ``p`` and ``v``.

    Args:
        text: The expression.

    Raises:
        ExpressionError: If it uses anything outside the whitelist.
    """

    def __init__(self, text: str):
        self.text = str(text).strip()
        try:
            self._tree = ast.parse(self.text, mode="eval").body
        except SyntaxError as exc:
            raise ExpressionError(
                f"invalid expression {self.text!r}: {exc.msg}"
            ) from exc
        self.keys: dict[str, set[str]] = {"p": set(), "v": set()}
        self._check(self._tree)

    def _check(self, node: ast.AST) -> None:
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                raise ExpressionError(
                    f"{node.value!r} is not a number in {self.text!r}"
                )
            return
        if isinstance(node, ast.Name):
            if node.id not in CONSTANTS:
                raise ExpressionError(
                    f"name {node.id!r} in {self.text!r}: write p['...'] or "
                    f"v['...'] (constants: {', '.join(CONSTANTS)})"
                )
            return
        if isinstance(node, ast.Subscript):
            key = _subscript_key(node)
            if not (
                isinstance(node.value, ast.Name)
                and node.value.id in MAPPINGS
                and isinstance(key, str)
            ):
                raise ExpressionError(
                    f"only p['<name>'] and v['<name>'] may be indexed, "
                    f"in {self.text!r}"
                )
            self.keys[node.value.id].add(key)
            return
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            self._check(node.left)
            self._check(node.right)
            return
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            self._check(node.operand)
            return
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in FUNCTIONS
            and not node.keywords
        ):
            for arg in node.args:
                self._check(arg)
            return
        raise ExpressionError(f"{type(node).__name__} is not allowed in {self.text!r}")

    def __call__(self, p: Mapping[str, Any], v: Mapping[str, Any]) -> float:
        """Evaluate with these two mappings."""
        return self._eval(self._tree, {"p": p, "v": v})

    def _eval(self, node, maps):
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.Name):
            return CONSTANTS[node.id]
        if isinstance(node, ast.Subscript):
            key = _subscript_key(node)
            mapping = maps[node.value.id]
            try:
                value = mapping[key]
            except KeyError as exc:
                raise ExpressionError(
                    f"{node.value.id}[{key!r}] is not there " f"(in {self.text!r})"
                ) from exc
            return _to_float(value)
        if isinstance(node, ast.BinOp):
            return _BIN[type(node.op)](
                self._eval(node.left, maps), self._eval(node.right, maps)
            )
        if isinstance(node, ast.UnaryOp):
            return _UNARY[type(node.op)](self._eval(node.operand, maps))
        return FUNCTIONS[node.func.id](*(self._eval(a, maps) for a in node.args))

    def __repr__(self) -> str:
        return f"Expression({self.text!r})"


def _subscript_key(node: ast.Subscript):
    key = node.slice
    if isinstance(key, ast.Index):  # pragma: no cover -- Python < 3.9
        key = getattr(key, "value")  # ast.Index wraps the key
    return key.value if isinstance(key, ast.Constant) else None


def _to_float(value: Any) -> float:
    """A number. QDO hands prop_to bare numbers; a unit string such as
    ``"120um"`` is refused rather than stripped, since stripping would
    read ``"1.2mm"`` as 1.2 in the same sum as microns."""
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ExpressionError(f"{value!r} is not a number") from exc
