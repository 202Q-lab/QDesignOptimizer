"""QDO objects <-> the YAML files of a queued study.

A project's studies are templates (``ministudies/``, ``opt_targets/``;
see ``qdo_templates``). Queueing one fills it in and writes the result
into the study's folder, where it can be read, argued about and edited;
running the study rebuilds the objects from those files. Nothing else is
read at run time, so what ran is what the folder says.

**MiniStudy (and anything inside it)** -- written as the constructor
call that rebuilds it: ``class``, the ``args`` of its ``__init__`` read
back from the attributes of the same name, and ``extra`` for attributes
set after construction. Callables are left out (QDO sets the render
function itself). ``dump_object`` rebuilds the object from what it wrote
and refuses if anything would come back different.

**OptTarget** -- either form:

* ``helper:`` + ``args:``: a QDO helper call
  (``get_opt_target_res_freq_via_length``); the formula stays QDO's,
  only the arguments are in the file;
* the explicit fields, with ``prop_to`` as an expression of ``p`` and
  ``v`` (``engine.expressions``).

An entry (or a nested study in a list) with ``active: false`` is kept in
its file and left out when the study is built.

An opt-target entry also carries what is the sequencer's, not QDO's
(``ENTRY_KEYS``): its ``name``, ``why``, ``active``, ``tolerance`` (how
close counts as converged) and ``simulation`` (the kind of study it is
tuned in). None of them reaches the OptTarget.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any

import yaml

from ..engine.expressions import Expression, ExpressionError
from ..engine.job import exponent_floats

OPT_TARGET = "qdesignoptimizer.design_analysis_types:OptTarget"
_TARGET_FIELDS = (
    "target_param_type",
    "involved_modes",
    "design_var",
    "design_var_constraint",
    "independent_target",
)


class CodecError(ValueError):
    """An object cannot be written to, or read from, a file faithfully."""


# -- references -----------------------------------------------------------


#: A file's ``class:``, ``helper:`` and ``ref:`` entries are imported and
#: CALLED when the study is checked or run, and the GUI checks on every
#: save, so what a file may name is kept to what a study needs, per kind
#: of entry, as exact modules:
#:
#: * ``class:`` -- QDO's study and target types;
#: * ``helper:`` -- QDO's opt-target helpers;
#: * ``ref:`` -- QDO's naming functions (a helper's ``design_var_*``
#:   argument, say), passed on, not called by the codec.
#:
#: and only a top-level name (``module:name``), defined in that module:
#: ``Class.method`` or a name the module merely imports is refused.
ROLE_MODULES = {
    "class": {
        "qdesignoptimizer.design_analysis_types",
        "qdesignoptimizer.sim_capacitance_matrix",
    },
    "helper": {"qdesignoptimizer.utils.optimization_targets"},
    "ref": {
        "qdesignoptimizer.utils.names_design_variables",
        "qdesignoptimizer.utils.names_parameters",
        "qdesignoptimizer.utils.names_qiskit_components",
    },
}

#: Top-level packages a study file may name for any kind of entry: what
#: a project adds (``QDOProject.code_modules``), its own code.
ALLOWED_MODULES: set[str] = set()


def allow_modules(*names: str) -> None:
    """Let study files also name these top-level modules."""
    ALLOWED_MODULES.update(n.split(".")[0] for n in names if n)


def _trusted(module_name: str) -> bool:
    return module_name.split(".")[0] in ALLOWED_MODULES


def _check_allowed(module_name: str, what: str, role: str | None = None) -> None:
    roles = [role] if role else list(ROLE_MODULES)
    allowed = set().union(*(ROLE_MODULES[r] for r in roles))
    if module_name not in allowed and not _trusted(module_name):
        raise CodecError(
            f"{what} is in {module_name}, which a study file may not name "
            f"{'as ' + role + ': ' if role else ''}"
            f"(allowed: {sorted(allowed)}; a project adds its own "
            f"through QDOProject.code_modules)"
        )


def ref_of(obj: Any, role: str | None = None) -> str:
    """``module:name`` of an importable top-level function or class,
    refused unless it may be read back as an entry of kind ``role`` (any
    kind when None)."""
    qual = getattr(obj, "__qualname__", "")
    module = getattr(obj, "__module__", "")
    if not module or not qual or "<" in qual or "." in qual:
        raise CodecError(f"{obj!r} is not importable by a top-level name")
    # Refused when the file is written, not first when it is read back.
    _check_allowed(module, f"{qual}", role)
    return f"{module}:{qual}"


def resolve_ref(ref: str, role: str = "ref") -> Any:
    """Import ``module:name`` for an entry of kind ``role`` (``class``,
    ``helper`` or ``ref``): a top-level function or class DEFINED in a
    module allowed for that kind (``ROLE_MODULES``) or in the project's
    own code.

    The module named is not enough: an allowed module imports others
    (``design_analysis:os``), and its classes have methods that do
    anything (``QAnsysRenderer.open_ansys``). Hence one name, no dots,
    and the object's ``__module__`` must be the module named.
    """
    if ":" not in ref:
        raise CodecError(f"{ref!r} is not module:name")
    module_name, qual = ref.split(":", 1)
    if "." in qual:
        raise CodecError(f"{ref}: only a top-level name may be referenced")
    if qual.startswith("_"):
        raise CodecError(f"{ref}: private names cannot be referenced")
    _check_allowed(module_name, ref, role)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise CodecError(f"cannot import {module_name}: {exc}") from exc
    try:
        obj = getattr(module, qual)
    except AttributeError as exc:
        raise CodecError(f"{module_name} has no {qual}") from exc
    if inspect.ismodule(obj) or not (inspect.isclass(obj) or inspect.isfunction(obj)):
        raise CodecError(f"{ref} is not a function or class")
    if role == "class" and not inspect.isclass(obj):
        raise CodecError(f"class: {ref} is not a class")
    if role == "helper" and not inspect.isfunction(obj):
        raise CodecError(f"helper: {ref} is not a function")
    defined = getattr(obj, "__module__", "") or "?"
    if defined != module_name and not (_trusted(module_name) and _trusted(defined)):
        raise CodecError(
            f"{ref} is defined in {defined}, not in {module_name}: "
            f"name it where it is defined"
        )
    return obj


# -- generic values -----------------------------------------------------


def encode(value: Any, where: str = "value") -> Any:
    """Plain YAML data for ``value``.

    Raises:
        CodecError: For something with no faithful file form.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value
    if hasattr(value, "item") and not hasattr(value, "__len__"):
        return encode(value.item(), where)  # numpy scalar
    if hasattr(value, "tolist"):
        return encode(value.tolist(), where)  # numpy array
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [encode(v, f"{where}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CodecError(f"{where} has a non-string key {key!r}")
            out[key] = encode(item, f"{where}.{key}")
        return out
    if inspect.isclass(value) or inspect.isfunction(value):
        return {"ref": ref_of(value, "ref")}
    if hasattr(value, "__dict__"):
        return dump_object(value, where)
    raise CodecError(f"{where}: {type(value).__name__} has no file form")


#: An entry switched off in the GUI: kept in its file, skipped when the
#: study is built (an opt target, a nested study in a list).
ACTIVE = "active"


def is_active(entry: Any) -> bool:
    """False only for a mapping that says ``active: false``."""
    return not (isinstance(entry, dict) and entry.get(ACTIVE) is False)


#: An opt-target entry's own fields, beside the OptTarget's.
NAME, TOLERANCE, SIMULATION = "name", "tolerance", "simulation"
ENTRY_KEYS = {NAME, "why", ACTIVE, TOLERANCE, SIMULATION}
SIMULATIONS = ("eigenmode", "capacitance")


def tolerance_of(entry: dict):
    """An entry's tolerance as ``(kind, value)`` -- ``("rel", 0.01)`` or
    ``("abs", 5e6)`` -- or None when it sets none. A bare number is
    relative.

    Raises:
        CodecError: For anything else.
    """
    raw = entry.get(TOLERANCE)
    if raw is None:
        return None
    if isinstance(raw, dict) and len(raw) == 1:
        kind, value = next(iter(raw.items()))
    else:
        kind, value = "rel", raw
    if isinstance(value, str):
        try:
            value = float(value)  # YAML 1.1 reads 5e6 as text
        except ValueError:
            pass
    if (
        kind not in ("rel", "abs")
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value <= 0
    ):
        raise CodecError(
            f"tolerance {raw!r}: give a positive number (relative) or "
            f"{{abs: <number>}} / {{rel: <number>}}"
        )
    return kind, float(value)


def check_entry_keys(entry: dict) -> None:
    """Refuse an entry whose own fields are malformed.

    Raises:
        CodecError: For a bad ``tolerance``, ``simulation`` or ``name``.
    """
    tolerance_of(entry)
    kind = entry.get(SIMULATION)
    if kind is not None and kind not in SIMULATIONS:
        raise CodecError(f"simulation {kind!r} is not one of {list(SIMULATIONS)}")
    name = entry.get(NAME)
    if name is not None and (not isinstance(name, str) or not name.strip()):
        raise CodecError(f"name {name!r} must be text")


def active_targets(entries) -> list:
    """The opt-target entries that are switched on."""
    return [e for e in entries or [] if is_active(e)]


def decode(data: Any) -> Any:
    """Inverse of ``encode``. A list item marked ``active: false`` is
    left out."""
    if isinstance(data, list):
        return [decode(v) for v in data if is_active(v)]
    if isinstance(data, dict):
        keys = set(data) - {ACTIVE}
        if keys == {"ref"}:
            return resolve_ref(data["ref"])
        if "class" in keys and keys <= {"class", "args", "extra"}:
            return load_object(data)
        return {k: decode(v) for k, v in data.items() if k != ACTIVE}
    return data


def dump_object(obj: Any, where: str = "object") -> dict:
    """The constructor call that rebuilds ``obj``, as data.

    Raises:
        CodecError: If an ``__init__`` argument is not stored on the
            object under its own name, or the rebuilt object differs.
    """
    cls = type(obj)
    params = [
        p
        for p in inspect.signature(cls.__init__).parameters.values()
        if p.name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]
    args: dict[str, Any] = {}
    for param in params:
        if hasattr(obj, param.name):
            value = getattr(obj, param.name)
            if callable(value) and not inspect.isclass(value):
                if inspect.isfunction(value) and "<" not in getattr(
                    value, "__qualname__", "<"
                ):
                    args[param.name] = {"ref": ref_of(value, "ref")}
                continue
            args[param.name] = encode(value, f"{where}.{param.name}")
        elif param.default is inspect.Parameter.empty:
            raise CodecError(
                f"{cls.__name__} does not keep its argument "
                f"{param.name!r}, so it cannot be written to a file"
            )
    try:
        rebuilt = cls(**decode(args))
    except Exception as exc:  # noqa: BLE001 -- reported
        raise CodecError(
            f"{cls.__name__} cannot be rebuilt from its own attributes: " f"{exc}"
        ) from exc
    extra = {}
    fresh = vars(rebuilt)
    for key, value in vars(obj).items():
        if key in args or key.startswith("__"):
            continue
        if callable(value) and not inspect.isclass(value):
            continue
        encoded = encode(value, f"{where}.{key}")
        if key not in fresh or _safe_encode(fresh[key]) != encoded:
            extra[key] = encoded
    data = {"class": ref_of(cls, "class"), "args": args}
    if extra:
        data["extra"] = extra
    return data


def _safe_encode(value):
    try:
        return encode(value)
    except CodecError:
        return object()


def blank_value(param: inspect.Parameter) -> Any:
    """A placeholder for a required argument, by its annotation: an
    empty list, mapping or text, 0 for a number, a Literal's first
    choice, else None (to be filled in)."""
    import typing

    hint = param.annotation
    origin = typing.get_origin(hint)
    if origin is typing.Literal:
        return typing.get_args(hint)[0]
    if origin is typing.Union:  # Optional[...]
        return None
    target = origin or hint
    if isinstance(hint, str):  # postponed annotations
        target = {
            "list": list,
            "dict": dict,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
        }.get(hint.split("[")[0])
    for kind, value in (
        (bool, False),
        (int, 0),
        (float, 0.0),
        (str, ""),
        (list, []),
        (tuple, []),
        (dict, {}),
    ):
        if target is kind:
            return value
    return None


def blank_object(ref: str) -> dict:
    """A class's constructor call as file data, every argument present:
    the defaults as they are, the required ones blank (``blank_value``).
    Not built, so no constructor check can refuse the blanks; the study
    check reports what still has to be filled in.

    Raises:
        CodecError: If ``ref`` is not an allowed class.
    """
    cls = resolve_ref(ref, "class")
    args = {}
    for param in inspect.signature(cls.__init__).parameters.values():
        if param.name == "self" or param.kind in (
            param.VAR_POSITIONAL,
            param.VAR_KEYWORD,
        ):
            continue
        if param.default is inspect.Parameter.empty:
            args[param.name] = blank_value(param)
        else:
            try:
                args[param.name] = encode(param.default, param.name)
            except CodecError:
                continue  # a default with no file form: left to the class
    return {"class": ref_of(cls, "class"), "args": args}


def blank_helper(ref: str) -> dict:
    """An opt-target helper's entry with its required arguments blank
    (the optional ones keep the helper's own defaults)."""
    fn = resolve_ref(ref, "helper")
    args = {
        p.name: blank_value(p)
        for p in inspect.signature(fn).parameters.values()
        if p.default is inspect.Parameter.empty
    }
    return {"helper": ref_of(fn, "helper"), "args": args}


def blank_target() -> dict:
    """An explicit OptTarget entry, every field blank."""
    return {
        "target_param_type": "",
        "involved_modes": [],
        "design_var": "",
        "design_var_constraint": {},
        "independent_target": False,
        "prop_to": "",
    }


def to_file_data(obj: Any) -> dict:
    """``dump_object``, checked through the YAML text itself.

    The data is written to YAML, read back, rebuilt and dumped again;
    anything that does not survive that trip is refused. Tuples are the
    one known normalisation: YAML has no tuple, so they come back as
    lists (QDO only unpacks and indexes its pin and port tuples).

    Raises:
        CodecError: If the YAML round trip changes anything.
    """
    data = dump_object(obj)
    text = yaml.safe_dump(data, sort_keys=False)
    again = dump_object(load_object(yaml.safe_load(text)))
    if again != data:
        raise CodecError(f"{type(obj).__name__} does not survive its own YAML file")
    return data


def load_object(data: dict) -> Any:
    """Rebuild an object written by ``dump_object``."""
    cls = resolve_ref(data["class"], "class")
    # YAML 1.1 reads 1e-3 (no dot) as text; a solver setting written that
    # way must still reach QDO as a number.
    obj = cls(**decode(exponent_floats(data.get("args") or {})))
    for key, value in (data.get("extra") or {}).items():
        setattr(obj, key, decode(exponent_floats(value)))
    return obj


# -- OptTarget ------------------------------------------------------------


class PropTo:
    """A ``prop_to`` callable built from an expression."""

    def __init__(self, text: str):
        self.expression = Expression(text)

    def __call__(self, p, v):
        return self.expression(p, v)

    def __repr__(self) -> str:
        return f"PropTo({self.expression.text!r})"


def data_to_target(entry: dict):
    """Rebuild one OptTarget from its file entry."""
    check_entry_keys(entry)
    if "helper" in entry:
        extra = set(entry) - {"helper", "args"} - ENTRY_KEYS
        if extra:
            raise CodecError(
                f"a helper entry takes helper, args (and "
                f"{', '.join(sorted(ENTRY_KEYS))}), not {sorted(extra)}"
            )
        fn = resolve_ref(entry["helper"], "helper")
        return fn(**decode(entry.get("args") or {}))
    allowed = set(_TARGET_FIELDS) | {"prop_to"} | ENTRY_KEYS
    extra = set(entry) - allowed
    if extra:
        raise CodecError(f"unknown target field(s) {sorted(extra)}")
    cls = resolve_ref(OPT_TARGET, "class")
    kwargs = {f: decode(entry[f]) for f in _TARGET_FIELDS if f in entry}
    prop = entry.get("prop_to")
    if prop is not None:
        try:
            kwargs["prop_to"] = PropTo(prop)
        except ExpressionError as exc:
            raise CodecError(f"prop_to: {exc}") from exc
    return cls(**kwargs)


def entry_targets(entry: dict) -> list:
    """The OptTargets one file entry gives: one, or every one of a helper
    that returns a list (QDO has such helpers). A helper that gives none
    is refused: it would leave the study silently without that target.

    Raises:
        CodecError: For an entry that does not build, or gives nothing.
    """
    built = data_to_target(entry)
    targets = list(built) if isinstance(built, (list, tuple)) else [built]
    if not targets:
        raise CodecError(
            f"{entry.get('helper', 'the entry')} gives no target with "
            f"these arguments"
        )
    return targets


def build_targets(entries) -> list:
    """Every OptTarget of a file's switched-on entries, flattened."""
    out = []
    for entry in active_targets(entries):
        out.extend(entry_targets(entry))
    return out
