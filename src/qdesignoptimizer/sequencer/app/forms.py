"""A study's files as forms: settings in, YAML out. Imports no Dash.

The GUI never shows a file's text. Each file is turned into a tree of
sections and fields here, the layout draws that tree, and an edit comes
back as ``{path: text}`` and is written into the file's data at that
path, converted to the type the value had:

* an object (``class:``/``args:``/``extra:``) is a section titled by its
  class, one field per argument;
* a nested object -- a capacitance study inside a MiniStudy, a junction
  setup -- is a section inside it, to any depth;
* a list of objects (``capacitance_matrix_studies``, the opt targets)
  is a section per item, each of which can be duplicated, removed or
  switched off (``active: false``: kept in the file, skipped when the
  study is built -- see ``qdo_files.is_active``);
* an opt target is a section titled by its helper (its ``args`` are the
  fields) or by what it tunes (every field of an explicit target, with
  ``prop_to`` as its expression);
* a scalar is a field; so is a list of scalars or of pins, written the
  way YAML writes a short list (``[res_x, end]``).

Items are added from the backend's own list (``item_choices``: QDO's
classes and helpers), so everything here stays something QDO builds.

The optional files (a study's own design variables or parameter
targets) are drawn as overrides: every project value, each with a
"set here" switch. Only the switched-on ones are written; the rest are
pulled from the project when the study starts.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any

import yaml

from ..engine.job import numbers

ACTIVE = "active"
_SCALARS = (type(None), bool, int, float, str)


class FormError(ValueError):
    """An edit that cannot go into the file, with the field it is for."""


@dataclass
class Field:
    """One editable value.

    Attributes:
        path: Where it is in the file's data.
        label: Its name.
        kind: ``bool``, ``number``, ``text``, ``list`` or ``none``.
        value: What the input shows (a bool for ``bool``, else text).
    """

    path: tuple
    label: str
    kind: str
    value: Any


@dataclass
class Section:
    """A group of fields: an object, a nested object, a list item.

    Attributes:
        path: Where it is in the file's data.
        title: Its heading.
        fields: Its own values.
        children: Nested sections.
        item: An item of a list: it can be duplicated, removed, switched
            off.
        active: False when switched off.
        note: A line under the heading.
    """

    path: tuple
    title: str
    fields: list = field(default_factory=list)
    children: list = field(default_factory=list)
    item: bool = False
    active: bool = True
    note: str = ""


def key(path) -> str:
    """A path as a string, for component ids."""
    return json.dumps(list(path))


def unkey(text: str) -> tuple:
    """Inverse of ``key``."""
    return tuple(json.loads(text))


# -- building ------------------------------------------------------------


def _is_object(node: Any) -> bool:
    return (
        isinstance(node, dict)
        and "class" in node
        and set(node) - {ACTIVE} <= {"class", "args", "extra"}
    )


def _simple(value: Any) -> bool:
    """A value one input can hold."""
    if isinstance(value, _SCALARS):
        return True
    if isinstance(value, dict):
        return not value
    if isinstance(value, (list, tuple)):
        return all(
            isinstance(v, _SCALARS)
            or (
                isinstance(v, (list, tuple)) and all(isinstance(x, _SCALARS) for x in v)
            )
            for v in value
        )
    return False


def _kind(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "text"
    return "list"


def show(value: Any) -> Any:
    """What an input shows for a value."""
    if isinstance(value, bool):
        return value
    if value is None:
        return ""
    if isinstance(value, (int, float, str)):
        return str(value)
    return yaml.safe_dump(
        value, default_flow_style=True, sort_keys=False, width=10**6
    ).strip()


def _short(ref: str) -> str:
    return str(ref).split(":")[-1].split(".")[-1]


def _members(node: dict, path: tuple, section: Section, extra=False, lists=()):
    """Fill ``section`` from a mapping's entries. A path in ``lists`` is a
    list one can add items to: a section even while empty (or None)."""
    for name, value in node.items():
        if name == ACTIVE:
            continue
        here = path + (name,)
        label = f"{name} (set after building)" if extra else name
        if here in lists and (value is None or isinstance(value, list)):
            section.children.append(_list(value or [], here, name, lists))
            continue
        if _simple(value) and not (
            isinstance(value, list) and value and _is_object(value[0])
        ):
            section.fields.append(Field(here, label, _kind(value), show(value)))
        elif _is_object(value):
            section.children.append(_object(value, here, name, lists=lists))
        elif isinstance(value, list):
            section.children.append(_list(value, here, name, lists))
        elif isinstance(value, dict):
            child = Section(here, name)
            _members(value, here, child, lists=lists)
            section.children.append(child)
        else:
            section.fields.append(Field(here, label, "list", show(value)))


def _object(node: dict, path: tuple, title: str = "", item=False, lists=()) -> Section:
    heading = _short(node["class"])
    section = Section(
        path,
        f"{title}: {heading}" if title and not item else heading,
        item=item,
        active=node.get(ACTIVE) is not False,
    )
    _members(node.get("args") or {}, path + ("args",), section, lists=lists)
    if node.get("extra"):
        _members(node["extra"], path + ("extra",), section, extra=True, lists=lists)
    return section


def _list(items: list, path: tuple, title: str, lists=()) -> Section:
    section = Section(path, title, note=f"{len(items)} item(s)")
    for i, item in enumerate(items):
        here = path + (i,)
        if _is_object(item):
            section.children.append(_object(item, here, item=True, lists=lists))
        elif isinstance(item, dict):
            child = Section(
                here,
                f"{title} {i + 1}",
                item=True,
                active=item.get(ACTIVE) is not False,
            )
            _members(item, here, child, lists=lists)
            section.children.append(child)
        else:
            section.fields.append(Field(here, f"{title}[{i}]", _kind(item), show(item)))
    return section


def _target(entry: dict, path: tuple, n: int) -> Section:
    if "helper" in entry:
        title = _short(entry["helper"])
        section = Section(path, f"{n}. {title}", item=True)
        _members(entry.get("args") or {}, path + ("args",), section)
        if "why" in entry:
            section.fields.append(
                Field(path + ("why",), "why", "text", show(entry["why"]))
            )
    else:
        title = (
            f"{entry.get('target_param_type', '?')} via "
            f"{entry.get('design_var', '?')}"
        )
        section = Section(path, f"{n}. {title}", item=True)
        _members(entry, path, section)
        section.note = "prop_to: an expression of p['<param>'], v['<var>']"
    section.active = entry.get(ACTIVE) is not False
    return section


def build(data: Any, title: str = "", lists=()) -> Section:
    """The form of one file's data.

    Args:
        data: The parsed file.
        title: The heading for a file that is not an object.
        lists: Paths of lists items can be added to (drawn as sections
            even while empty).
    """
    lists = {tuple(p) for p in lists}
    if _is_object(data):
        return _object(data, (), lists=lists)
    if isinstance(data, dict) and isinstance(data.get("targets"), list):
        section = Section(("targets",), title or "Opt targets")
        for i, entry in enumerate(data["targets"]):
            here = ("targets", i)
            if isinstance(entry, dict):
                section.children.append(_target(entry, here, i + 1))
        section.note = (
            f"{sum(1 for e in data['targets'] if _on(e))} of "
            f"{len(data['targets'])} switched on"
        )
        return section
    section = Section((), title)
    if isinstance(data, dict):
        _members(data, (), section, lists=lists)
    return section


def _on(entry: Any) -> bool:
    return not (isinstance(entry, dict) and entry.get(ACTIVE) is False)


def fields_of(section: Section) -> list[Field]:
    """Every field of a form, depth first."""
    out = list(section.fields)
    for child in section.children:
        out += fields_of(child)
    return out


# -- editing -------------------------------------------------------------


def get(data: Any, path: tuple) -> Any:
    """The value at ``path``."""
    for step in path:
        data = data[step]
    return data


def _parent(data: Any, path: tuple):
    if not path:
        raise FormError("the file itself cannot be changed that way")
    return get(data, path[:-1]), path[-1]


def parse(raw: Any, original: Any, label: str = "value") -> Any:
    """An input's content as the type the value had.

    Raises:
        FormError: If it cannot be that type.
    """
    if isinstance(original, bool):
        return bool(raw)
    text = "" if raw is None else str(raw).strip()
    if isinstance(original, str):
        return str(raw if raw is not None else "")
    if original is None:
        if not text:
            return None
        return _yaml(text, label)
    if isinstance(original, (int, float)):
        value = _yaml(text, label) if text else None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise FormError(f"{label}: {text!r} is not a number")
        return value
    value = _yaml(text, label) if text else type(original)()
    if isinstance(original, (list, tuple)) and not isinstance(value, list):
        raise FormError(f"{label}: {text!r} is not a list, e.g. [a, b]")
    if isinstance(original, dict) and not isinstance(value, dict):
        raise FormError(f"{label}: {text!r} is not a mapping, e.g. {{}}")
    return value


def _yaml(text: str, label: str) -> Any:
    try:
        return numbers(yaml.safe_load(text))
    except yaml.YAMLError as exc:
        raise FormError(f"{label}: {text!r} does not read: {exc}") from exc


def apply(data: Any, edits: dict) -> Any:
    """``data`` with every edit written in (a copy).

    Args:
        data: The file's data.
        edits: ``{key(path): what the input holds}``.

    Raises:
        FormError: For a value that cannot go in, or a path that is no
            longer there.
    """
    out = copy.deepcopy(data)
    for text, raw in edits.items():
        path = unkey(text)
        try:
            parent, last = _parent(out, path)
            original = parent[last]
        except (KeyError, IndexError, TypeError) as exc:
            raise FormError(
                f"{'.'.join(map(str, path))} is no longer in the file"
            ) from exc
        parent[last] = parse(raw, original, str(path[-1]))
    return out


def add_item(data: Any, path: tuple, item: Any) -> Any:
    """``data`` with ``item`` appended to the list at ``path`` (made a
    list first if it is None) -- a copy.

    Raises:
        FormError: If there is no list there.
    """
    out = copy.deepcopy(data)
    try:
        parent, last = _parent(out, tuple(path))
        current = parent[last]
    except (KeyError, IndexError, TypeError) as exc:
        raise FormError("that list is no longer in the file") from exc
    if current is None:
        current = parent[last] = []
    if not isinstance(current, list):
        raise FormError(f"{path[-1]} is not a list")
    current.append(copy.deepcopy(item))
    return out


def operate(data: Any, path: tuple, op: str) -> Any:
    """``data`` with one list item duplicated, removed or switched on/off
    (a copy).

    Raises:
        FormError: For a path that is not a list item, or an unknown op.
    """
    out = copy.deepcopy(data)
    try:
        parent, last = _parent(out, tuple(path))
        item = parent[last]
    except (KeyError, IndexError, TypeError) as exc:
        raise FormError("that item is no longer in the file") from exc
    if not isinstance(parent, list):
        raise FormError("only an item of a list can be changed that way")
    if op == "remove":
        del parent[last]
    elif op == "duplicate":
        parent.insert(last + 1, copy.deepcopy(item))
    elif op == "toggle":
        if not isinstance(item, dict):
            raise FormError("only an object can be switched off")
        if item.get(ACTIVE) is False:
            del item[ACTIVE]
        else:
            item[ACTIVE] = False
    else:
        raise FormError(f"unknown operation {op!r}")
    return out


# -- overrides (a study's own design variables / parameter targets) ------


@dataclass
class Override:
    """One project value a study may set for itself."""

    name: str
    own: bool
    value: str
    project: str
    source: str = ""


def overrides(
    own: dict | None, project: dict, sources: dict | None = None
) -> list[Override]:
    """Every value a study could set here: the project's, each marked
    whether the study's own file sets it (and to what), and which study
    last wrote the project's value (``sources``)."""
    own = own or {}
    sources = sources or {}
    names = list(project) + [k for k in own if k not in project]
    return [
        Override(
            n,
            n in own,
            show(own[n] if n in own else project.get(n)),
            show(project.get(n)) if n in project else "(not in project)",
            sources.get(n, ""),
        )
        for n in names
    ]


def apply_overrides(project: dict, rows: dict) -> dict:
    """The study's own file from the override rows.

    Args:
        project: The project's values (sets each value's type).
        rows: ``{name: (set here, text)}``.

    Raises:
        FormError: For a value that cannot be its project value's type.
    """
    out = {}
    for name, (own, raw) in rows.items():
        if own:
            out[name] = parse(raw, project.get(name), name)
    return out


def header(text: str) -> str:
    """The leading comment block of a YAML file, without its ``#``."""
    lines = []
    for line in text.splitlines():
        if not line.startswith("#"):
            break
        lines.append(line[1:].removeprefix(" "))
    return "\n".join(lines)
