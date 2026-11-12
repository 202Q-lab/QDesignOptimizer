"""A project's templates: MiniStudies and opt targets, as YAML.

A design project holds::

    <design>/ministudies/<name>.yaml     one MiniStudy, and nothing else
    <design>/opt_targets/*.yaml          the project's opt targets
    (or one file, <design>/opt_targets.yaml)

**A ministudy template is the MiniStudy alone**, exactly as a queued
``ministudy.yaml`` holds it (``class``, ``args``, ``extra``)::

    class: qdesignoptimizer.design_analysis_types:MiniStudy
    args:
      qiskit_component_names: ["name_resonator_ATS{n}_"]  # quote {..}
      modes: ["resonator_ATS{n}"]
      nbr_passes: 15
      ...

Everything the solver is told -- passes, ``delta_f``, mesh -- is there
and nowhere else. A study is not a template: it exists only as a folder
under ``out/``, made by pairing one of these with the opt targets that
fit it.

**Which studies a template makes.** A template with ``{placeholders}``
is a family: one study per subsystem whose values cover them (the
project's ``subjects``, each with its own values and ``{subject}``, its
key), called ``<subsystem>.<file stem>``. A template without is one
study: ``<subsystem>.<rest>`` when the file is named
``<subsystem>_<rest>.yaml`` (it then takes that subsystem's values too,
for its opt targets), else just its stem.

**Opt targets** are the project's, not a study's. Every entry has a
``name``, unique in the project however the entries are spread over
files (so a file can be split or merged without touching anything that
names a target)::

    targets:
    - name: "{subject}_freq"
      helper: qdesignoptimizer.utils.optimization_targets:get_opt_target_res_freq_via_length
      args: {resonator: "resonator_{subject}"}
      tolerance: 0.002          # relative; or {abs: 5.0e6}
    - name: "{subject}_kappa"
      ...
      simulation: capacitance   # the kind of study it is tuned in
      active: false             # carried into a study switched off

When a study is made it takes every target whose modes are all modes of
its MiniStudy (``QDOBackend``); see there for which start switched on.

Names a design defines once -- a component, a mode, a pin -- can come
from the project rather than be typed into every template: an adapter
with a ``template_values()`` hook supplies ``{name: value}`` to every
template (a subsystem's own values win), e.g. ``{tee_ATS1}``.

Two kinds of filling, both done when a study is queued, so the queued
files hold plain values:

* ``{name}`` in any string value: the subsystem's values, plus
  ``{subject}`` (its key). In a ministudy an unknown name is refused; an
  opt target that needs a name the study does not set is not for that
  study.
* a string starting with ``=``: an expression (the ``prop_to``
  evaluator) of ``p`` -- the project's parameter targets -- and ``v`` --
  its design variables, as they stand when the study is queued, e.g.
  ``mode_freq_GHz: "=p['resonator_ATS{n}_freq'] / 1e9"``.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..engine.expressions import Expression, ExpressionError
from ..engine.job import dump_yaml

MINISTUDY_DIR, TARGETS_DIR = "ministudies", "opt_targets"
#: What a ministudy template may hold at its top level: a MiniStudy.
_MINISTUDY_KEYS = {"class", "args", "extra"}

_MINISTUDY_HEADER = """\
MiniStudy template: the MiniStudy alone. Queueing it fills in
  {name}  in any string -- a subsystem's values (plus {subject}, its
          key); with any, it makes one study per subsystem that sets them
  "=..."  an expression of p['<parameter target>'] and
          v['<design variable>'], as they stand when queued
and gives the study every opt target (../opt_targets/) whose modes are
all modes of this MiniStudy."""

_TARGETS_HEADER = """\
Opt targets of the project. Each has a name, unique in the project; a
study takes those whose modes are all modes of its MiniStudy.
Optional per target: tolerance (relative, or {abs: ...}), simulation
(eigenmode / capacitance: the kind of study it is tuned in), active."""


class TemplateError(ValueError):
    """A template cannot be read or filled in."""


class MissingValue(TemplateError):
    """A ``{placeholder}`` nothing sets."""


@dataclass(frozen=True)
class Template:
    """One queueable study: a ministudy template, for one subsystem.

    Attributes:
        key: ``<subject>.<name>``, or ``<name>`` without a subject.
        subject: The subsystem key ("" for none).
        name: The study name.
        ministudy_file: The template file.
        values: The subsystem's placeholder values.
        family: ``capacitance`` when the MiniStudy only runs capacitance
            studies, else ``eigenmode``.
        source: The template, for messages.
    """

    key: str
    subject: str
    name: str
    ministudy_file: Path
    values: dict = field(default_factory=dict)
    family: str = ""
    source: str = ""

    def call_text(self) -> str:
        """What generated it, for ``generated_by``."""
        text = f"{self.ministudy_file.parent.name}/{self.ministudy_file.name}"
        return f"{text} ({self.subject})" if self.subject else text


def target_files(targets_dir: Path) -> list[Path]:
    """The project's opt-target files: every ``*.yaml`` in the folder,
    and ``<folder>.yaml`` next to it (a small project's single file)."""
    targets_dir = Path(targets_dir)
    files = sorted(targets_dir.glob("*.yaml"))
    single = targets_dir.with_suffix(".yaml")
    if single.is_file():
        files.append(single)
    return files


def template_files(ministudy_dir: Path, targets_dir: Path) -> list[Path]:
    """Every template file, for change detection."""
    return sorted(Path(ministudy_dir).glob("*.yaml")) + target_files(targets_dir)


def placeholders_in(value: Any) -> set[str]:
    """Every ``{name}`` in the strings of ``value``, all the way down.

    Raises:
        TemplateError: For a string that is not a valid format string.
    """
    if isinstance(value, dict):
        return set().union(*(placeholders_in(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(placeholders_in(v) for v in value))
    if not isinstance(value, str):
        return set()
    try:
        return {f for _, f, _, _ in string.Formatter().parse(value) if f}
    except ValueError as exc:
        raise TemplateError(f"{value!r}: {exc}") from exc


def discover_templates(
    ministudy_dir: Path,
    subjects: dict[str, dict] | None = None,
    project_values: dict | None = None,
) -> tuple[list[Template], list[str]]:
    """Every study the ministudy templates make, and one line per
    problem.

    Args:
        subjects: ``{subsystem key: its placeholder values}``.
        project_values: Names the project supplies to every template.
    """
    subjects = subjects or {}
    project_values = project_values or {}
    found, problems = [], []
    for path in sorted(Path(ministudy_dir).glob("*.yaml")):
        where = f"{path.parent.name}/{path.name}"
        try:
            data = _read(path)
            extra = sorted(set(data) - _MINISTUDY_KEYS)
            if extra:
                raise TemplateError(
                    f"{extra}: a ministudy template is the MiniStudy "
                    f"alone (class, args, extra)"
                )
            if "class" not in data:
                raise TemplateError("needs class: (the MiniStudy)")
            args = data.get("args") or {}
            family = (
                "capacitance"
                if isinstance(args, dict) and args.get("run_capacitance_studies_only")
                else "eigenmode"
            )
            needed = placeholders_in(data) - set(project_values)
            if not needed:
                subject, name = _split_stem(path.stem, subjects)
                found.append(
                    Template(
                        key=f"{subject}.{name}" if subject else name,
                        subject=subject,
                        name=name,
                        ministudy_file=path,
                        # Its subsystem's values still apply: the opt
                        # targets written for that subsystem are its.
                        values=dict(subjects.get(subject) or {}),
                        family=family,
                        source=str(path),
                    )
                )
                continue
            made = 0
            for subject, values in subjects.items():
                values = dict(values or {})
                if needed <= set(values) | {"subject"}:
                    made += 1
                    found.append(
                        Template(
                            key=f"{subject}.{path.stem}",
                            subject=str(subject),
                            name=path.stem,
                            ministudy_file=path,
                            values=values,
                            family=family,
                            source=str(path),
                        )
                    )
            if not made:
                raise TemplateError(
                    f"uses {sorted(needed)}, which no subsystem sets "
                    f"(subsystems: {sorted(subjects) or 'none'})"
                )
        except TemplateError as exc:
            text = str(exc)
            problems.append(text if text.startswith(where) else f"{where}: {text}")
    return found, problems


def _split_stem(stem: str, subjects: dict) -> tuple[str, str]:
    """``(subject, name)`` of a template without placeholders: the
    longest subsystem key the file name starts with (``squid_kappa`` ->
    ``squid``, ``kappa``), else no subject."""
    for key in sorted(map(str, subjects), key=len, reverse=True):
        if stem.startswith(key + "_") and len(stem) > len(key) + 1:
            return key, stem[len(key) + 1 :]
    return "", stem


def load_targets(targets_dir: Path) -> tuple[list[dict], list[str]]:
    """The project's opt-target entries, in file order, and one line per
    problem: a file that cannot be read, an entry without a ``name``, a
    name used twice in the project.

    An entry with a problem is left out, so the rest stay usable.
    """
    entries, problems = [], []
    seen: dict[str, str] = {}
    for path in target_files(targets_dir):
        where = (
            path.name
            if path.parent != Path(targets_dir)
            else f"{path.parent.name}/{path.name}"
        )
        try:
            data = _read(path)
        except TemplateError as exc:
            problems.append(str(exc))
            continue
        listed = data.get("targets")
        if not isinstance(listed, list):
            problems.append(f"{where}: needs a targets: list")
            continue
        for n, entry in enumerate(listed, 1):
            if not isinstance(entry, dict):
                problems.append(f"{where}: target {n} is not a mapping")
                continue
            name = entry.get("name")
            if not isinstance(name, str) or not name.strip():
                problems.append(f"{where}: target {n} needs a name:")
                continue
            if name in seen:
                problems.append(
                    f"{where}: target name {name!r} is also used in "
                    f"{seen[name]}; names are unique in the project"
                )
                continue
            seen[name] = where
            entries.append(entry)
    return entries, problems


def fill(value: Any, values: dict, p: dict, v: dict, where: str = "") -> Any:
    """Fill placeholders and expressions, all the way down.

    Raises:
        MissingValue: For a placeholder ``values`` does not set.
        TemplateError: For a bad expression.
    """
    if isinstance(value, dict):
        return {k: fill(x, values, p, v, f"{where}.{k}") for k, x in value.items()}
    if isinstance(value, list):
        return [fill(x, values, p, v, f"{where}[{i}]") for i, x in enumerate(value)]
    if not isinstance(value, str):
        return value
    text = _placeholders(value, values, where)
    if text.startswith("="):
        try:
            return Expression(text[1:])(p, value_numbers(v))
        except (ExpressionError, ArithmeticError, ValueError) as exc:
            raise TemplateError(f"{where}: {text!r}: {exc}") from exc
    return text


_WITH_UNIT = re.compile(
    r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*([A-Za-z]*)\s*$"
)


def split_unit(value: Any) -> tuple[float, str] | None:
    """``"400um"`` -> ``(400.0, "um")``, ``5`` -> ``(5.0, "")``; None for
    what is not a number with an optional unit."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value), ""
    match = _WITH_UNIT.match(str(value))
    if not match:
        return None
    return float(match.group(1)), match.group(2)


def value_numbers(v: dict) -> dict:
    """Design variables as numbers, their units dropped, as QDO hands
    them to ``prop_to``: ``"400um"`` is ``400.0``. A value that is not a
    number with a unit stays as it is (and an expression reading it is
    refused).

    Units are dropped, not converted: ``v['a'] + v['b']`` with ``a`` in
    um and ``b`` in mm adds the bare numbers, exactly as a ``prop_to``
    in QDO would. Keep the variables an expression combines in one unit.
    """
    out = {}
    for key, value in (v or {}).items():
        split = split_unit(value)
        out[key] = split[0] if split else value
    return out


def _placeholders(text: str, values: dict, where: str) -> str:
    try:
        names = [f for _, f, _, _ in string.Formatter().parse(text) if f]
    except ValueError as exc:
        raise TemplateError(f"{where}: {text!r}: {exc}") from exc
    if not names:
        return text
    missing = sorted(set(names) - set(values))
    if missing:
        raise MissingValue(
            f"{where}: {text!r} uses {missing}, which is not set (set: "
            f"{sorted(values)})"
        )
    return text.format(**values)


def render(
    template: Template,
    targets: list[dict],
    p: dict,
    v: dict,
    project_values=None,
) -> tuple[dict, list]:
    """The ministudy data for one subsystem, and every project opt
    target that can be filled in with its values (which of them fit the
    study's modes is the backend's to say).

    Args:
        targets: The project's entries (``load_targets``).
        project_values: Named values the project supplies to every
            template (``QDOProject.template_values``); a subsystem's own
            values, then ``{subject}``, take precedence.

    Raises:
        TemplateError: If the ministudy cannot be read or filled, or a
            target's expression fails.
    """
    values = {**(project_values or {}), **template.values}
    if template.subject:
        values["subject"] = template.subject
    ministudy = fill(_read(template.ministudy_file), values, p, v, "ministudy")
    filled = []
    for entry in targets:
        try:
            filled.append(fill(entry, values, p, v, f"target {entry.get('name')}"))
        except MissingValue:
            continue  # written for other subsystems
    return ministudy, filled


def _read(path: Path) -> dict:
    """A template file's mapping; the error names the file."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise TemplateError(
            f"{path.parent.name}/{path.name}: {exc} (a {{placeholder}} "
            f"inside [...] or {{...}} needs quotes)"
        ) from exc
    if not isinstance(data, dict):
        raise TemplateError(f"{path.parent.name}/{path.name}: not a mapping")
    return data


_SAFE = re.compile(r"^[A-Za-z0-9_\-]+$")


def check_name(name: str) -> None:
    """Refuse a template name that is not a plain file stem.

    Raises:
        TemplateError: For anything but letters, digits, ``_`` and ``-``.
    """
    if not _SAFE.match(name or ""):
        raise TemplateError(f"{name!r}: use letters, digits, _ and - only")


def write_ministudy(ministudy_dir: Path, name: str, ministudy: dict) -> Path:
    """Write a study's MiniStudy as ``ministudies/<name>.yaml``.

    The template is for one subsystem, names as they are; turn it into a
    family by putting ``{placeholders}`` where the names differ.

    Raises:
        TemplateError: For a bad name, or a file that already exists.
    """
    check_name(name)
    path = Path(ministudy_dir) / f"{name}.yaml"
    if path.exists():
        raise TemplateError(f"{path.name} already exists; pick another name")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_yaml(ministudy, _MINISTUDY_HEADER), encoding="utf-8")
    return path


def write_targets(targets_dir: Path, name: str, entries: list) -> Path:
    """Write named opt targets as ``opt_targets/<name>.yaml``.

    Raises:
        TemplateError: For a bad name, a file that exists, an entry
            without a name, or a name the project already uses.
    """
    check_name(name)
    path = Path(targets_dir) / f"{name}.yaml"
    if path.exists():
        raise TemplateError(f"{path.name} already exists; pick another name")
    taken = {e["name"] for e in load_targets(targets_dir)[0]}
    names = [e.get("name") for e in entries if isinstance(e, dict)]
    if len(names) != len(entries) or not all(isinstance(n, str) and n for n in names):
        raise TemplateError("every opt target needs a name")
    twice = sorted(
        {n for n in names if isinstance(n, str) and (names.count(n) > 1 or n in taken)}
    )
    if twice:
        raise TemplateError(
            f"opt target name(s) {twice} are already used in the project; "
            f"names are unique"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_yaml({"targets": entries}, _TARGETS_HEADER), encoding="utf-8")
    return path
