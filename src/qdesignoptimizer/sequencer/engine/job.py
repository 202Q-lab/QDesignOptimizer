"""``job.yaml``: the queue entry of one study, and how it runs.

A study is its whole folder (for QDO: a MiniStudy, opt targets, the
values it ran with). This file is the part the engine reads: which
study it is, where it belongs and how the queue treats it. Example::

    version: 1
    id: s07-revalidate-ats1_eigenmode
    study: ATS1.eigenmode            # the study key (or ``task:``)
    subsystem: ATS1                  # (or ``subject:``)
    sequence: {id: s07, name: revalidate, entry: 3}
    settings: {iterations: 2}
    checks: [mode_identity]
    save: true                       # write tuned values back on success
    on_failure: continue             # abort (default) or continue
    hold: true                       # on hold: the runner stops here
    notes: ""
    created: "2026-11-11 10:00:00"
    generated_by: ministudies/ATS1_eigenmode.yaml

Everything the backend needs besides these lives in the backend's own
files next to it (``ministudy.yaml``, ``opt_targets.yaml``, ...).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .backend import Words

JOB_FILE = "job.yaml"
#: The file's name before 2026-11-12; still read, never written.
OLD_JOB_FILE = "study.yaml"
VERSION = 1
ON_FAILURE = ("abort", "continue")
_KEYS = {
    "version",
    "id",
    "settings",
    "checks",
    "save",
    "on_failure",
    "hold",
    "notes",
    "created",
    "generated_by",
    "sequence",
    "requeued_from",
    "requeued_as",
}


class JobError(ValueError):
    """``job.yaml`` cannot be used. Carries every problem found."""

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


@dataclass
class Job:
    """One parsed ``job.yaml``. ``raw`` is the mapping as in the file."""

    id: str
    task: str
    subject: str = ""
    settings: dict = field(default_factory=dict)
    checks: list[str] = field(default_factory=list)
    save: bool = False
    on_failure: str = "abort"
    hold: bool = False
    notes: str = ""
    sequence: dict = field(default_factory=dict)
    created: str = ""
    generated_by: str = ""
    raw: dict = field(default_factory=dict, repr=False)

    def to_raw(self, words: Words = Words()) -> dict:
        """The mapping to write, in a fixed, readable order."""
        out: dict[str, Any] = {
            "version": VERSION,
            "id": self.id,
            words.task: self.task,
        }
        if self.subject:
            out[words.subject] = self.subject
        if self.sequence:
            out["sequence"] = dict(self.sequence)
        out["settings"] = dict(self.settings)
        out["checks"] = list(self.checks)
        out["save"] = bool(self.save)
        out["on_failure"] = self.on_failure
        if self.hold:
            out["hold"] = True
        out["notes"] = self.notes
        if self.created:
            out["created"] = self.created
        if self.generated_by:
            out["generated_by"] = self.generated_by
        for key in ("requeued_from", "requeued_as"):
            if self.raw.get(key):
                out[key] = self.raw[key]
        return out


def parse_job(raw: Any, words: Words = Words(), folder_id: str = "") -> Job:
    """Parse a ``job.yaml`` mapping.

    Raises:
        JobError: Listing every problem -- and only JobError, whatever
            the file holds, so one malformed study cannot take a caller
            (the GUI's queue view) down with it.
    """
    try:
        return _parse_job(raw, words, folder_id)
    except JobError:
        raise
    except Exception as exc:  # noqa: BLE001 -- any malformed content
        raise JobError(
            [f"{JOB_FILE} cannot be read: {type(exc).__name__}: {exc}"]
        ) from exc


def _parse_job(raw: Any, words: Words, folder_id: str) -> Job:
    if not isinstance(raw, dict):
        raise JobError([f"{JOB_FILE} must hold a mapping"])
    problems: list[str] = []
    task_keys = [k for k in ("task", words.task) if k in raw]
    subject_keys = [k for k in ("subject", words.subject) if k in raw]
    known = _KEYS | {"task", words.task, "subject", words.subject}
    extra = sorted(set(raw) - known)
    if extra:
        problems.append(f"unknown key(s) {extra}")
    if raw.get("version", VERSION) != VERSION:
        problems.append(
            f"version {raw.get('version')!r}: this engine " f"reads version {VERSION}"
        )
    if len(set(task_keys)) != 1:
        problems.append(f"name exactly one {words.task!r}")
    task = str(raw[task_keys[0]]) if task_keys else ""
    job_id = str(raw.get("id") or folder_id)
    if folder_id and job_id != folder_id:
        problems.append(f"id {job_id!r} does not match its folder " f"{folder_id!r}")
    settings = raw.get("settings") or {}
    if not isinstance(settings, dict):
        problems.append("settings must be a mapping")
        settings = {}
    settings = numbers(settings)
    checks = raw.get("checks") or []
    if isinstance(checks, str):
        checks = [checks]
    if not isinstance(checks, list) or not all(isinstance(c, str) for c in checks):
        problems.append(f"checks must be a list of names, not {checks!r}")
        checks = []
    for key in ("notes", "created", "generated_by"):
        if raw.get(key) is not None and not isinstance(raw.get(key), str):
            problems.append(f"{key} must be text")
    hold = raw.get("hold", False)
    if not isinstance(hold, bool):
        problems.append(f"hold must be true or false, not {hold!r}")
        hold = False
    save = raw.get("save", False)
    if not isinstance(save, bool):
        problems.append(f"save must be true or false, not {save!r}")
        save = False
    on_failure = raw.get("on_failure", "abort")
    if on_failure not in ON_FAILURE:
        problems.append(
            f"on_failure {on_failure!r} is not one of " f"{list(ON_FAILURE)}"
        )
    sequence = raw.get("sequence") or {}
    if not isinstance(sequence, dict):
        problems.append("study sequence must be a mapping")
        sequence = {}
    if problems:
        raise JobError(problems)
    return Job(
        id=job_id,
        task=task,
        subject=str(raw[subject_keys[0]]) if subject_keys else "",
        settings=dict(settings),
        checks=[str(c) for c in checks],
        save=save,
        on_failure=on_failure,
        hold=hold,
        notes=str(raw.get("notes") or ""),
        sequence=dict(sequence),
        created=str(raw.get("created") or ""),
        generated_by=str(raw.get("generated_by") or ""),
        raw=dict(raw),
    )


_NUMBER = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


#: The task of a study made from scratch (no template): ``custom.<name>``.
CUSTOM = "custom"


def slug_name(text: str) -> str:
    """A name safe in a task key and a file name: letters, digits, _."""
    return re.sub(r"[^0-9A-Za-z_]+", "_", str(text or "")).strip("_")


def is_custom(task: str) -> bool:
    """Whether a task names a study made from scratch."""
    return str(task).startswith(CUSTOM + ".")


def numbers(value: Any) -> Any:
    """Read number-looking strings as numbers, all the way down.

    PyYAML follows YAML 1.1, where ``1e-3`` and ``4.9e9`` (no dot or no
    sign in the exponent) are strings. A setting written that way would
    reach the solver as text, so settings are passed through this.
    """
    if isinstance(value, dict):
        return {k: numbers(v) for k, v in value.items()}
    if isinstance(value, list):
        return [numbers(v) for v in value]
    if isinstance(value, str) and _NUMBER.match(value.strip()):
        text = value.strip()
        if re.fullmatch(r"[+-]?\d+", text):
            return int(text)
        return float(text)
    return value


_EXPONENT = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)[eE][+-]?\d+$")


def exponent_floats(value: Any) -> Any:
    """Read exponent-form strings (``1e-3``, ``4.9e9``) as floats, all
    the way down, and nothing else.

    Those are what PyYAML (YAML 1.1) leaves as text although they were
    written as numbers. Unlike ``numbers`` this keeps ``"15"`` or
    ``"2025"`` text: in an object's arguments a string of digits can be a
    name (a component, a pin, a design name) and must stay one.
    """
    if isinstance(value, dict):
        return {k: exponent_floats(v) for k, v in value.items()}
    if isinstance(value, list):
        return [exponent_floats(v) for v in value]
    if isinstance(value, str) and _EXPONENT.match(value.strip()):
        return float(value.strip())
    return value


def job_file(folder: Path) -> Path:
    """A study folder's queue-entry file: ``job.yaml``, or the
    ``study.yaml`` of a folder written before the rename when that is
    the one it has."""
    folder = Path(folder)
    old = folder / OLD_JOB_FILE
    if old.exists() and not (folder / JOB_FILE).exists():
        return old
    return folder / JOB_FILE


def load_job(folder: Path, words: Words = Words()) -> Job:
    """Read ``<folder>/job.yaml``.

    Raises:
        JobError: If it is missing or not valid.
    """
    path = job_file(folder)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise JobError([f"{path.parent.name} has no {JOB_FILE}"]) from exc
    except yaml.YAMLError as exc:
        raise JobError([f"{JOB_FILE} is not valid YAML: {exc}"]) from exc
    return parse_job(raw, words, folder_id=_folder_id(Path(folder)))


def _folder_id(folder: Path) -> str:
    """A queued folder is named by its id; a completed one carries it
    before its start time (``parse_completed``)."""
    name = folder.name
    if name.startswith(".claimed-"):
        return name[len(".claimed-") :]
    return parse_completed(name)[0]


FAILED = "_failed"
#: A start time as folder names carry it (``state.stamp``), with the
#: ``-2``, ``-3`` that keeps two starts in one second apart.
_STAMP = r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}(?:-\d+)?"
_COMPLETED = re.compile(rf"^(?P<id>.+)_(?P<stamp>{_STAMP})$")
_OLD = re.compile(rf"^ministudy_(?P<stamp>{_STAMP})_(?P<id>.+)$")


def completed_name(job_id: str, stamp: str) -> str:
    """``<id>_<start>``, e.g. ``s01-tune_ATS1-ats1_kappa_2026-11-11T
    14-02-33``: a name sort groups each study sequence's studies."""
    return f"{job_id}_{stamp}"


def parse_completed(name: str) -> tuple[str, str]:
    """``(id, start)`` of a completed folder's name (``_failed`` or not).

    The start time has a fixed shape, so it is found from the right
    although ids hold underscores. Reads the names used before
    2026-11-12 too (``ministudy_<start>_<id>``); a name that is neither
    is all id.
    """
    base = name[: -len(FAILED)] if name.endswith(FAILED) else name
    old = _OLD.match(base)
    if old:
        return old.group("id"), old.group("stamp")
    new = _COMPLETED.match(base)
    if new:
        return new.group("id"), new.group("stamp")
    return base, ""


def is_completed_name(name: str) -> bool:
    """Whether a folder in ``completed/`` is a study's."""
    return bool(parse_completed(name)[1])


def dump_yaml(data: Any, header: str | None = None) -> str:
    """YAML text, block style, keys in the given order, with a comment
    header (each line prefixed ``# ``)."""
    text = yaml.safe_dump(
        data,
        sort_keys=False,
        allow_unicode=False,
        width=79,
        default_flow_style=False,
    )
    if header:
        lines = "".join(
            f"# {line}\n" if line else "#\n" for line in header.splitlines()
        )
        text = lines + text
    return text
