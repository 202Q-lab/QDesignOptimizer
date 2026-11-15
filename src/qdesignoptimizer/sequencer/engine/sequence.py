"""Study sequence files: a named sequence of studies, enqueued in one go.

A study sequence does not run. Enqueueing it writes one study folder per
entry, tagged ``s<NN>-<entry>``; from then on each folder is an ordinary
queued study. Study sequences come from a project's adapter (``sequencer.py``) or
from a YAML file like this::

    version: 1
    name: Revalidate every parameter
    campaign: tuneup_chip       # optional: queue its studies in a campaign
    why_ordered: >-
      Linewidths first: 2-4 min capacitance studies surface a broken
      model before an eigenmode iteration commits 40 min.
    defaults:
      settings: {iterations: 2}
      checks: [mode_identity]
      save: true
      on_failure: continue
    studies:
    - study: snail_chain.kappa
    - {study: ATS1.eigenmode, settings: {iterations: 3}}

Entries take the backend's word (``study:``) or ``task:``; anything an
entry sets wins over ``defaults``, ``settings`` key by key.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

import yaml

from .backend import SequenceEntry, SequenceSpec, Words
from .job import ON_FAILURE, dump_yaml, numbers

_TOP = {
    "version",
    "name",
    "why_ordered",
    "defaults",
    "studies",
    "notes",
    "campaign",
}
_ENTRY = {"settings", "checks", "save", "on_failure", "notes"}


class SequenceError(ValueError):
    """A study sequence file cannot be used. Carries every problem found."""

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


def load_sequence(path: Path, words: Words = Words()) -> SequenceSpec:
    """Read a study sequence file.

    Raises:
        SequenceError: Listing every problem.
    """
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise SequenceError([f"{path.name}: not valid YAML: {exc}"]) from exc
    return parse_sequence(raw, path.stem, words, source=str(path))


def parse_sequence(
    raw, key: str, words: Words = Words(), source: str = ""
) -> SequenceSpec:
    """A study sequence from its file's mapping, named ``key``.

    Raises:
        SequenceError: Listing every problem.
    """
    path = Path(source or f"{key}.yaml")
    problems: list[str] = []
    if not isinstance(raw, dict):
        raise SequenceError([f"{path.name} must hold a mapping"])
    extra = sorted(set(raw) - _TOP)
    if extra:
        problems.append(f"unknown top-level key(s) {extra}")
    defaults = raw.get("defaults") or {}
    if not isinstance(defaults, dict):
        problems.append("defaults must be a mapping")
        defaults = {}
    elif not isinstance(defaults.get("settings") or {}, dict):
        problems.append("defaults: settings must be a mapping")
        defaults = {k: v for k, v in defaults.items() if k != "settings"}
    bad = sorted(set(defaults) - _ENTRY)
    if bad:
        problems.append(f"defaults cannot set {bad}")
    campaign = raw.get("campaign") or ""
    if not isinstance(campaign, str):
        problems.append("campaign must be a campaign's name")
        campaign = ""
    studies = raw.get("studies") or []
    if not isinstance(studies, list):
        problems.append("studies must be a list")
        studies = []
    entries = []
    for n, item in enumerate(studies, 1):
        if isinstance(item, str):
            item = {words.task: item}
        if not isinstance(item, dict):
            problems.append(f"entry {n} is not a mapping")
            continue
        if not isinstance(item.get("settings") or {}, dict):
            problems.append(f"entry {n}: settings must be a mapping")
            continue
        keys = [k for k in ("task", words.task) if k in item]
        unknown = sorted(set(item) - _ENTRY - {"task", words.task})
        if unknown:
            problems.append(f"entry {n}: unknown key(s) {unknown}")
        if len(set(keys)) != 1:
            problems.append(f"entry {n}: name exactly one {words.task!r}")
            continue
        merged = {**defaults, **item}
        settings = numbers(
            {
                **(defaults.get("settings") or {}),
                **(item.get("settings") or {}),
            }
        )
        on_failure = merged.get("on_failure")
        if on_failure is not None and on_failure not in ON_FAILURE:
            problems.append(f"entry {n}: on_failure {on_failure!r}")
        save = merged.get("save")
        if save is not None and not isinstance(save, bool):
            problems.append(f"entry {n}: save must be true or false")
        checks = merged.get("checks")
        if isinstance(checks, str):
            # One name, not its letters: tuple("mode_identity") would
            # give every study the checks 'm', 'o', 'd', ...
            checks = [checks]
        if checks is not None and (
            not isinstance(checks, list) or not all(isinstance(c, str) for c in checks)
        ):
            problems.append(f"entry {n}: checks must be a list of names")
            continue
        entries.append(
            SequenceEntry(
                task=str(item[keys[0]]),
                settings=settings,
                checks=tuple(checks) if checks is not None else None,
                save=save,
                on_failure=on_failure,
                notes=str(merged.get("notes") or ""),
            )
        )
    if not entries:
        problems.append(f"{path.name} lists no studies")
    if problems:
        raise SequenceError([f"{path.name}: {p}" for p in problems])
    return SequenceSpec(
        key=key,
        label=str(raw.get("name") or key),
        entries=tuple(entries),
        why_ordered=str(raw.get("why_ordered") or ""),
        source=source,
        campaign=campaign.replace("\\", "/").strip("/"),
    )


def load_sequence_dir(
    folder: Path, words: Words = Words()
) -> tuple[list[SequenceSpec], list[str]]:
    """Every ``*.yaml`` study sequence in a folder, and the problems of those
    that could not be read (so one bad file does not hide the rest)."""
    specs: list[SequenceSpec] = []
    problems: list[str] = []
    folder = Path(folder)
    if not folder.is_dir():
        return specs, problems
    for path in sorted(folder.glob("*.yaml")):
        try:
            specs.append(load_sequence(path, words))
        except SequenceError as exc:
            problems.extend(exc.problems)
        except Exception as exc:  # noqa: BLE001 -- one file, not the catalog
            problems.append(f"{path.name}: cannot be read: {exc}")
    return specs, problems


_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def check_sequence_key(key: str) -> None:
    """Refuse a file name that is not letters, digits, ``_`` and ``-``.

    Raises:
        SequenceError: Saying so.
    """
    if not _KEY.fullmatch(str(key)):
        raise SequenceError(
            [f"{key!r}: a study sequence name is letters, digits, _ and -"]
        )


def save_sequence(
    folder: Path,
    key: str,
    raw: dict,
    words: Words = Words(),
    overwrite: bool = False,
) -> Path:
    """Write study sequence ``key`` as ``<folder>/<key>.yaml``, after
    checking it by the same rules a file is read by.

    Args:
        raw: The file's mapping (``name``, ``why_ordered``,
            ``campaign``, ``defaults``, ``studies``, ...).
        overwrite: Replace an existing file (editing it); otherwise one
            with this name is refused.

    Raises:
        SequenceError: For a bad name, an existing file, or content that
            would not read back.
    """
    check_sequence_key(key)
    folder = Path(folder)
    path = folder / f"{key}.yaml"
    if path.exists() and not overwrite:
        raise SequenceError([f"{path.name} exists already; pick another name"])
    raw = {"version": 1, **{k: v for k, v in raw.items() if k != "version"}}
    parse_sequence(raw, key, words, source=str(path))  # raises what is wrong
    folder.mkdir(parents=True, exist_ok=True)
    # A temp file of its own, so two saves at once never share one.
    fd, name = tempfile.mkstemp(prefix=f".{key}.", suffix=".tmp", dir=folder)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(dump_yaml(raw))
        if overwrite:
            os.replace(tmp, path)
        else:
            try:
                # Fails if a save of the same name won the race.
                os.link(tmp, path)
            except FileExistsError as exc:
                raise SequenceError(
                    [f"{path.name} exists already; pick another name"]
                ) from exc
    finally:
        tmp.unlink(missing_ok=True)
    return path
