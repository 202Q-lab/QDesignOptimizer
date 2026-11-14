"""Study sequence files: a named sequence of studies, enqueued in one go.

A study sequence does not run. Enqueueing it writes one study folder per
entry, tagged ``s<NN>-<entry>``; from then on each folder is an ordinary
queued study. Study sequences come from a project's adapter (``sequencer.py``) or
from a YAML file like this::

    version: 1
    name: Revalidate every parameter
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

from pathlib import Path

import yaml

from .backend import SequenceEntry, SequenceSpec, Words
from .job import ON_FAILURE, numbers

_TOP = {"version", "name", "why_ordered", "defaults", "studies", "notes"}
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
    problems: list[str] = []
    if not isinstance(raw, dict):
        raise SequenceError([f"{path.name} must hold a mapping"])
    extra = sorted(set(raw) - _TOP)
    if extra:
        problems.append(f"unknown top-level key(s) {extra}")
    defaults = raw.get("defaults") or {}
    bad = sorted(set(defaults) - _ENTRY)
    if bad:
        problems.append(f"defaults cannot set {bad}")
    entries = []
    for n, item in enumerate(raw.get("studies") or [], 1):
        if isinstance(item, str):
            item = {words.task: item}
        if not isinstance(item, dict):
            problems.append(f"entry {n} is not a mapping")
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
        key=path.stem,
        label=str(raw.get("name") or path.stem),
        entries=tuple(entries),
        why_ordered=str(raw.get("why_ordered") or ""),
        source=str(path),
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
    return specs, problems
