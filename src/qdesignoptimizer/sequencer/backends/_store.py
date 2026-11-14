"""A store kept as one flat JSON file (QDO's design variables file)."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

_TRIES = 50  # ~5 s at 0.1 s


def read_json(path: Path) -> dict[str, Any]:
    """The file's mapping, or {} when it does not exist.

    Retries while Windows refuses the read (the file being replaced
    this instant by another process).
    """
    path = Path(path)
    for attempt in range(_TRIES):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (PermissionError, json.JSONDecodeError):
            if attempt == _TRIES - 1:
                raise
            time.sleep(0.1)
    return {}


def update_json(
    path: Path, values: dict[str, Any], allow_new: bool = False
) -> dict[str, Any]:
    """Write ``values`` into the file atomically; return what changed.

    The replace is retried: on Windows ``os.replace`` fails while any
    process has the target open, and the queue GUI reads the design
    variables file every poll -- without the retry a solved study's
    write-back failed whenever the two met (found in review).

    Args:
        path: The JSON file.
        values: Key -> new value.
        allow_new: Accept keys the file does not hold yet. Off for a
            design variables file, where a new key is a typo that QDO
            would silently ignore.

    Raises:
        KeyError: For a key the file lacks, unless ``allow_new``.
        PermissionError: If the file stayed locked for ~5 s; the file is
            then unchanged and no temp file is left behind.
    """
    path = Path(path)
    current = read_json(path)
    unknown = sorted(set(values) - set(current))
    if unknown and not allow_new:
        raise KeyError(f"{path.name} has no {unknown}")
    changed = {k: v for k, v in values.items() if current.get(k) != v}
    current.update(values)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}-", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(current, handle, indent=4)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())  # never an empty store after a power cut
    try:
        for attempt in range(_TRIES):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if attempt == _TRIES - 1:
                    raise
                time.sleep(0.1)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return changed


# -- who wrote what ----------------------------------------------------------
#
# ``<store>.meta.json`` next to the store (``design_variables.json`` ->
# ``design_variables.meta.json``): per value the study that last wrote
# it back, and the writes before that. Only values a study tuned are
# ever written back (its switched-on opt targets' design variables that
# moved), so only those are recorded. Kept beside the store, so it is
# committed with it.

#: Writes kept per value.
HISTORY = 20


def _read_record(path: Path):
    """The record's mapping; {} when there is none; None when it is
    damaged (not JSON, or not a mapping). No waiting on a damaged file:
    it is replaced atomically, so it is never seen half-written."""
    for attempt in range(_TRIES):
        try:
            text = path.read_text(encoding="utf-8")
            break
        except FileNotFoundError:
            return {}
        except PermissionError:
            if attempt == _TRIES - 1:
                raise
            time.sleep(0.1)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def meta_path(store: Path) -> Path:
    """The record of a store's write-backs."""
    store = Path(store)
    return store.with_name(f"{store.stem}.meta.json")


def read_meta(store: Path) -> dict[str, Any]:
    """``{value name: {"latest": {...}, "history": [...]}}`` ({} for a
    damaged file, e.g. a git merge left in it)."""
    return _read_record(meta_path(store)) or {}


def record_writes(
    store: Path,
    updates: dict[str, Any],
    previous: dict[str, Any],
    source: dict[str, Any],
    details: dict[str, dict] | None = None,
) -> None:
    """Note a write-back: each value, what it replaced, and the study
    (``source``: id, task, study sequence, folder, settings, time) that wrote
    it, plus ``details[name]`` where the backend has more to say about
    that one value (the simulation, the target it chased).

    The same run (same study folder, which carries its start time)
    writing the same value again is not a second write (a recovered
    write-back already recorded) and is skipped. Ids alone are not
    enough: they restart at s01 once ``out/`` is deleted. A damaged
    record (a git merge conflict in it) is set aside as
    ``<name>.damaged-<time>.json`` and a fresh one started, so recording
    does not stop for good.
    """
    path = meta_path(store)
    data = _read_record(path)
    if data is None:
        aside = path.with_name(
            f"{path.stem}.damaged-{time.strftime('%Y%m%dT%H%M%S')}.json"
        )
        os.replace(path, aside)
        data = {}
    for name, value in updates.items():
        item = data.get(name) or {}
        latest = item.get("latest") or {}
        same_run = latest.get("folder") == source.get("folder") and (
            latest.get("study") == source.get("study")
        )
        if same_run and latest.get("value") == value:
            continue
        entry = {
            "value": value,
            "previous": previous.get(name),
            **source,
            **((details or {}).get(name) or {}),
        }
        history = (list(item.get("history") or []) + [entry])[-HISTORY:]
        data[name] = {"latest": entry, "history": history}
    _write(path, data)


def _write(path: Path, data: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}-", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, indent=4)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        for attempt in range(_TRIES):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == _TRIES - 1:
                    raise
                time.sleep(0.1)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def sources_text(store: Path) -> dict[str, str]:
    """``{value name: "<study>  <time>"}`` for the GUI; a value that no
    longer is what that study wrote (edited by hand since) says so."""
    current = read_json(store)
    out = {}
    for name, item in read_meta(store).items():
        latest = (item or {}).get("latest") or {}
        if not latest.get("study"):
            continue
        kind = (latest.get("simulation") or {}).get("kind")
        text = (
            f"{latest['study']}"
            + (f" ({kind})" if kind else "")
            + f"  {latest.get('written', '')}"
        )
        if name in current and current[name] != latest.get("value"):
            text = f"edited since; last written by {text}"
        out[name] = text
    return out
