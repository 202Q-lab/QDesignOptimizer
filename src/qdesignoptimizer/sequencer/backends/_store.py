"""A store kept as one flat JSON file (QDO's design variables file)."""

from __future__ import annotations

import functools
import json
import os
import tempfile
import threading
import time
from contextlib import contextmanager
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


def store_lock(store: Path):
    """The lock held around every read-modify-write of ``store`` and of
    its record (``.meta.json``), machine-wide: without it a runner's
    write-back and a ``promote`` (or a second runner's) could each read
    the file and the later write drop the earlier's values. It lives in
    the user's sequencer folder, not next to the store, so no lock file
    ends up in a project. (A hand edit in an editor is not covered.)"""
    import hashlib

    from ..engine import state as st
    from ..engine.workspace import user_home

    name = hashlib.sha1(str(Path(store).resolve()).lower().encode("utf-8")).hexdigest()[
        :16
    ]
    return st.file_lock(user_home() / "locks" / f"store-{name}.lock")


def update_json(
    path: Path, values: dict[str, Any], allow_new: bool = False
) -> dict[str, Any]:
    """Write ``values`` into the file atomically; return what changed.
    Holds ``store_lock`` from the read to the write.

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
    with store_lock(path):
        return _update_json(path, values, allow_new)


def _update_json(path: Path, values: dict[str, Any], allow_new: bool) -> dict[str, Any]:
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
    enough: they restart at s001 once ``out/`` is deleted. A damaged
    record (a git merge conflict in it) is set aside as
    ``<name>.damaged-<time>.json`` and a fresh one started, so recording
    does not stop for good.
    """
    with store_lock(store):
        _record_writes(meta_path(store), updates, previous, source, details)


def _record_writes(path, updates, previous, source, details) -> None:
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
        who = latest.get("study") or (
            f"promoted from campaign {latest['promoted_from']}"
            if latest.get("promoted_from")
            else ""
        )
        if not who:
            continue
        kind = (latest.get("simulation") or {}).get("kind")
        text = (
            f"{who}" + (f" ({kind})" if kind else "") + f"  {latest.get('written', '')}"
        )
        if name in current and current[name] != latest.get("value"):
            text = f"edited since; last written by {text}"
        out[name] = text
    return out


class SwitchableStore:
    """The store a backend reads and writes: the project's, or a
    campaign's (``engine/campaign.py``).

    Nothing is switched from outside. A method given a study
    (``check_folder``, ``resolve_inputs``, ``run``, ...) uses that
    study's campaign's store for the length of the call
    (``@uses_job_store``); ``store_values`` and ``commit`` take the
    campaign as an argument; ``note_commit`` reads it from its source.
    So the choice travels with each call, also through a worker process
    that serves several GUI threads one call at a time.
    """

    #: The store's file name; a campaign's own copy has the same name.
    store_file_name = "design_variables.json"

    @property
    def default_store(self) -> Path:
        """The project's own store."""
        raise NotImplementedError

    @property
    def project_root(self) -> Path:
        """The project's folder (``campaigns/`` is in it)."""
        raise NotImplementedError

    def store_for(self, campaign: str = "") -> Path:
        """The store of campaign ``campaign`` ("" for the project)."""
        from ..engine import campaign as cp

        return cp.store_file(
            self.project_root, campaign, self.store_file_name, self.default_store
        )

    @property
    def store_file(self) -> Path:
        """The store of the call under way: its campaign's, or the
        project's."""
        local = self.__dict__.get("_store_local")
        override = getattr(local, "path", None) if local is not None else None
        return override or self.default_store

    @contextmanager
    def store_of(self, campaign: str | None):
        """Use campaign ``campaign``'s store meanwhile, in this thread
        (None: leave it as it is; "": the project's)."""
        if campaign is None:
            yield
            return
        if "_store_local" not in self.__dict__:
            self.__dict__["_store_local"] = threading.local()
        local = self.__dict__["_store_local"]
        previous = getattr(local, "path", None)
        local.path = self.store_for(campaign) if campaign else None
        try:
            yield
        finally:
            local.path = previous


def uses_job_store(method):
    """Run a backend method that is given a study (``job``) against that
    study's campaign's store."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        job = kwargs.get("job")
        if job is None:
            job = next((a for a in args if hasattr(a, "campaign")), None)
        with self.store_of(getattr(job, "campaign", None) or None):
            return method(self, *args, **kwargs)

    return wrapper
