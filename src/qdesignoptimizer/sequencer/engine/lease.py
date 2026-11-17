"""Who is editing a study: ``<study>/.editing.json``.

The GUI and an agent at the command line may both change a queued (or
held) study. While one of them edits it, it holds a lease -- a small file
in the study's folder saying who and until when -- and the other is
refused instead of the two overwriting each other::

    {"by": "you (GUI)", "since": "2026-11-16T10:02:11", "until": 1794820951.2}

A lease is short (``LEASE_S``) and renewed by its holder while it edits
(the GUI on every poll), so a closed GUI or a crashed command leaves
nothing that blocks for long. It is a dot-file: the queue, the runner's
copies and the GUI's change detection all skip it.

``by`` is who is shown; ``holder`` tells holders apart that show the
same name (two browser tabs of one GUI, two agents both called
``claude``): each GUI tab and each command has its own, so they keep
each other out too. A lease without one is held by its ``by``.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from . import state as st

LEASE_FILE = ".editing.json"
#: Seconds a lease lasts unless renewed.
LEASE_S = 15.0


class LeaseHeld(RuntimeError):
    """Someone else is editing the study (``lease``: theirs)."""

    def __init__(self, message: str, lease: dict | None = None):
        super().__init__(message)
        self.lease = dict(lease or {})


def holder_of(lease: dict) -> str:
    """Who holds a lease (its ``holder``, else its ``by``)."""
    return str(lease.get("holder") or lease.get("by") or "")


@contextmanager
def lease_guard():
    """Held while a lease is read and acted on (taken, or the study
    moved): no one takes it in between."""
    from .workspace import user_home  # pylint: disable=import-outside-toplevel

    with st.file_lock(user_home() / "locks" / "lease.lock", wait=10.0):
        yield


def read_lease(folder: Path) -> dict | None:
    """The live lease on a study, or None (none, expired or unreadable)."""
    path = Path(folder) / LEASE_FILE
    try:
        data = st.read_json(path)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or float(data.get("until") or 0) < time.time():
        return None
    return data


def describe(lease: dict) -> str:
    """``claude (since 10:02:11)``."""
    since = str(lease.get("since") or "")
    return f"{lease.get('by', '?')}" + (f" (since {since[11:19]})" if since else "")


def take_lease(
    folder: Path, by: str, seconds: float = LEASE_S, holder: str = ""
) -> None:
    """Take or renew the lease on a study for ``holder`` (``by`` when
    not given), shown as ``by``.

    Raises:
        LeaseHeld: If someone else holds a live lease.
        FileNotFoundError: If the study folder is gone.
        state.LockHeld: If the lease lock stays held for 10 s.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    holder = holder or by
    with lease_guard():
        held = read_lease(folder)
        if held and holder_of(held) != holder:
            raise LeaseHeld(f"being edited by {describe(held)}", held)
        since = held.get("since") if held else st.now()
        _write_in(
            folder,
            {
                "by": by,
                "holder": holder,
                "since": since,
                "until": time.time() + seconds,
            },
        )


def check_free(folder: Path, holder: str) -> None:
    """Refuse when someone other than ``holder`` ("" for no one) holds
    a live lease on the study.

    Raises:
        LeaseHeld: If they do.
    """
    held = read_lease(folder)
    if held and holder_of(held) != holder:
        raise LeaseHeld(f"being edited by {describe(held)}", held)


def drop_lease(folder: Path) -> None:
    """Remove a study's lease file, whoever's: it moved, and the lease
    was for where it was."""
    try:
        os.remove(Path(folder) / LEASE_FILE)
    except OSError:
        pass


def _write_in(folder: Path, data: dict) -> None:
    """Write the lease file into ``folder`` -- only if it is still there.
    (``state.write_json`` would make the folder anew, and a runner that
    just took the study could not put it back next to that.)

    Raises:
        FileNotFoundError: If the folder is gone.
    """
    tmp = folder / f".{LEASE_FILE}.tmp"
    tmp.write_text(json.dumps(data), encoding="utf-8")  # no folder: raises
    for attempt in range(40):
        try:
            os.replace(tmp, folder / LEASE_FILE)
            return
        except PermissionError:  # a reader has it open this instant
            if attempt == 39:
                raise
            time.sleep(0.05)


def release_lease(folder: Path, holder: str) -> None:
    """Drop ``holder``'s lease on a study (nothing if it holds none)."""
    folder = Path(folder)
    held = read_lease(folder)
    if held is not None and holder_of(held) != holder:
        return
    try:
        os.remove(folder / LEASE_FILE)
    except OSError:
        pass
