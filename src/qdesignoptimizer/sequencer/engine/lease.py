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
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from . import state as st

LEASE_FILE = ".editing.json"
#: Seconds a lease lasts unless renewed.
LEASE_S = 15.0


class LeaseHeld(RuntimeError):
    """Someone else is editing the study."""


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


def take_lease(folder: Path, by: str, seconds: float = LEASE_S) -> None:
    """Take or renew the lease on a study for ``by``.

    Raises:
        LeaseHeld: If someone else holds a live lease.
        FileNotFoundError: If the study folder is gone.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    from .workspace import user_home  # pylint: disable=import-outside-toplevel

    with st.file_lock(user_home() / "locks" / "lease.lock", wait=10.0):
        held = read_lease(folder)
        if held and held.get("by") != by:
            raise LeaseHeld(f"being edited by {describe(held)}")
        since = held.get("since") if held else st.now()
        st.write_json(
            folder / LEASE_FILE,
            {"by": by, "since": since, "until": time.time() + seconds},
        )


def release_lease(folder: Path, by: str) -> None:
    """Drop ``by``'s lease on a study (nothing if it holds none)."""
    folder = Path(folder)
    held = read_lease(folder)
    if held is not None and held.get("by") != by:
        return
    try:
        os.remove(folder / LEASE_FILE)
    except OSError:
        pass
