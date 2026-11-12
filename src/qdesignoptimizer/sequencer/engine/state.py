"""Files that say where things stand. Nothing lives only in memory.

``<out>/runner.json``   the runner: alive, idle, which study it is on
``<out>/runner.lock``   one runner per project at a time
``<out>/runner.log``    everything the runner logged
``<out>/STOP``          present = stop once the current study is done
``<study>/result.json`` one study's outcome (running / done / failed)

The runner writes them, atomically; the GUI, a shell or an agent over
Remote Control read the same files, so closing the GUI never orphans a
solve and a run started from a terminal looks the same as one started
from the GUI.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

RUNNER_FILE, LOCK_FILE, LOG_FILE, STOP_FILE = (
    "runner.json",
    "runner.lock",
    "runner.log",
    "STOP",
)
RESULT_FILE = "result.json"

#: Held around every read-modify-write of an ``order.json`` in this
#: process: the GUI's launch thread and its request threads would
#: otherwise each read the old order and the second write drop the
#: first's studies from it. (Re-entrant: a workspace reorder writes
#: each project's order inside its own.)
ORDER_LOCK = threading.RLock()

#: Study statuses (``queued`` is never written: a folder in queue/ is).
QUEUED, RUNNING, DONE, FAILED = "queued", "running", "done", "failed"
#: Written just before the store is written back, so a runner that dies
#: there leaves a record the next one can settle against the store.
COMMITTING = "committing"
#: Runner statuses.
IDLE, BUSY, FINISHED, STOPPED, ABORTED, ERROR, HELD = (
    "idle",
    "running",
    "finished",
    "stopped",
    "aborted",
    "error",
    "held",
)


def now() -> str:
    """Local time to the second, as the files write it."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def stamp() -> str:
    """A start time for a folder name: ``2026-11-11T14-02-33``."""
    return time.strftime("%Y-%m-%dT%H-%M-%S")


def write_json(path: Path, data: Any, seconds: float = 1.0) -> None:
    """Write JSON atomically (temp file + ``os.replace``).

    Windows refuses ``os.replace`` while a reader holds the target (the
    GUI polls, the virus scanner looks), so it retries for ``seconds``;
    on giving up the temp file is removed and the error raised.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}-", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, indent=2, default=_jsonable)
        handle.write("\n")
    deadline = time.monotonic() + seconds
    while True:
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if time.monotonic() > deadline:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
            time.sleep(0.05)


def read_json(path: Path) -> Any | None:
    """Read JSON, or None when the file is not there."""
    path = Path(path)
    for attempt in range(5):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (PermissionError, json.JSONDecodeError):
            if attempt == 4:
                raise
            time.sleep(0.05)
    return None


@dataclass
class RunnerState:
    """``runner.json``."""

    status: str = IDLE
    mode: str = ""
    pid: int = 0
    pid_started: float = 0.0
    started: str = ""
    finished: str = ""
    current: str = ""
    message: str = ""
    done: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, out_dir: Path) -> RunnerState:
        """The runner state, or an idle one when there is none."""
        data = read_json(Path(out_dir) / RUNNER_FILE) or {}
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, out_dir: Path) -> None:
        """Write it, waiting as long as for a ``result.json``: the GUI
        polls this file, and giving up would end the whole runner."""
        write_json(Path(out_dir) / RUNNER_FILE, asdict(self), 30.0)

    def alive(self, out_dir: Path) -> bool:
        """Whether a runner is going: this state says so AND the runner
        lock is held. A runner that died leaves ``running`` behind; the
        operating system has released its lock, so it reads as dead."""
        return self.status == BUSY and lock_held(out_dir)


def result_of(folder: Path) -> dict:
    """A study's ``result.json``, {} when it has none."""
    return read_json(Path(folder) / RESULT_FILE) or {}


def request_stop(out_dir: Path) -> None:
    """Ask the runner to stop once the current study is done."""
    (Path(out_dir) / STOP_FILE).write_text(now(), encoding="utf-8")


def stop_requested(out_dir: Path) -> bool:
    """Whether a stop was asked for."""
    return (Path(out_dir) / STOP_FILE).exists()


def clear_stop(out_dir: Path) -> None:
    """Forget an earlier stop request."""
    try:
        (Path(out_dir) / STOP_FILE).unlink()
    except FileNotFoundError:
        pass


# -- the one-runner lock ---------------------------------------------------
#
# An operating-system lock on ``runner.lock`` (msvcrt on Windows, flock
# elsewhere), held for as long as the runner's process lives. The system
# drops it when the process ends however it ends -- a crash, Task
# Manager, a power cut -- so a lock can never be left behind, and the
# check needs neither psutil nor a pid. The file itself stays on disk and
# carries nothing; who holds it is in ``runner.json``.


class LockHeld(RuntimeError):
    """Another live process holds the runner lock."""


class RunnerLock:
    """A held runner lock; ``release_lock`` gives it back."""

    def __init__(self, path: Path, handle):
        self.path = path
        self.handle = handle


def _try_lock(handle) -> bool:
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl  # pylint: disable=import-error

            fcntl.flock(  # type: ignore[attr-defined]
                handle.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,  # type: ignore[attr-defined]
            )
        return True
    except OSError:
        return False


def _unlock(handle) -> None:
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl  # pylint: disable=import-error

            fcntl.flock(  # type: ignore[attr-defined]
                handle.fileno(), fcntl.LOCK_UN  # type: ignore[attr-defined]
            )
    except OSError:
        pass


def acquire_lock(out_dir: Path, wait: float = 1.0) -> RunnerLock:
    """Take the one-runner lock, trying for up to ``wait`` seconds.

    Of two starts racing (a double click, the GUI and a shell) the
    operating system lets exactly one have it. The short wait is for
    ``lock_held``: the GUI's check takes the lock for an instant to see
    whether it is free, and a start landing in that instant must not be
    refused (3 in 500 were, in review).

    Raises:
        LockHeld: If another process holds it the whole time.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / LOCK_FILE
    handle = open(path, "a+b")
    deadline = time.monotonic() + wait
    locked = _try_lock(handle)
    while not locked and time.monotonic() < deadline:
        time.sleep(0.05)
        locked = _try_lock(handle)
    if not locked:
        handle.close()
        runner = RunnerState.load(out_dir)
        who = (
            f"pid {runner.pid} since {runner.started}"
            if runner.pid
            else "another process"
        )
        raise LockHeld(who)
    return RunnerLock(path, handle)


def release_lock(lock: RunnerLock | None) -> None:
    """Give a held lock back (None is a no-op)."""
    if lock is None:
        return
    _unlock(lock.handle)
    lock.handle.close()


def lock_held(out_dir: Path) -> bool:
    """Whether some process holds the runner lock right now."""
    path = Path(out_dir) / LOCK_FILE
    if not path.exists():
        return False
    try:
        handle = open(path, "a+b")
    except OSError:
        return True
    try:
        if _try_lock(handle):
            _unlock(handle)
            return False
        return True
    finally:
        handle.close()


def process_started(pid: int) -> float:
    """When a process started (epoch seconds), 0.0 if unknown."""
    try:
        import psutil
    except ImportError:
        return 0.0
    try:
        return psutil.Process(pid).create_time()
    except Exception:  # noqa: BLE001 -- no such process, access denied
        return 0.0


def _jsonable(value: Any) -> Any:
    """Fallback for values json cannot write (numpy scalars, sets)."""
    if isinstance(value, set):
        return sorted(value)
    for attr in ("item", "tolist"):
        if hasattr(value, attr):
            try:
                return getattr(value, attr)()
            except Exception:  # noqa: BLE001
                pass
    return str(value)
