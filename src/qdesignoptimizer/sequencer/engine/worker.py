"""One project's backend in a process of its own.

Design folders are not packages: an adapter puts its design folder on
the import path and the project's modules import each other by bare
name (``import names as n``, ``simulations._runner``). Two
projects in one Python process would then share ``names``,
``parameter_targets`` and ``simulations`` -- the second project quietly
gets the first one's modules, with no error. So when several projects
are open, each backend lives in a worker process and is called through
``RemoteBackend``, which looks like the backend itself.

The worker is ``python -m qdesignoptimizer.sequencer.engine.worker``; it connects
back over a local socket (``multiprocessing.connection``, an auth key in
its environment) and exits when that connection closes, so it never
outlives the GUI or the runner that started it.
"""

from __future__ import annotations

import logging
import os
import secrets
import subprocess
import sys
import threading
from multiprocessing.connection import Client, Connection, Listener
from pathlib import Path
from typing import Any

#: Marks the log handler a ``log`` message attached, to replace it.
_RUNNER_LOG = "_qdo_sequencer_runner_log"

#: Methods polled every two seconds by the GUI: while the worker is busy
#: with something long (a deep check builds the design), the last answer
#: for the same arguments is returned instead of waiting.
_POLLED = {"catalog_with_problems", "check_folder"}
#: Read once when the worker starts; they do not change.
_STATIC = ("name", "words", "out_dir")
_AUTH_ENV = "QDO_SEQUENCER_WORKER_KEY"


class WorkerError(RuntimeError):
    """The worker died or could not be started."""


class RemoteBackend:
    """A backend running in a worker process.

    Args:
        ref: The project's adapter (``sequencer.py`` or
            ``file.py:factory``), as ``open_project`` takes it.
    """

    def __init__(self, ref: str):
        self.ref = str(ref)
        self._lock = threading.RLock()
        self._conn: Connection | None = None
        self._proc: subprocess.Popen | None = None
        self._methods: set[str] = set()
        self._values: dict[str, Any] = {}
        self._last: dict[tuple, Any] = {}
        self._log_path: str | None = None
        self.catalog_problems: list[str] = []
        self._start()

    # -- process ---------------------------------------------------------

    def _start(self) -> None:
        key = secrets.token_bytes(16)
        listener = Listener(("127.0.0.1", 0), authkey=key)
        env = {**os.environ, _AUTH_ENV: key.hex()}
        port = listener.address[1]
        flags = 0
        if os.name == "nt":
            flags = subprocess.CREATE_NO_WINDOW
        self._proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "qdesignoptimizer.sequencer.engine.worker",
                str(port),
                self.ref,
            ],
            env=env,
            stdin=subprocess.DEVNULL,
            creationflags=flags,
        )
        # accept() has no timeout; a worker that dies before connecting
        # must not hang the GUI, so wait in a thread.
        box: dict[str, Any] = {}

        def accept():
            try:
                box["conn"] = listener.accept()
            except Exception as exc:  # noqa: BLE001 -- reported below
                box["error"] = exc

        thread = threading.Thread(target=accept, daemon=True)
        thread.start()
        thread.join(120)
        listener.close()
        if "conn" not in box:
            self._kill()
            raise WorkerError(
                f"the worker for {self.ref} did not start"
                + (f": {box['error']}" if "error" in box else "")
            )
        conn: Connection = box["conn"]
        self._conn = conn
        status, value = conn.recv()
        if status != "ok":
            self._kill()
            raise WorkerError(f"{self.ref}: {value}")
        self._methods = set(value["methods"])
        self._values = dict(value["values"])
        if self._log_path:
            self._send(("log", self._log_path))

    def _kill(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:
                pass
            self._conn = None
        if self._proc is not None:
            try:
                self._proc.wait(10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None

    def close(self) -> None:
        """End the worker (it also ends by itself when this one ends)."""
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.send(("quit",))
                except OSError:
                    pass
            self._kill()

    def reload(self) -> None:
        """Start a fresh worker: every project module is imported anew."""
        with self._lock:
            self.close()
            self._last.clear()
            self._start()

    def log_to(self, path: Path) -> None:
        """Send the worker's log records to the runner's log too."""
        self._log_path = str(path)
        with self._lock:
            self._send(("log", self._log_path))

    # -- calls -----------------------------------------------------------

    def _send(self, message: tuple) -> Any:
        if self._conn is None:
            self._start()
        assert self._conn is not None  # _start connects or raises
        try:
            self._conn.send(message)
            status, value = self._conn.recv()
        except (EOFError, OSError) as exc:
            self._kill()
            raise WorkerError(
                f"the worker for {self.ref} died ({type(exc).__name__}); "
                f"the next call starts a new one"
            ) from exc
        if status == "err":
            raise value
        return value

    def _call(self, name: str, *args, **kwargs) -> Any:
        polled = name in _POLLED
        key = (name, repr(args), repr(sorted(kwargs.items())))
        if polled and key in self._last:
            if not self._lock.acquire(blocking=False):
                return self._last[key]
        else:
            self._lock.acquire()
        try:
            value = self._send(("call", name, args, kwargs))
        finally:
            self._lock.release()
        if polled:
            self._last[key] = value
        return value

    def catalog(self):
        """The catalog; its problems land in ``catalog_problems``."""
        catalog, problems = self._call("catalog_with_problems")
        self.catalog_problems = problems
        return catalog

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        values = self.__dict__.get("_values", {})
        if name in values:
            return values[name]
        if name in self.__dict__.get("_methods", set()):
            return lambda *a, **k: self._call(name, *a, **k)
        raise AttributeError(f"{self.ref} has no {name!r}")


# -- the worker side ---------------------------------------------------------


def _describe(backend) -> dict:
    methods = []
    for name in dir(backend):
        if name.startswith("_") or name in _STATIC:
            continue
        try:
            value = getattr(type(backend), name, None)
        except Exception:  # noqa: BLE001
            continue
        if callable(value) or callable(getattr(backend, name, None)):
            methods.append(name)
    methods.append("catalog_with_problems")
    return {
        "methods": methods,
        "values": {k: getattr(backend, k) for k in _STATIC},
    }


def _portable(exc: BaseException) -> BaseException:
    """An exception that survives pickling, keeping type and message."""
    import pickle

    try:
        pickle.loads(pickle.dumps(exc))
        return exc
    except Exception:  # noqa: BLE001
        return RuntimeError(f"{type(exc).__name__}: {exc}")


def serve(port: int, ref: str) -> None:
    """The worker's loop: load the backend, answer calls until closed."""
    from .project import open_project

    conn = Client(("127.0.0.1", port), authkey=bytes.fromhex(os.environ[_AUTH_ENV]))
    try:
        backend = open_project(ref).backend
        conn.send(("ok", _describe(backend)))
    except Exception as exc:  # noqa: BLE001 -- sent back
        conn.send(("err", f"{type(exc).__name__}: {exc}"))
        return
    while True:
        try:
            message = conn.recv()
        except (EOFError, OSError):
            return  # the GUI or the runner is gone
        if message[0] == "quit":
            return
        try:
            if message[0] == "log":
                # One runner log at a time: the one log_to last named. A
                # re-sent log_to, or a later runner with another log, must
                # neither double every line nor keep writing to the old.
                for old in list(logging.getLogger().handlers):
                    if getattr(old, _RUNNER_LOG, False):
                        logging.getLogger().removeHandler(old)
                        old.close()
                handler = logging.FileHandler(message[1], encoding="utf-8")
                setattr(handler, _RUNNER_LOG, True)
                handler.setFormatter(
                    logging.Formatter(
                        "%(asctime)s %(levelname)s %(name)s: %(message)s",
                        "%Y-%m-%d %H:%M:%S",
                    )
                )
                logging.getLogger().addHandler(handler)
                logging.getLogger().setLevel(logging.INFO)
                value = None
            elif message[1] == "catalog_with_problems":
                catalog = backend.catalog()
                value = (
                    catalog,
                    list(getattr(backend, "catalog_problems", []) or []),
                )
            else:
                _, name, args, kwargs = message
                value = getattr(backend, name)(*args, **kwargs)
            reply: tuple[str, Any] = ("ok", value)
        except BaseException as exc:  # noqa: BLE001 -- sent back
            reply = ("err", _portable(exc))
        try:
            conn.send(reply)
        except Exception as exc:  # noqa: BLE001 -- an unpicklable result
            conn.send(("err", RuntimeError(f"cannot send the result back: {exc}")))


if __name__ == "__main__":
    serve(int(sys.argv[1]), sys.argv[2])
