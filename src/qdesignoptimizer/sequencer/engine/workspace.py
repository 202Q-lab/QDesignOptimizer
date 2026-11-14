"""Several projects, one queue.

Opened from folders or adapter files::

    python -m qdesignoptimizer.sequencer.app <design A> <design B>

A folder is searched for its adapter (``simulations/sequencer.py``, else
every ``sequencer.py`` below it that defines ``make_backend``; ``out/``
and the like are skipped), so a parent folder opens every design in it.

Every study stays in its own project's ``out/queue/<id>/`` and moves to
that project's ``out/completed/``: a project is complete on its own and
opens alone exactly as before. What several projects share:

* **the order** -- ``<state>/order.json``, a list of ``<project>/<id>``;
  every reorder also writes each project's own ``order.json``, so a
  project opened alone keeps the relative order;
* **the runner** -- ``<state>/runner.json``, ``runner.log``, ``STOP``,
  where ``<state>`` is a folder per set of projects under the user's
  ``qdo_sequencer`` folder (``QDO_SEQUENCER_HOME`` overrides it).

One project alone keeps everything in its ``out/``, as it always did.
Either way a runner also takes the machine lock (``<home>/runner.lock``),
so two runners -- two GUIs, a GUI and a shell -- never solve at once.

Keys: a study is ``<project>/<id>`` when several projects are open and
``<id>`` when one is; ``locate`` accepts either, and a bare id that only
one project has.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable

from . import state as st
from .job import parse_completed
from .project import Project, open_project
from .queue import QueueError

ADAPTER = "sequencer.py"
_SKIP = {"out", ".git", "__pycache__", ".ipynb_checkpoints", "node_modules"}


def user_home() -> Path:
    """The user's ``qdo_sequencer`` folder (machine lock, shared state)."""
    if os.environ.get("QDO_SEQUENCER_HOME"):
        return Path(os.environ["QDO_SEQUENCER_HOME"])
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "qdo_sequencer"
    return Path.home() / ".local" / "state" / "qdo_sequencer"


def find_adapters(path: str | Path) -> list[str]:
    """The adapter(s) a command-line argument names.

    A ``.py`` file or ``file.py:factory`` or a dotted module is itself;
    a folder is searched (see the module docstring).

    Raises:
        QueueError: If a folder holds no adapter.
    """
    text = str(path)
    candidate = Path(text)
    if not candidate.is_dir():
        return [text]
    direct = candidate / "simulations" / ADAPTER
    if direct.is_file():
        return [str(direct.resolve())]
    found = []
    for folder, dirs, files in os.walk(candidate):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP)
        if ADAPTER in files:
            file = Path(folder) / ADAPTER
            try:
                if "make_backend" in file.read_text(encoding="utf-8"):
                    found.append(str(file.resolve()))
            except OSError:
                continue
    if not found:
        raise QueueError(f"no {ADAPTER} with make_backend under {candidate}")
    return found


class Workspace:
    """The open projects, their merged queue and the runner's folder.

    Args:
        projects: The projects, in the order they were named.
        refs: The adapter of each (to start a runner); may be empty for
            injected projects.
        state_dir: Where the runner's files and the shared order live.
    """

    def __init__(
        self,
        projects: list[Project],
        refs: list[str] | None = None,
        state_dir: Path | None = None,
    ):
        if not projects:
            raise QueueError("no project to open")
        self.projects: dict[str, Project] = {}
        for project in projects:
            # Not starting with "~": the GUI marks a study sequence draft so.
            base = project.out.parent.name.lstrip("~") or "project"
            name, n = base, 2
            while name in self.projects:
                name, n = f"{base}-{n}", n + 1
            self.projects[name] = project
        self.refs = list(refs or [])
        self.single = len(projects) == 1
        if state_dir is None:
            state_dir = (
                projects[0].out
                if self.single
                else workspace_dir([p.out for p in projects])
            )
        self.state_dir = Path(state_dir)
        self.order_file = self.state_dir / "order.json"

    @classmethod
    def of(cls, project: Project, ref: str = "") -> Workspace:
        """One project, as before: everything in its ``out/``."""
        return cls([project], [ref] if ref else [])

    # -- names -----------------------------------------------------------

    def key(self, name: str, local: str) -> str:
        """The key the GUI and the CLI use for a study or a folder."""
        return local if self.single else f"{name}/{local}"

    def name_of(self, project: Project) -> str:
        """The name a project goes by here."""
        return next(n for n, p in self.projects.items() if p is project)

    def locate(self, key: str) -> tuple[Project, str]:
        """The project and its own id for a key.

        Raises:
            QueueError: For an unknown project or an ambiguous bare id.
        """
        key = str(key)
        if "/" in key:
            name, local = key.split("/", 1)
            if name not in self.projects:
                raise QueueError(f"no project {name!r} is open")
            return self.projects[name], local
        if self.single:
            return next(iter(self.projects.values())), key
        holders = [
            p
            for p in self.projects.values()
            if p.queue.folder(key).is_dir() or (p.queue.completed_dir / key).is_dir()
        ]
        if len(holders) == 1:
            return holders[0], key
        raise QueueError(
            f"{key!r}: name the project as <project>/{key}"
            if holders
            else f"{key!r} is in none of the open projects"
        )

    def locate_task(self, key: str) -> tuple[Project, str]:
        """The project of a study or study sequence key (``<project>/``
        prefix, or a key only one project has)."""
        if "/" in key:
            return self.locate(key)
        if self.single:
            return next(iter(self.projects.values())), key
        holders = [
            p
            for p in self.projects.values()
            if p.backend.catalog().task(key) is not None
            or p.backend.catalog().sequence(key) is not None
        ]
        if len(holders) == 1:
            return holders[0], key
        raise QueueError(
            f"{key!r}: name the project as <project>/{key}"
            if holders
            else f"no project has {key!r}"
        )

    # -- the queue -------------------------------------------------------

    def ids(self) -> list[str]:
        """Every queued study's key, in run order.

        One project: its own order. Several: the shared order, then
        anything it does not list (new, or queued elsewhere), project
        by project in each one's own order.
        """
        if self.single:
            return next(iter(self.projects.values())).queue.ids()
        present = [
            self.key(name, i)
            for name, project in self.projects.items()
            for i in project.queue.ids()
        ]
        listed = st.read_json(self.order_file) or []
        known = set(present)
        order = [k for k in listed if k in known]
        seen = set(order)
        return order + [k for k in present if k not in seen]

    def added(self, project: Project, ids: Iterable[str]) -> list[str]:
        """Put studies just queued in ``project`` at the end of the
        shared order; returns their keys.

        Without this a new study would sit with its own project's
        unlisted ones rather than at the bottom where it was added.
        """
        name = self.name_of(project)
        keys = [self.key(name, i) for i in ids]
        if not self.single:
            with st.ORDER_LOCK:
                listed = [k for k in self.ids() if k not in keys]
                st.write_json(self.order_file, listed + keys)
        return keys

    def save_order(self, keys: Iterable[str]) -> None:
        """Set the run order (keys as ``ids`` gives them).

        Raises:
            QueueError: If ``keys`` is not exactly the queued keys.
        """
        keys = [str(k) for k in keys]
        if self.single:
            next(iter(self.projects.values())).queue.save_order(keys)
            return
        with st.ORDER_LOCK:
            current = self.ids()
            if sorted(keys) != sorted(current) or len(set(keys)) != len(keys):
                raise QueueError("the queue changed meanwhile; reloaded")
            st.write_json(self.order_file, keys)
            for name, project in self.projects.items():
                prefix = f"{name}/"
                own = [k[len(prefix) :] for k in keys if k.startswith(prefix)]
                project.queue.save_order(own)

    def completed(self, limit: int | None = None) -> list[tuple]:
        """``(project, folder)`` of completed studies, newest first."""
        runs = [
            (project, path)
            for project in self.projects.values()
            for path in project.queue.completed(limit)
        ]
        runs.sort(key=lambda r: _started(r[1]), reverse=True)
        return runs[:limit] if limit else runs

    def close(self) -> None:
        """End the projects' worker processes, if any."""
        for project in self.projects.values():
            closer = getattr(project.backend, "close", None)
            if closer is not None and _is_remote(project.backend):
                closer()


def workspace_dir(outs: list[Path]) -> Path:
    """The state folder for a set of projects (the same set, in any
    order, gets the same folder)."""
    roots = sorted(str(Path(o).resolve()).lower() for o in outs)
    digest = hashlib.sha256("\n".join(roots).encode("utf-8")).hexdigest()
    names = "+".join(sorted(Path(o).parent.name for o in outs))[:60]
    return user_home() / "workspaces" / f"{names}-{digest[:8]}"


def _started(folder: Path) -> tuple:
    """Sort key: the study's start time (from its folder name), then
    when its result was last written -- two studies of different
    projects can start in the same second."""
    stamp = parse_completed(folder.name)[1]
    try:
        written = (folder / st.RESULT_FILE).stat().st_mtime_ns
    except OSError:
        written = 0
    return stamp, written


def _is_remote(backend) -> bool:
    from .worker import RemoteBackend

    return isinstance(backend, RemoteBackend)


def open_workspace(
    paths: Iterable[str | Path], isolate: bool | None = None
) -> Workspace:
    """Open every project the paths name.

    Args:
        paths: Adapter files, ``file.py:factory`` refs or folders.
        isolate: Run each backend in a worker process. Default: when
            more than one project is open (see ``worker``).

    Raises:
        QueueError: If a folder holds no adapter.
        BackendError, WorkerError: If a project cannot be loaded.
    """
    refs: list[str] = []
    for path in paths:
        for ref in find_adapters(path):
            if ref not in refs:
                refs.append(ref)
    if isolate is None:
        isolate = len(refs) > 1
    projects = []
    try:
        for ref in refs:
            if isolate:
                from .worker import RemoteBackend

                projects.append(Project(RemoteBackend(ref)))
            else:
                projects.append(open_project(ref))
    except BaseException:
        for project in projects:  # the workers already started
            closer = getattr(project.backend, "close", None)
            if closer is not None and _is_remote(project.backend):
                closer()
        raise
    return Workspace(projects, refs)
