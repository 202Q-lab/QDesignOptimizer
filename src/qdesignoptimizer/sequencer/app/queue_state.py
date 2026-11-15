"""Bridge between the GUI and the projects' folders. Imports no Dash.

Nothing is cached but the loaded backends: every poll re-reads the
queue folders, ``order.json``, ``runner.json`` and the completed
results, because the runner, a shell and an agent change them too.

Several projects share one queue through a ``Workspace``; a study's key
is then ``<project>/<id>`` (a plain id when one project is open).
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from ..engine import state as st
from ..engine.campaign import problems_of as campaign_problems
from ..engine.job import (
    JOB_FILE,
    JobError,
    dump_yaml,
    job_file,
    load_job,
    numbers,
    parse_completed,
    parse_job,
)
from ..engine.project import Project
from ..engine.queue import INPUTS, RESOLVED, QueueError
from ..engine.runner import ACTIVE_FILE
from ..engine.workspace import Workspace, find_adapters, open_workspace, user_home
from . import forms

#: What the page says with no project connected.
NO_PROJECT = "no project folder is connected: + Connected project folders"


def repo_label(folder: str | Path) -> str:
    """A project folder as the GUI names it: relative to the root of the
    git repository it is in (``tutorials/.../chip_qdo``), or its own
    name outside one."""
    folder = Path(folder).resolve()
    for parent in [folder, *folder.parents]:
        if (parent / ".git").exists():
            if parent == folder:
                return folder.name
            return folder.relative_to(parent).as_posix()
    return folder.name


def read_connected(path: Path) -> list[str]:
    """The remembered project folders ([] when none or unreadable)."""
    data = st.read_json(path)
    if not isinstance(data, list):
        return []
    return [str(p) for p in data if isinstance(p, str)]


class GuiError(RuntimeError):
    """An action the GUI refuses, with the reason."""


class Bridge:
    """The open project(s), as the GUI sees them.

    Args:
        ref: A project's ``sequencer.py`` or folder, or a list of them
            (several projects sharing one queue).
        project: Injected project (tests); else opened from ``ref``.
        workspace: Injected workspace (tests).
        remember: A file to keep the connected project folders in, for
            the next start (``connect`` / ``disconnect`` write it).
    """

    def __init__(
        self,
        ref: str | list = "",
        project: Project | None = None,
        workspace: Workspace | None = None,
        remember: Path | None = None,
    ):
        self.remember = remember
        self.refs = (
            [str(r) for r in ref]
            if isinstance(ref, list)
            else ([str(ref)] if ref else [])
        )
        self.ref = self.refs[0] if self.refs else ""
        if project is not None:
            workspace = Workspace.of(project)
        self._workspace = workspace
        self._injected = workspace is not None
        # Deep checks rebuild QDO objects -- too slow for every 2 s poll.
        # The last deep result per folder is kept with the files'
        # modification times and shown until a file changes.
        self._deep: dict[str, tuple[tuple, list[str]]] = {}

    @property
    def workspace(self) -> Workspace:
        """The open projects (loaded on first use).

        Always in worker processes, one project too: the GUI answers in
        its server's threads, and building a qiskit-metal design (the
        name check after every save) needs a main thread, which each
        worker has. The command line runs in-process.
        """
        if self._workspace is None:
            if not self.refs:
                raise GuiError(NO_PROJECT)
            self._workspace = open_workspace(self.refs, isolate=True)
        return self._workspace

    # -- connected project folders --------------------------------------

    def connected(self) -> list[dict]:
        """The connected project folders: as given, how the GUI names
        them, and the full path (shown on hover)."""
        if self._injected:
            return [
                {
                    "path": str(p.out.parent),
                    "label": repo_label(p.out.parent),
                    "title": str(p.out.parent.resolve()),
                }
                for p in self.workspace.projects.values()
            ]
        return [
            {
                "path": ref,
                "label": repo_label(_folder_of(ref)),
                "title": str(_folder_of(ref)),
            }
            for ref in self.refs
        ]

    def _runner_going(self) -> bool:
        if self._workspace is None:
            return False
        out = self._workspace.state_dir
        return st.lock_held(out) or st.RunnerState.load(out).alive(out)

    def _reopen(self, refs: list[str]) -> None:
        """Open ``refs`` in place of what is open (the runner must be
        stopped: it runs the queue of the projects it was started with)."""
        if self._injected:
            raise GuiError("these projects were not opened from folders")
        if self._runner_going():
            raise GuiError(
                "the runner is going: stop it (Stop after this study) before "
                "connecting or disconnecting a project folder"
            )
        new = None
        if refs:
            try:
                new = open_workspace(refs, isolate=True)
            except Exception as exc:  # noqa: BLE001 -- shown to the user
                raise GuiError(f"cannot open it: {exc}") from exc
        if self._workspace is not None:
            self._workspace.close()
        self._workspace = new
        self.refs = list(refs)
        self.ref = self.refs[0] if self.refs else ""
        self._deep.clear()
        if self.remember is not None:
            self.remember.parent.mkdir(parents=True, exist_ok=True)
            st.write_json(self.remember, self.refs)

    def connect(self, path: str) -> str:
        """Connect a project folder (or its ``sequencer.py``): its queue
        joins the shared one."""
        text = str(path or "").strip().strip('"')
        if not text:
            raise GuiError("type the path of a project folder")
        target = Path(text).expanduser()
        if not target.exists():
            raise GuiError(f"{text} does not exist")
        target = target.resolve()
        try:
            find_adapters(target)
        except QueueError as exc:
            raise GuiError(str(exc)) from exc
        folder = _folder_of(str(target))
        if any(_folder_of(r) == folder for r in self.refs):
            raise GuiError(f"{repo_label(folder)} is connected already")
        self._reopen([*self.refs, str(target)])
        return f"connected {repo_label(folder)}"

    def disconnect(self, path: str) -> str:
        """Disconnect a project folder: its studies stay in its ``out/``
        and come back when it is connected again."""
        if path not in self.refs:
            raise GuiError(f"{path} is not connected")
        self._reopen([r for r in self.refs if r != path])
        return (
            f"disconnected {repo_label(_folder_of(path))}; its studies stay "
            "in its out/"
        )

    @property
    def project(self) -> Project:
        """The project, when one is open."""
        workspace = self.workspace
        if not workspace.single:
            raise GuiError("several projects are open; name one")
        return next(iter(workspace.projects.values()))

    def _study(self, key: str):
        """The project (scoped to the hold folder for a study on hold)
        and the study's own id."""
        held = _split_hold(key)
        if held is not None:
            project = self._project_named(held[0], key)
            scoped = _Scoped(project, project.hold)
            scoped.key_prefix = str(key).rsplit("/", 1)[0]
            return scoped, held[1]
        return self._loc(key)

    def _project_named(self, head: str, key: str) -> Project:
        workspace = self.workspace
        if head:
            if head not in workspace.projects:
                raise GuiError(f"no project {head!r} is open")
            return workspace.projects[head]
        if workspace.single:
            return next(iter(workspace.projects.values()))
        raise GuiError(f"{key!r}: name the project")

    def _hold_key(self, project: Project, local: str) -> str:
        return self.workspace.key(
            self.workspace.name_of(project), f"{HOLD_MARK}/{local}"
        )

    def _loc(self, key: str) -> tuple[Project, str]:
        """The project and its own id (or folder name) for a key."""
        try:
            return self.workspace.locate(key)
        except QueueError as exc:
            raise GuiError(str(exc)) from exc

    def _key(self, project, local: str) -> str:
        if isinstance(project, _Scoped):
            return f"{project.key_prefix}/{local}"
        return self.workspace.key(self.workspace.name_of(project), local)

    def _signature(self, key: str) -> tuple:
        project, job_id = self._study(key)
        folder = project.queue.folder(job_id)
        try:
            return tuple(
                sorted(
                    (p.name, p.stat().st_mtime_ns)
                    for p in folder.iterdir()
                    if p.is_file()
                )
            )
        except FileNotFoundError:
            return ()

    def _problems(self, key: str) -> list[str]:
        """Deep problems while the files are unchanged, else shallow."""
        cached = self._deep.get(key)
        if cached and cached[0] == self._signature(key):
            return cached[1]
        project, job_id = self._study(key)
        return project.problems(job_id)

    def _write(self, key: str, name: str, text: str) -> None:
        """Write a study file, or say why not: the runner may have
        claimed the study since the page was drawn."""
        project, job_id = self._study(key)
        try:
            project.queue.write_file(job_id, name, text)
        except (QueueError, OSError) as exc:
            if not project.queue.folder(job_id).is_dir():
                raise GuiError(
                    f"{job_id} is no longer there (started, moved or "
                    f"removed meanwhile); nothing was saved"
                ) from exc
            # Still queued: the write itself failed (a file held open by
            # the virus scanner, a full disk). Say that, not "gone".
            raise GuiError(
                f"could not save {name}: {type(exc).__name__}: {exc}; " f"try again"
            ) from exc

    def _taken(self, key: str) -> bool:
        """Whether a study just written is no longer queued: the runner
        took it after the write landed (a write after the take is
        refused by ``_write``), so it runs with the edit."""
        project, job_id = self._study(key)
        return not project.queue.folder(job_id).is_dir()

    def _deep_check(self, key: str) -> list[str]:
        project, job_id = self._study(key)
        problems = project.problems(job_id, deep=True)
        self._deep[key] = (self._signature(key), problems)
        return problems

    def reload(self) -> None:
        """Read the catalog (the templates) again.

        A project in a worker process gets a fresh worker, which imports
        ``sequencer.py`` and everything behind it anew; an injected one
        (tests) re-reads its templates through the backend's own
        ``reload``.
        """
        workspace = self._workspace
        if workspace is not None:
            for project in workspace.projects.values():
                hook = getattr(project.backend, "reload", None)
                if hook is not None:
                    hook()
        self._deep.clear()

    def close(self) -> None:
        """End worker processes (the GUI is closing)."""
        if self._workspace is not None:
            self._workspace.close()

    # -- reading ---------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Everything the page shows, as plain data."""
        if self._workspace is None and not self.refs and not self._injected:
            return {
                "empty": NO_PROJECT,
                "rows": [],
                "held": [],
                "completed": [],
                "runner": {},
                "words": {},
                "catalog": [],
                "campaigns": [],
                "projects": [],
                "connected": [],
            }
        workspace = self.workspace
        first = next(iter(workspace.projects.values()))
        words = first.words
        catalogs = {name: p.backend.catalog() for name, p in workspace.projects.items()}
        rows = []
        for n, key in enumerate(workspace.ids(), 1):
            project, job_id = workspace.locate(key)
            name = workspace.name_of(project)
            catalog = catalogs[name]
            try:
                job = project.job(job_id)
            except JobError as exc:
                rows.append(
                    {
                        "id": key,
                        "project": name,
                        "n": n,
                        "task": "",
                        "label": "",
                        "subject": "",
                        "sequence": "",
                        "settings": "",
                        "notes": "",
                        "problems": exc.problems,
                    }
                )
                continue
            spec = catalog.task(job.task)
            rows.append(
                {
                    "id": key,
                    "project": name,
                    "n": n,
                    "task": job.task,
                    "label": spec.label if spec else job.task,
                    "subject": job.subject,
                    "sequence": job.sequence.get("id", ""),
                    "sequence_name": job.sequence.get("name", ""),
                    "settings": ", ".join(f"{k} {v}" for k, v in job.settings.items()),
                    "save": job.save,
                    "notes": job.notes,
                    "campaign": job.campaign,
                    "problems": self._problems(key),
                    "warnings": project.warnings(job_id),
                }
            )
        held = []
        for name, project in workspace.projects.items():
            catalog = catalogs[name]
            for local in project.hold.ids():
                key = self._hold_key(project, local)
                try:
                    job = load_job(project.hold.folder(local), project.words)
                except JobError as exc:
                    held.append(
                        {
                            "id": key,
                            "project": name,
                            "label": local,
                            "problems": exc.problems,
                        }
                    )
                    continue
                spec = catalog.task(job.task)
                held.append(
                    {
                        "id": key,
                        "project": name,
                        "task": job.task,
                        "label": spec.label if spec else job.task,
                        "subject": job.subject,
                        "settings": ", ".join(
                            f"{k} {v}" for k, v in job.settings.items()
                        ),
                        "notes": job.notes,
                        "campaign": job.campaign,
                        "problems": self._problems(key),
                    }
                )
        out = workspace.state_dir
        runner = st.RunnerState.load(out)
        alive = runner.alive(out)
        elsewhere = ""
        if not alive and st.lock_held(user_home()):
            active = st.read_json(user_home() / ACTIVE_FILE) or {}
            elsewhere = (
                "another runner is going on this machine"
                + (
                    f" ({', '.join(active['projects'])})"
                    if active.get("projects")
                    else ""
                )
                + "; this queue starts when it is done"
            )
        completed = []
        for project, path in workspace.completed(25):
            res = st.result_of(path)
            status = res.get("status", "?")
            if status in (st.RUNNING, st.COMMITTING) and not alive:
                # Cut off with its runner; the next runner records it
                # failed. Until then say so, not "running".
                status = "interrupted"
            completed.append(
                {
                    "name": self._key(project, path.name),
                    "project": workspace.name_of(project),
                    "id": parse_completed(path.name)[0],
                    "status": status,
                    "task": res.get(words.task, ""),
                    "started": res.get("started", ""),
                    "finished": res.get("finished", ""),
                    "message": res.get("message", ""),
                    "committed": res.get("committed", {}),
                    "campaign": res.get("campaign", ""),
                }
            )
        problems = []
        for name, project in workspace.projects.items():
            for problem in getattr(project.backend, "catalog_problems", []) or []:
                problems.append(problem if workspace.single else f"{name}: {problem}")
        catalog_rows = []
        for name, catalog in catalogs.items():
            prefix = "" if workspace.single else f"{name} / "
            catalog_rows += [
                {
                    "kind": "task",
                    "key": workspace.key(name, t.key),
                    "label": prefix + t.label,
                    "minutes": t.minutes,
                }
                for t in catalog.tasks
            ]
        for name, catalog in catalogs.items():
            prefix = "" if workspace.single else f"{name} / "
            catalog_rows += [
                {
                    "kind": "sequence",
                    "key": workspace.key(name, c.key),
                    "label": prefix + c.label,
                    "count": len(c.entries),
                    # The file, or "" for one built in the adapter's code.
                    "source": (
                        _relative(workspace.projects[name], c.source)
                        if c.source
                        else ""
                    ),
                }
                for c in catalog.sequences
            ]
        return {
            "project": (
                str(first.backend.out_dir.parent)
                if workspace.single
                else "; ".join(str(p.out.parent) for p in workspace.projects.values())
            ),
            "multi": not workspace.single,
            # The project picker: the name keys use, the folder relative
            # to its repository, and the full path on hover.
            "projects": [
                {
                    "value": name,
                    "label": repo_label(p.out.parent),
                    "title": str(p.out.parent.resolve()),
                }
                for name, p in workspace.projects.items()
            ],
            "connected": self.connected(),
            "catalog_problems": problems,
            "words": {
                k: getattr(words, k)
                for k in ("task", "sequence", "subject", "store", "live")
            },
            "rows": rows,
            "held": held,
            "completed": completed,
            "runner": {
                "status": (
                    runner.status
                    if alive or runner.status != st.BUSY
                    else "stopped unexpectedly"
                ),
                "alive": alive,
                "current": runner.current,
                "message": runner.message,
                "started": runner.started,
                "finished": runner.finished,
                "stop_requested": alive and st.stop_requested(out),
                "elsewhere": elsewhere,
            },
            "catalog": catalog_rows,
            "campaigns": self.campaigns(),
        }

    # -- campaigns -------------------------------------------------------

    def campaigns(self) -> list[dict]:
        """Every campaign of every open project, parents first: key,
        title, depth and the store its studies use (relative to the
        project)."""
        out = []
        for name, project in self.workspace.projects.items():
            try:
                found = project.campaigns()
            except Exception:  # noqa: BLE001 -- the card says none
                continue
            for item in found:
                out.append(
                    {
                        "project": name,
                        "key": item.key,
                        "title": item.title,
                        "depth": item.depth,
                        "own_store": item.own_store,
                        "store": _relative(project, project.campaign_store(item.key)),
                    }
                )
        return out

    def _project_for(self, project: str = ""):
        workspace = self.workspace
        if project:
            if project not in workspace.projects:
                raise GuiError(f"no project {project!r}")
            return workspace.projects[project]
        if not workspace.single:
            raise GuiError("pick the project the campaign is in")
        return next(iter(workspace.projects.values()))

    def new_campaign(
        self, key: str, question: str = "", own_store: bool = True, project: str = ""
    ) -> str:
        """Make a campaign (a sub-campaign when ``key`` has a ``/``)."""
        target = self._project_for(project)
        try:
            folder = target.new_campaign(key, question, own_store)
        except QueueError as exc:
            raise GuiError(str(exc)) from exc
        store = (
            "with its own copy of the design variables"
            if own_store
            else "sharing its parent's design variables"
        )
        return f"made campaign {key.strip()} {store}: {_relative(target, folder)}"

    def promote(self, key: str, project: str = "") -> str:
        """Write a campaign's changed values one level up."""
        target = self._project_for(project)
        try:
            changed = target.promote(key)
        except QueueError as exc:
            raise GuiError(str(exc)) from exc
        # The file it went to, not "campaign X": a campaign sharing its
        # parent's store passes the values straight through.
        where = _relative(target, target.campaign_store(key.rpartition("/")[0]))
        if not changed:
            return f"nothing to promote: {where} has the same values"
        self._deep.clear()
        return f"promoted to {where}: " + ", ".join(sorted(changed))

    def convergence_png(self, kind: str, key: str) -> str | None:
        """A completed study's ``convergence.png``, base64; None when it
        is not a completed study or has no such picture (a fake-backend
        study, or one that failed before its first iteration)."""
        if kind != "completed":
            return None
        try:
            project, local = self._study(key)
        except GuiError:
            return None
        path = project.queue.completed_dir / local / "convergence.png"
        try:
            return base64.b64encode(path.read_bytes()).decode("ascii")
        except OSError:
            return None

    def details(self, kind: str, key: str) -> dict[str, Any]:
        """One study's files and record, for the details pane.

        Args:
            kind: ``queue`` (a queued id) or ``completed`` (a folder).
            key: The id or the folder name.
        """
        project, local = self._study(key)
        if kind in ("queue", "hold"):
            folder = project.queue.folder(local)
        else:
            folder = project.queue.completed_dir / local
        if not folder.is_dir():
            raise GuiError(f"{key} is no longer there")
        specs = project.backend.files()
        campaign = _campaign_of(project, folder)
        files = []
        for spec in specs:
            path = folder / spec.name
            text = path.read_text(encoding="utf-8") if path.exists() else None
            try:
                data = _parse(spec.name, text) if text is not None else None
                problem = ""
            except (ValueError, yaml.YAMLError) as exc:
                data, problem = None, f"{spec.name} does not read: {exc}"
            files.append(
                {
                    "name": spec.name,
                    "label": spec.label,
                    "about": spec.about,
                    "optional": spec.optional,
                    "text": text,
                    "data": data,
                    "problem": problem,
                    "project": (
                        _project_values(project, spec.name, campaign)
                        if spec.optional and kind in ("queue", "hold")
                        else None
                    ),
                    "sources": (
                        _value_sources(project, spec.name, campaign)
                        if spec.optional
                        else {}
                    ),
                    "addable": (
                        _addable(project, spec.name)
                        if kind in ("queue", "hold")
                        else {}
                    ),
                    "target_templates": (
                        _target_templates(project)
                        if kind in ("queue", "hold")
                        and isinstance(data, dict)
                        and isinstance(data.get("targets"), list)
                        else None
                    ),
                }
            )
        job_text = (
            job_file(folder).read_text(encoding="utf-8")
            if job_file(folder).exists()
            else ""
        )
        try:
            job = (
                parse_job(yaml.safe_load(job_text) or {}, project.words)
                if job_text
                else None
            )
        except JobError:
            job = None
        raw = job.to_raw(project.words) if job else {}
        return {
            "kind": kind,
            "key": key,
            "folder": str(folder),
            "job": raw,
            "job_text": job_text,
            "files": files,
            "result": st.result_of(folder) if kind == "completed" else {},
            "has_inputs": (folder / INPUTS).is_dir(),
            "has_resolved": (folder / RESOLVED).is_dir(),
            "settings_keys": sorted(project.backend.settings_keys()),
            "campaigns": _campaign_keys(project),
            "problems": (self._problems(key) if kind in ("queue", "hold") else []),
            "warnings": (project.warnings(local) if kind in ("queue", "hold") else []),
        }

    def file_spec(self, job_id: str, name: str) -> dict[str, Any]:
        """One queued study file as the details pane draws it."""
        data = self.details("queue", job_id)
        spec = next((f for f in data["files"] if f["name"] == name), None)
        if spec is None:
            raise GuiError(f"{name} is not a file of this study")
        return spec

    def log_tail(self, lines: int = 80) -> str:
        """The last lines of the runner's log."""
        path = self.workspace.state_dir / st.LOG_FILE
        if not path.exists():
            return ""
        with open(path, encoding="utf-8", errors="replace") as handle:
            return "".join(deque(handle, maxlen=lines))

    def describe(self, key: str) -> dict[str, Any]:
        """What a queued study would hand its solver. Rebuilds objects."""
        project, job_id = self._study(key)
        folder = project.queue.folder(job_id)
        return project.backend.describe(folder, project.job(job_id))

    # -- queue edits -----------------------------------------------------

    def add(self, kind: str, key: str, campaign: str = "") -> str:
        """Queue one study, or every study of a study sequence (in
        ``campaign`` when given)."""
        try:
            project, local = self.workspace.locate_task(key)
            if "::" in campaign:  # the Add bar's picker, several projects
                owner, _, campaign = campaign.rpartition("::")
                if owner != self.workspace.name_of(project):
                    raise GuiError(
                        f"campaign {campaign!r} is in {owner}, not in "
                        f"{self.workspace.name_of(project)}"
                    )
            if kind == "task":
                job_id = project.add(local, campaign=campaign)
                keys = self.workspace.added(project, [job_id])
                return f"queued {keys[0]}" + _warned(project, [job_id])
            spec = project.backend.catalog().sequence(local)
            if spec is None:
                raise GuiError(f"no {project.words.sequence} {key!r}")
            if campaign:
                spec = replace(spec, campaign=campaign)
            ids = project.add_sequence(spec)
            self.workspace.added(project, ids)
            return (
                f"queued {len(ids)} from {key} ({ids[0].split('-', 1)[0]})"
                + _warned(project, ids)
            )
        except QueueError as exc:
            raise GuiError(str(exc)) from exc

    def reorder(self, order: list[str]) -> None:
        """Put the queue in ``order`` (drag and drop)."""
        try:
            self.workspace.save_order([str(i) for i in order])
        except QueueError as exc:
            raise GuiError(str(exc)) from exc

    def move(self, job_id: str, delta: int) -> None:
        """Move a queued study up (-1) or down (+1)."""
        ids = self.workspace.ids()
        if job_id not in ids:
            raise GuiError(f"{job_id} is no longer queued")
        i = ids.index(job_id)
        j = i + delta
        if not 0 <= j < len(ids):
            raise GuiError("already at the end")
        ids[i], ids[j] = ids[j], ids[i]
        self.reorder(ids)

    def remove(self, job_id: str) -> str:
        """Take a study out of the queue or off hold (moved to
        out/removed/)."""
        try:
            if _split_hold(job_id) is not None:
                project, local = self._study(job_id)
            else:
                project, local = self.workspace.locate(job_id)
            dest = project.queue.remove(local)
        except (QueueError, OSError) as exc:
            raise GuiError(str(exc)) from exc
        return f"removed {job_id} (kept in {dest.parent.name}/{dest.name})"

    def save_as_template(self, kind: str, key: str, name: str) -> str:
        """Write a queued or completed study's MiniStudy as a template
        (and its new opt targets into the project), which then appears
        in the Add menu."""
        project, local = self._study(key)
        saver = getattr(project.backend, "save_template", None)
        if saver is None:
            raise GuiError("this backend has no templates")
        folder = (
            project.queue.folder(local)
            if kind in ("queue", "hold")
            else project.queue.completed_dir / local
        )
        try:
            job = parse_job(
                yaml.safe_load(job_file(folder).read_text("utf-8")),
                project.words,
            )
            paths = saver(folder, job, (name or "").strip())
        except Exception as exc:  # noqa: BLE001 -- shown to the user
            raise GuiError(f"not saved: {exc}") from exc
        self._deep.clear()
        root = project.out.parent
        shown = ", ".join(p.relative_to(root).as_posix() for p in paths)
        return f"saved {shown}; it is in the Add menu"

    def park(self, job_id: str) -> str:
        """Put a queued study on hold: out of the run order."""
        try:
            project, local = self.workspace.locate(job_id)
            project.park(local)
        except (QueueError, OSError) as exc:
            raise GuiError(str(exc)) from exc
        self._deep.pop(job_id, None)
        return f"{job_id} is on hold; the runner skips it until it is queued again"

    def hold_key_of(self, key: str) -> str:
        """The key a queued study (``key``) has once on hold."""
        head, _, local = str(key).rpartition("/")
        return f"{head}/{HOLD_MARK}/{local}" if head else f"{HOLD_MARK}/{local}"

    def unpark(self, key: str) -> str:
        """Move a study on hold to the end of the queue."""
        if _split_hold(key) is None:
            raise GuiError(f"{key} is not on hold")
        scoped, local = self._study(key)
        project = scoped._project  # pylint: disable=protected-access
        try:
            project.unpark(local)
        except (QueueError, OSError) as exc:
            raise GuiError(str(exc)) from exc
        self._deep.pop(key, None)
        queued = self.workspace.added(project, [local])[0]
        return f"queued {queued}" + _warned(project, [local])

    def save_job(self, job_id: str, fields: dict[str, Any]) -> str:
        """Change ``job.yaml`` fields of a queued study."""
        project, local = self._study(job_id)
        folder = project.queue.folder(local)
        try:
            job = load_job(folder, project.words)
        except JobError as exc:
            raise GuiError("; ".join(exc.problems)) from exc
        raw = job.to_raw(project.words)
        raw.update(fields)
        if not raw.get("campaign"):
            raw.pop("campaign", None)
        elif raw["campaign"] != job.campaign:
            bad = campaign_problems(project.root, raw["campaign"])
            if bad:
                raise GuiError("not saved: " + bad[0])
        try:
            parse_job(raw, project.words, folder_id=local)
        except JobError as exc:
            raise GuiError("not saved: " + "; ".join(exc.problems)) from exc
        self._write(job_id, JOB_FILE, dump_yaml(raw))
        if self._taken(job_id):
            return "saved; " + TAKEN
        problems = project.problems(local)
        return "saved" + (f"; it cannot run yet: {problems[0]}" if problems else "")

    def save_file(self, job_id: str, name: str, text: str) -> str:
        """Replace one of a queued study's files.

        Text that does not parse is refused; anything that parses is
        written and then checked (objects rebuilt), the result reported.
        """
        project, _ = self._study(job_id)
        names = {spec.name for spec in project.backend.files()}
        if name not in names:
            raise GuiError(f"{name} is not a file of this study")
        try:
            if name.endswith(".json"):
                json.loads(text)
            else:
                yaml.safe_load(text)
        except (json.JSONDecodeError, yaml.YAMLError) as exc:
            raise GuiError(f"not saved, {name} does not parse: {exc}") from exc
        self._write(job_id, name, text)
        if self._taken(job_id):
            return f"saved {name}; " + TAKEN
        problems = self._deep_check(job_id)
        return f"saved {name}" + (
            f"; it cannot run yet: {'; '.join(problems[:2])}"
            if problems
            else "; it checks out"
        )

    def save_form(
        self,
        key: str,
        name: str,
        edits: dict | None = None,
        overrides: dict | None = None,
    ) -> str:
        """Write a form's edits into one of a queued study's files.

        Args:
            key: The study.
            name: The file.
            edits: ``{forms.key(path): input}`` for a structured file.
            overrides: ``{name: (set here, input)}`` for an optional
                file drawn as overrides of the project's values; none
                set deletes the file, so all of it comes from the
                project.
        """
        project, local = self._study(key)
        path = project.queue.folder(local) / name
        old = path.read_text(encoding="utf-8") if path.exists() else None
        try:
            if overrides is not None:
                data = forms.apply_overrides(
                    _project_values(
                        project,
                        name,
                        _campaign_of(project, project.queue.folder(local)),
                    )
                    or {},
                    overrides,
                )
                if not data:
                    if old is not None:
                        self.delete_optional(key, name)
                    return (
                        f"{name}: nothing set here, so every value comes "
                        f"from the project"
                    )
            else:
                data = forms.apply(_parse(name, old or ""), edits or {})
        except forms.FormError as exc:
            raise GuiError(f"not saved: {exc}") from exc
        return self.save_file(key, name, _render(name, data, old))

    def form_op(
        self,
        key: str,
        name: str,
        path: list,
        op: str,
        edits: dict | None = None,
    ) -> str:
        """Duplicate, remove or switch off one list item of a file (an
        opt target, a nested study), keeping the form's other edits."""
        project, local = self._study(key)
        file = project.queue.folder(local) / name
        if not file.exists():
            raise GuiError(f"{name} is not in this study")
        old = file.read_text(encoding="utf-8")
        try:
            data = forms.apply(_parse(name, old), edits or {})
            data = forms.operate(data, tuple(path), op)
        except forms.FormError as exc:
            raise GuiError(f"not saved: {exc}") from exc
        done = {
            "remove": "removed",
            "duplicate": "duplicated",
            "toggle": "switched",
        }[op]
        return f"{done}; " + self.save_file(key, name, _render(name, data, old))

    def form_add(
        self,
        key: str,
        name: str,
        path: list,
        choice: str,
        edits: dict | None = None,
    ) -> str:
        """Add a new, blank item (``backend.make_item``) to a list of one
        of a study's files, keeping the form's other edits."""
        project, local = self._study(key)
        maker = getattr(project.backend, "make_item", None)
        if maker is None:
            raise GuiError("this backend cannot add items")
        if not choice:
            raise GuiError("pick what to add first")
        file = project.queue.folder(local) / name
        old = file.read_text(encoding="utf-8")
        try:
            item = maker(name, choice)
            data = forms.apply(_parse(name, old), edits or {})
            data = forms.add_item(data, tuple(path), item)
        except forms.FormError as exc:
            raise GuiError(f"not added: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 -- shown to the user
            raise GuiError(f"not added: {type(exc).__name__}: {exc}") from exc
        return "added; fill it in. " + self.save_file(
            key, name, _render(name, data, old)
        )

    def load_targets(
        self,
        key: str,
        name: str,
        template: str,
        edits: dict | None = None,
    ) -> str:
        """Append one of the project's opt targets, by name, to a
        study's."""
        project, local = self._study(key)
        loader = getattr(project.backend, "template_targets", None)
        if loader is None:
            raise GuiError("this backend has no target templates")
        if not template:
            raise GuiError("pick an opt target first")
        file = project.queue.folder(local) / name
        old = file.read_text(encoding="utf-8")
        try:
            entries = loader(template)
            data = forms.apply(_parse(name, old), edits or {})
            for entry in entries:
                data = forms.add_item(data, ("targets",), entry)
        except forms.FormError as exc:
            raise GuiError(f"not added: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 -- shown to the user
            raise GuiError(f"not added: {exc}") from exc
        return f"added {len(entries)} target(s) from {template}; " + (
            self.save_file(key, name, _render(name, data, old))
        )

    def save_targets_template(
        self,
        key: str,
        name: str,
        template: str,
        edits: dict | None = None,
    ) -> str:
        """Add a study's opt targets (form edits included) to the
        project's: those it does not have yet, as
        ``opt_targets/<template>.yaml``."""
        project, local = self._study(key)
        saver = getattr(project.backend, "save_targets_template", None)
        if saver is None:
            raise GuiError("this backend has no target templates")
        if edits:
            self.save_form(key, name, edits=edits)
        try:
            path = saver(project.queue.folder(local), (template or "").strip())
        except Exception as exc:  # noqa: BLE001 -- shown to the user
            raise GuiError(f"not saved: {exc}") from exc
        root = project.out.parent
        return (
            f"saved {path.relative_to(root).as_posix()}; a new study "
            f"takes these when its modes fit, and any study can add them "
            f"from its Opt targets tab"
        )

    def create_optional(self, job_id: str, name: str) -> None:
        """Write an empty optional file (only what you add then counts)."""
        project, _ = self._study(job_id)
        spec = next(
            (s for s in project.backend.files() if s.name == name and s.optional),
            None,
        )
        if spec is None:
            raise GuiError(f"{name} is not an optional file")
        text = (
            "{}\n"
            if name.endswith(".json")
            else dump_yaml(
                {},
                f"{spec.label} for this study only. Add entries; the rest "
                f"come from the project when it starts.",
            )
        )
        self._write(job_id, name, text)

    def delete_optional(self, job_id: str, name: str) -> None:
        """Delete an optional file, so all of it comes from the project."""
        project, local = self._study(job_id)
        spec = next(
            (s for s in project.backend.files() if s.name == name and s.optional),
            None,
        )
        if spec is None:
            raise GuiError(f"{name} is not an optional file")
        if not project.queue.folder(local).is_dir():
            raise GuiError(
                f"{job_id} is no longer in the queue (it may "
                f"have just started); nothing was deleted"
            )
        try:
            project.queue.delete_file(local, name)
        except QueueError as exc:
            raise GuiError(
                f"{job_id} is no longer in the queue (it may "
                f"have just started); nothing was deleted"
            ) from exc

    def edit_copy(self, folder_name: str) -> str:
        """A copy of a completed study, as it was queued, made on hold
        (to be edited, then queued). Returns its key."""
        project, local = self._loc(folder_name)
        try:
            new_id = project.hold.requeue(
                project.queue.completed_dir / local, project.words
            )
        except QueueError as exc:
            raise GuiError(str(exc)) from exc
        return self._hold_key(project, new_id)

    def requeue(self, folder_name: str, exact: bool = False) -> str:
        """Queue a completed study again as a fresh copy: as it was
        queued (project values pulled again), or ``exact``ly as it ran."""
        project, local = self._study(folder_name)
        try:
            new_id = project.queue.requeue(
                project.queue.completed_dir / local, project.words, exact
            )
        except QueueError as exc:
            raise GuiError(str(exc)) from exc
        return f"queued {self.workspace.added(project, [new_id])[0]}"

    # -- the editor: templates, study sequences, campaigns ----------------

    def template_view(self, key: str) -> dict[str, Any]:
        """A study template, for the editor: what it is and what it runs."""
        try:
            project, local = self.workspace.locate_task(key)
        except QueueError as exc:
            raise GuiError(str(exc)) from exc
        spec = project.backend.catalog().task(local)
        if spec is None:
            raise GuiError(f"no {project.words.task} {key!r}")
        return {
            "key": key,
            "label": spec.label,
            "subject": spec.subject,
            "family": spec.family,
            "summary": spec.summary,
            "campaigns": _campaign_keys(project),
        }

    def new_study(
        self, task: str = "", name: str = "", campaign: str = "", project: str = ""
    ) -> str:
        """Make a study on hold, from template ``task`` (a catalog key) or,
        with no task, from scratch named ``name``. Returns its key."""
        try:
            if task:
                target, local = self.workspace.locate_task(task)
                job_id = target.add(local, campaign=campaign, hold=True)
            else:
                target = self._project_for(project)
                job_id = target.add_blank(name, hold=True, campaign=campaign)
        except QueueError as exc:
            raise GuiError(str(exc)) from exc
        return self._hold_key(target, job_id)

    def sequence_view(self, key: str = "", project: str = "") -> dict[str, Any]:
        """A study sequence template for the editor: its file's mapping,
        whether it can be saved under this name, and the choices. With
        no key: a new, empty one."""
        if key:
            try:
                target, local = _locate_sequence(self.workspace, key)
                raw = target.sequence_raw(local)
            except QueueError as exc:
                raise GuiError(str(exc)) from exc
            from_code = target.sequence_from_code(local)
        else:
            target = self._project_for(project)
            local, raw, from_code = "", {"studies": []}, False
        words = target.words
        catalog = target.backend.catalog()
        studies = []
        for item in raw.get("studies") or []:
            if isinstance(item, str):
                item = {words.task: item}
            item = dict(item) if isinstance(item, dict) else {}
            task = str(item.pop(words.task, None) or item.pop("task", None) or "")
            studies.append(
                {
                    "task": task,
                    "settings": _flow(item.pop("settings", None)),
                    "notes": str(item.pop("notes", "") or ""),
                    # Per-study checks, save, on_failure: kept as written.
                    "extra": item,
                }
            )
        defaults: dict = (
            raw["defaults"] if isinstance(raw.get("defaults"), dict) else {}
        )
        checks = defaults.get("checks") or []
        if isinstance(checks, str):  # one name, as the file may write it
            checks = [checks]
        # What the editor has no field for (notes, keys of later versions):
        # written back as it was.
        shown_top = {
            "version",
            "name",
            "why_ordered",
            "campaign",
            "defaults",
            "studies",
        }
        shown_defaults = {"settings", "checks", "save", "on_failure"}
        keep = {
            "top": {k: v for k, v in raw.items() if k not in shown_top},
            "defaults": {k: v for k, v in defaults.items() if k not in shown_defaults},
        }
        return {
            "key": local,
            "project": self.workspace.name_of(target),
            "new": not key,
            "from_code": from_code,
            "name": str(raw.get("name") or ""),
            "why_ordered": str(raw.get("why_ordered") or ""),
            "campaign": str(raw.get("campaign") or ""),
            "defaults": {
                "settings": _flow(defaults.get("settings")),
                "checks": ", ".join(str(c) for c in checks),
                "save": bool(defaults.get("save")),
                "on_failure": defaults.get("on_failure") or "abort",
            },
            "studies": studies,
            "keep": keep,
            "tasks": [
                {"value": t.key, "label": f"{t.label} ({t.key})"} for t in catalog.tasks
            ],
            "campaigns": _campaign_keys(target),
            "check_names": sorted(target.backend.check_names()),
        }

    def save_sequence(
        self, key: str, form: dict, overwrite: bool, project: str = ""
    ) -> str:
        """Write a study sequence template from the editor's ``form``
        (``sequence_view``'s shape). Returns what was written."""
        target = self._project_for(project)
        words = target.words
        try:
            defaults: dict[str, Any] = {}
            settings = _mapping(form["defaults"].get("settings"), "default settings")
            if settings:
                defaults["settings"] = settings
            checks = [
                c.strip()
                for c in (form["defaults"].get("checks") or "").split(",")
                if c.strip()
            ]
            if checks:
                defaults["checks"] = checks
            defaults["save"] = bool(form["defaults"].get("save"))
            defaults["on_failure"] = form["defaults"].get("on_failure") or "abort"
            studies = []
            for n, row in enumerate(form.get("studies") or [], 1):
                if not row.get("task"):
                    raise GuiError(f"study {n}: pick which {words.task} it is")
                item: dict[str, Any] = {words.task: row["task"]}
                row_settings = _mapping(row.get("settings"), f"study {n}'s settings")
                if row_settings:
                    item["settings"] = row_settings
                item.update(row.get("extra") or {})
                if row.get("notes"):
                    item["notes"] = row["notes"]
                studies.append(item)
        except ValueError as exc:
            raise GuiError(str(exc)) from exc
        if not studies:
            raise GuiError(f"add at least one {words.task} first")
        keep = form.get("keep") or {}
        raw: dict[str, Any] = {"name": form.get("name") or key}
        if form.get("why_ordered"):
            raw["why_ordered"] = form["why_ordered"]
        if form.get("campaign"):
            raw["campaign"] = form["campaign"]
        raw.update(keep.get("top") or {})
        raw["defaults"] = {**defaults, **(keep.get("defaults") or {})}
        raw["studies"] = studies
        try:
            path = target.save_sequence(key, raw, overwrite=overwrite)
        except QueueError as exc:
            raise GuiError(str(exc)) from exc
        return f"saved {_relative(target, path)}"

    def campaign_view(self, key: str, project: str = "") -> dict[str, Any]:
        """A campaign for the editor: its README, the store its studies
        use, and how many studies of it are queued, on hold or done."""
        target = self._project_for(project)
        found = {c.key: c for c in target.campaigns()}
        if key not in found:
            raise GuiError(f"no campaign {key!r}")
        info = found[key]
        readme = info.folder / "README.md"
        counts = {"queued": 0, "on hold": 0, "completed": 0}
        for name, queue in (("queued", target.queue), ("on hold", target.hold)):
            for local in queue.ids():
                try:
                    if load_job(queue.folder(local), target.words).campaign == key:
                        counts[name] += 1
                except JobError:
                    pass
        for path in target.queue.completed():
            if (st.result_of(path) or {}).get("campaign") == key:
                counts["completed"] += 1
        parent = key.rpartition("/")[0]
        return {
            "key": key,
            "project": self.workspace.name_of(target),
            "title": info.title,
            "readme": readme.read_text(encoding="utf-8") if readme.is_file() else "",
            "folder": _relative(target, info.folder),
            "store": _relative(target, target.campaign_store(key)),
            "own_store": info.own_store,
            "promote_to": _relative(target, target.campaign_store(parent)),
            "counts": counts,
        }

    def preview(self, kind: str, key: str) -> list[dict[str, Any]]:
        """Pictures of what a study simulates, one per solver run (its
        components and the airbridges that solve adds), for the details
        pane."""
        project, local = self._study(key)
        hook = getattr(project.backend, "preview", None)
        if hook is None:
            raise GuiError("this backend cannot draw its studies")
        folder = (
            project.queue.folder(local)
            if kind in ("queue", "hold")
            else project.queue.completed_dir / local
        )
        try:
            return hook(folder)
        except Exception as exc:  # noqa: BLE001 -- shown to the user
            raise GuiError(f"cannot draw it: {type(exc).__name__}: {exc}") from exc

    def check_all(self) -> dict[str, list[str]]:
        """Every queued study's problems, objects rebuilt (no solve)."""
        return {k: self._deep_check(k) for k in self.workspace.ids()}

    # -- running ---------------------------------------------------------

    def start(self) -> str:
        """Launch the runner as a detached process.

        The GUI only starts it: the run is tracked through runner.json,
        outlives the window, and looks the same as one started from a
        terminal.
        """
        workspace = self.workspace
        out = workspace.state_dir
        if st.lock_held(out) or st.RunnerState.load(out).alive(out):
            raise GuiError("the runner is already going")
        if st.lock_held(user_home()):
            raise GuiError(
                "another runner is already going on this machine; start "
                "this queue when it is done"
            )
        if not workspace.ids():
            raise GuiError("the queue is empty")
        if not workspace.refs:
            raise GuiError("this project was not opened from a file")
        args = [sys.executable, "-m", "qdesignoptimizer.sequencer", workspace.refs[0]]
        for ref in workspace.refs[1:]:
            args += ["-p", ref]
        args += ["run", "--yes"]
        out.mkdir(parents=True, exist_ok=True)
        flags = 0
        if os.name == "nt":
            flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        with open(out / "launch.log", "a", encoding="utf-8") as log:
            log.write(f"--- {st.now()} {' '.join(args)}\n")
            log.flush()

            def launch(creationflags):
                subprocess.Popen(
                    args,
                    cwd=str(Path.cwd()),
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    creationflags=creationflags,
                    close_fds=True,
                )

            if os.name == "nt":
                # Out of the GUI's job object, so closing the terminal
                # the GUI was started from does not kill a running
                # solve. A job that forbids breaking away refuses the
                # flag; then start inside it, as before.
                try:
                    launch(flags | subprocess.CREATE_BREAKAWAY_FROM_JOB)
                except OSError:
                    launch(flags)
            else:
                launch(flags)
        return "runner started"

    def stop(self) -> str:
        """Ask the runner to stop once the current study is done."""
        out = self.workspace.state_dir
        runner = st.RunnerState.load(out)
        if not runner.alive(out):
            raise GuiError("the runner is not going")
        st.request_stop(out)
        return (
            f"stop requested: {runner.current or 'the current study'} "
            f"finishes first"
        )


def _folder_of(ref: str) -> Path:
    """The project folder a connected path names (a ``sequencer.py``
    sits in the folder or its ``simulations/``)."""
    text = str(ref)
    if ".py:" in text:  # file.py:factory
        text = text.split(".py:", 1)[0] + ".py"
    path = Path(text).resolve()
    if path.suffix == ".py":
        path = path.parent
        if path.name == "simulations":
            path = path.parent
    return path


def _parse(name: str, text: str):
    """A study file's data (JSON or YAML)."""
    if name.endswith(".json"):
        return json.loads(text) if text.strip() else {}
    return yaml.safe_load(text) if text.strip() else {}


def _render(name: str, data, old: str | None) -> str:
    """A study file's text from its data, keeping its comment header."""
    if name.endswith(".json"):
        return json.dumps(data, indent=4) + "\n"
    return dump_yaml(data, forms.header(old or "") or None)


def _locate_sequence(workspace, key: str):
    """The project and own key of a study sequence; a template file saved
    a moment ago counts even before every catalog lists it."""
    try:
        return workspace.locate_task(key)
    except QueueError:
        head, _, local = str(key).rpartition("/")
        projects = (
            [workspace.projects[head]]
            if head in workspace.projects
            else list(workspace.projects.values()) if not head else []
        )
        for project in projects:
            folder = project.sequence_dir
            if folder is not None and (folder / f"{local}.yaml").is_file():
                return project, local
        raise


def _flow(value) -> str:
    """A mapping as one line of YAML (``iterations: 3, seconds: 4``) for a
    text field; "" for none."""
    if not value:
        return ""
    if not isinstance(value, dict):
        raise GuiError(f"{value!r} is not a mapping (key: value, ...)")
    text = yaml.safe_dump(dict(value), default_flow_style=True, sort_keys=False).strip()
    return text[1:-1] if text.startswith("{") else text


def _mapping(text, what: str) -> dict:
    """A one-line YAML mapping typed into a text field (``_flow``'s
    form), as a dict.

    Raises:
        ValueError: If it does not read as a mapping.
    """
    text = (text or "").strip()
    if not text:
        return {}
    try:
        value = yaml.safe_load("{" + text + "}" if not text.startswith("{") else text)
    except yaml.YAMLError as exc:
        raise ValueError(f"{what}: not read as name: value pairs ({exc})") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{what}: write name: value pairs")
    return numbers(value)


def _relative(project, path) -> str:
    """``path`` relative to the project's folder where it can be."""
    if path is None:
        return ""
    try:
        return Path(path).resolve().relative_to(project.root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _campaign_keys(project) -> list[str]:
    """The project's campaign keys, parents first ([] if unreadable)."""
    try:
        return [c.key for c in project.campaigns()]
    except Exception:  # noqa: BLE001 -- the picker is then empty
        return []


#: Said after a save that the runner took the study right after: the
#: edit is in, but a later one to the same study would not be.
TAKEN = (
    "the runner took the study right after, with this edit; any further "
    "edit is too late (requeue it to change more)"
)


def _campaign_of(project, folder) -> str:
    """The campaign a study's ``job.yaml`` names ("" for none, or when
    it cannot be read)."""
    try:
        return load_job(folder, project.words).campaign
    except Exception:  # noqa: BLE001 -- its problems are shown elsewhere
        return ""


def _project_values(project, name: str, campaign: str = "") -> dict | None:
    """The project's values an optional file overrides (a campaign
    study's: its campaign's store), if the backend says
    (``project_values``)."""
    hook = getattr(project.backend, "project_values", None)
    if hook is None:
        return None
    try:
        values = hook(name, campaign=campaign) if campaign else hook(name)
    except Exception:  # noqa: BLE001 -- the form says it cannot list them
        return None
    return dict(values) if values is not None else None


def _addable(project, name: str) -> dict:
    """``{forms.key(list path): choices}``: what can be added where."""
    hook = getattr(project.backend, "item_choices", None)
    if hook is None:
        return {}
    try:
        return {forms.key(p): c for p, c in (hook(name) or {}).items()}
    except Exception:  # noqa: BLE001 -- nothing can be added then
        return {}


def _target_templates(project) -> list | None:
    hook = getattr(project.backend, "target_templates", None)
    if hook is None:
        return None
    try:
        return list(hook())
    except Exception:  # noqa: BLE001
        return []


def _value_sources(project, name: str, campaign: str = "") -> dict:
    """Which study last wrote each project value, if the backend keeps
    that record (``value_sources``)."""
    hook = getattr(project.backend, "value_sources", None)
    if hook is None:
        return {}
    try:
        return dict((hook(name, campaign=campaign) if campaign else hook(name)) or {})
    except Exception:  # noqa: BLE001 -- the column is left out
        return {}


#: Marks a study on hold in a key: ``@hold/<id>``, or
#: ``<project>/@hold/<id>`` with several projects open.
HOLD_MARK = "@hold"


def _split_hold(key) -> tuple[str, str] | None:
    """``(project, id)`` of a key of a study on hold, None for any other
    key. ``project`` is "" when the key names none."""
    key = str(key)
    if key.startswith(HOLD_MARK + "/"):
        return "", key[len(HOLD_MARK) + 1 :]
    marker = f"/{HOLD_MARK}/"
    if marker in key:
        head, local = key.split(marker, 1)
        if "/" not in head:
            return head, local
    return None


class _Scoped:
    """A project seen through its hold folder: its ``queue`` is
    ``out/hold/``, so every editing method works on a study on hold
    unchanged."""

    def __init__(self, project: Project, queue):
        self._project = project
        self.queue = queue
        self.backend = project.backend
        self.words = project.words
        self.out = project.out
        self.key_prefix = ""

    def job(self, job_id: str):
        return load_job(self.queue.folder(job_id), self.words)

    @property
    def root(self) -> Path:
        return self._project.root

    def campaigns(self):
        return self._project.campaigns()

    def campaign_store(self, key: str):
        return self._project.campaign_store(key)

    def warnings(self, job_id: str) -> list[str]:
        return self._project.warnings(job_id)

    def problems(self, job_id: str, deep: bool = False, folder=None):
        return self._project.problems(
            job_id, deep=deep, folder=folder or self.queue.folder(job_id)
        )


def _warned(project, ids) -> str:
    """`` -- warning: ...`` for the first of ``ids`` that has one."""
    for job_id in ids:
        found = project.warnings(job_id)
        if found:
            more = sum(1 for i in ids if project.warnings(i)) - 1
            extra = f" (and {more} more)" if more > 0 else ""
            return f" -- warning, {job_id}{extra}: {found[0]}"
    return ""
