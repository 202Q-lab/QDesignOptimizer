"""Bridge between the GUI and the projects' folders. Imports no Dash.

Nothing is cached but the loaded backends: every poll re-reads the
queue folders, ``order.json``, ``runner.json`` and the completed
results, because the runner, a shell and an agent change them too.

Several projects share one queue through a ``Workspace``; a study's key
is then ``<project>/<id>`` (a plain id when one project is open).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path
from typing import Any

import yaml

from ..engine import state as st
from ..engine.job import (
    JOB_FILE,
    JobError,
    dump_yaml,
    job_file,
    load_job,
    parse_completed,
    parse_job,
)
from ..engine.project import Project
from ..engine.queue import INPUTS, RESOLVED, QueueError
from ..engine.runner import ACTIVE_FILE
from ..engine.workspace import Workspace, open_workspace, user_home
from . import forms


class GuiError(RuntimeError):
    """An action the GUI refuses, with the reason."""


class Bridge:
    """The open project(s), as the GUI sees them.

    Args:
        ref: A project's ``sequencer.py`` or folder, or a list of them
            (several projects sharing one queue).
        project: Injected project (tests); else opened from ``ref``.
        workspace: Injected workspace (tests).
    """

    def __init__(
        self,
        ref: str | list = "",
        project: Project | None = None,
        workspace: Workspace | None = None,
    ):
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
        # Launches running in threads of their own: key -> status.
        self._launches: dict[str, dict] = {}
        self._launch_guard = threading.Lock()
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
            self._workspace = open_workspace(self.refs, isolate=True)
            # A launch a crash cut short is finished or undone now, not
            # only when a runner next starts: its studies are hidden
            # until then.
            for project in self._workspace.projects.values():
                try:
                    project.recover_launches()
                except Exception:  # noqa: BLE001 -- the runner retries
                    pass
        return self._workspace

    @property
    def project(self) -> Project:
        """The project, when one is open."""
        workspace = self.workspace
        if not workspace.single:
            raise GuiError("several projects are open; name one")
        return next(iter(workspace.projects.values()))

    def _study(self, key: str):
        """The project (scoped to a draft for a draft study's key) and
        the study's own id."""
        parts = _split_draft(key)
        if parts is not None:
            project, name = self._draft(key)
            scoped = _Scoped(project, project.draft_queue(name))
            scoped.draft_key = str(key).rsplit("/", 1)[0]
            return scoped, parts[2]
        return self._loc(key)

    def _draft(self, key: str) -> tuple[Project, str]:
        """The project and draft name of a draft's (or draft study's)
        key: ``~<draft>`` or ``<project>/~<draft>``, plus ``/<id>``."""
        parts = _split_draft(key)
        if parts is None:
            raise GuiError(f"{key!r} is not a draft")
        head, draft, _ = parts
        workspace = self.workspace
        if head:
            if head not in workspace.projects:
                raise GuiError(f"no project {head!r} is open")
            project = workspace.projects[head]
        elif workspace.single:
            project = next(iter(workspace.projects.values()))
        else:
            raise GuiError(f"{key!r}: name the project")
        try:
            project.draft_queue(draft)
        except QueueError as exc:
            raise GuiError(str(exc)) from exc
        return project, draft

    def _draft_key(self, project: Project, name: str) -> str:
        return self.workspace.key(self.workspace.name_of(project), DRAFT_MARK + name)

    def _loc(self, key: str) -> tuple[Project, str]:
        """The project and its own id (or folder name) for a key."""
        try:
            return self.workspace.locate(key)
        except QueueError as exc:
            raise GuiError(str(exc)) from exc

    def _key(self, project, local: str) -> str:
        if isinstance(project, _Scoped):
            return f"{project.draft_key}/{local}"
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
                    f"{job_id} is no longer in the queue (it may have just "
                    f"started); nothing was saved"
                ) from exc
            # Still queued: the write itself failed (a file held open by
            # the virus scanner, a full disk). Say that, not "gone".
            raise GuiError(
                f"could not save {name}: {type(exc).__name__}: {exc}; " f"try again"
            ) from exc

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
                        "campaign": "",
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
                    "campaign": job.campaign.get("id", ""),
                    "campaign_name": job.campaign.get("name", ""),
                    "settings": ", ".join(f"{k} {v}" for k, v in job.settings.items()),
                    "save": job.save,
                    "hold": job.hold,
                    "notes": job.notes,
                    "problems": self._problems(key),
                    "warnings": project.warnings(job_id),
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
                    "kind": "campaign",
                    "key": workspace.key(name, c.key),
                    "label": prefix + c.label,
                    "count": len(c.entries),
                }
                for c in catalog.campaigns
            ]
        return {
            "project": (
                str(first.backend.out_dir.parent)
                if workspace.single
                else "; ".join(str(p.out.parent) for p in workspace.projects.values())
            ),
            "multi": not workspace.single,
            "projects": list(workspace.projects),
            "catalog_problems": problems,
            "words": {
                k: getattr(words, k)
                for k in ("task", "campaign", "subject", "store", "live")
            },
            "rows": rows,
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
        }

    def details(self, kind: str, key: str) -> dict[str, Any]:
        """One study's files and record, for the details pane.

        Args:
            kind: ``queue`` (a queued id) or ``completed`` (a folder).
            key: The id or the folder name.
        """
        project, local = self._study(key)
        if kind in ("queue", "draft"):
            folder = project.queue.folder(local)
        else:
            folder = project.queue.completed_dir / local
        if not folder.is_dir():
            raise GuiError(f"{key} is no longer there")
        specs = project.backend.files()
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
                        _project_values(project, spec.name)
                        if spec.optional and kind in ("queue", "draft")
                        else None
                    ),
                    "sources": (
                        _value_sources(project, spec.name) if spec.optional else {}
                    ),
                    "addable": (
                        _addable(project, spec.name)
                        if kind in ("queue", "draft")
                        else {}
                    ),
                    "target_templates": (
                        _target_templates(project)
                        if kind in ("queue", "draft")
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
        return {
            "kind": kind,
            "key": key,
            "folder": str(folder),
            "job": job.to_raw(project.words) if job else {},
            "job_text": job_text,
            "files": files,
            "result": st.result_of(folder) if kind == "completed" else {},
            "has_inputs": (folder / INPUTS).is_dir(),
            "has_resolved": (folder / RESOLVED).is_dir(),
            "settings_keys": sorted(project.backend.settings_keys()),
            "problems": (self._problems(key) if kind in ("queue", "draft") else []),
            "warnings": (project.warnings(local) if kind in ("queue", "draft") else []),
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

    def add(self, kind: str, key: str) -> str:
        """Queue one study, or every study of a campaign."""
        try:
            project, local = self.workspace.locate_task(key)
            if kind == "task":
                job_id = project.add(local)
                keys = self.workspace.added(project, [job_id])
                return f"queued {keys[0]}" + _warned(project, [job_id])
            spec = project.backend.catalog().campaign(local)
            if spec is None:
                raise GuiError(f"no {project.words.campaign} {key!r}")
            ids = project.add_campaign(spec)
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
        """Take a study out of the queue (moved to out/removed/)."""
        try:
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
            if kind in ("queue", "draft")
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

    def toggle_hold(self, job_id: str) -> str:
        """Put a queued study on hold, or release it."""
        try:
            project, local = self.workspace.locate(job_id)
            hold = not project.job(local).hold
            project.set_hold(local, hold)
        except (JobError, QueueError, OSError) as exc:
            raise GuiError(str(exc)) from exc
        return f"{job_id} {'on hold' if hold else 'released'}"

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
        try:
            parse_job(raw, project.words, folder_id=local)
        except JobError as exc:
            raise GuiError("not saved: " + "; ".join(exc.problems)) from exc
        self._write(job_id, JOB_FILE, dump_yaml(raw))
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
                    _project_values(project, name) or {}, overrides
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
        project.queue.delete_file(local, name)

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

    # -- drafts (a campaign put together before it is queued) ------------

    def drafts(self) -> list[dict]:
        """Every draft of every open project."""
        out = []
        for name, project in self.workspace.projects.items():
            for draft in project.drafts():
                out.append(
                    {
                        **draft,
                        "key": self._draft_key(project, draft["name"]),
                        "project": name,
                    }
                )
        return out

    def new_draft(
        self, project: str = "", label: str = "", from_campaign: str = ""
    ) -> str:
        """Start a draft campaign. Returns its key.

        Args:
            project: The project's name (only needed with several open).
            label: The campaign's name.
            from_campaign: A campaign key: its studies, as a start.
        """
        workspace = self.workspace
        try:
            if from_campaign:
                target, key = workspace.locate_task(from_campaign)
                spec = target.backend.catalog().campaign(key)
                if spec is None:
                    raise GuiError(f"no campaign {from_campaign!r}")
            else:
                spec = None
                if project:
                    target = workspace.projects[project]
                elif workspace.single:
                    target = next(iter(workspace.projects.values()))
                else:
                    raise GuiError("pick the project the campaign is for")
            name = target.new_draft(label, spec)
        except (QueueError, KeyError) as exc:
            raise GuiError(str(exc)) from exc
        return self._draft_key(target, name)

    def draft_view(self, key: str) -> dict[str, Any]:
        """A draft as the popup draws it."""
        project, name = self._draft(key)
        meta = next(d for d in project.drafts() if d["name"] == name)
        queue = project.draft_queue(name)
        scoped = _Scoped(project, queue)
        scoped.draft_key = key
        catalog = project.backend.catalog()
        studies = []
        for n, local in enumerate(queue.ids(), 1):
            study_key = f"{key}/{local}"
            try:
                job = scoped.job(local)
            except JobError as exc:
                studies.append({"key": study_key, "n": n, "problems": exc.problems})
                continue
            spec = catalog.task(job.task)
            studies.append(
                {
                    "key": study_key,
                    "n": n,
                    "task": job.task,
                    "label": spec.label if spec else job.task,
                    "settings": ", ".join(f"{k} {v}" for k, v in job.settings.items()),
                    "save": job.save,
                    "problems": self._problems(study_key),
                }
            )
        return {
            "key": key,
            "label": meta.get("label", ""),
            "from": meta.get("from", ""),
            "project": self.workspace.name_of(project),
            "studies": studies,
            "tasks": [
                {"value": t.key, "label": f"{t.label} ({t.key})"} for t in catalog.tasks
            ],
        }

    def draft_add(self, key: str, task: str) -> str:
        """Add one study to a draft. Returns the study's key."""
        project, name = self._draft(key)
        try:
            local = project.draft_add(name, task)
        except QueueError as exc:
            raise GuiError(str(exc)) from exc
        return f"{key}/{local}"

    def draft_move(self, study_key: str, delta: int) -> None:
        """Move a draft study up (-1) or down (+1)."""
        project, local = self._study(study_key)
        ids = project.queue.ids()
        i = ids.index(local)
        j = i + delta
        if not 0 <= j < len(ids):
            raise GuiError("already at the end")
        ids[i], ids[j] = ids[j], ids[i]
        project.queue.save_order(ids)

    def draft_blank(self, key: str, name: str = "") -> str:
        """Add a study made from scratch to a draft. Returns its key."""
        project, draft = self._draft(key)
        try:
            local = project.draft_blank(draft, name)
        except QueueError as exc:
            raise GuiError(str(exc)) from exc
        return f"{key}/{local}"

    def draft_remove(self, study_key: str) -> str:
        """Take a study out of a draft (kept in out/removed/)."""
        project, local = self._study(study_key)
        try:
            project.queue.remove(local)
        except (QueueError, OSError) as exc:
            raise GuiError(str(exc)) from exc
        return f"removed {local} from the campaign"

    def draft_label(self, key: str, label: str) -> None:
        """Rename a draft campaign."""
        project, name = self._draft(key)
        project.set_draft_label(name, label)

    def start_launch(self, key: str, label: str = "") -> str:
        """Launch a draft in a thread of its own; ``launch_status`` says
        how far it got. Checking every study builds the design, which
        takes seconds per study, so the GUI does not wait for it.

        The building itself runs in the project's worker process; this
        thread only waits for it. If the GUI is closed mid-launch, the
        launch is finished or undone when it opens again.
        """
        project, name = self._draft(key)
        with self._launch_guard:
            current = self._launches.get(key)
            if current and current["state"] == "running":
                raise GuiError("this campaign is already being launched")
            status = {
                "state": "running",
                "checked": 0,
                "total": len(project.draft_queue(name).ids()),
                "current": "",
                "message": "",
                "started": st.now(),
            }
            self._launches[key] = status

        def progress(checked, total, study):
            status.update(checked=checked, total=total, current=study)

        def work():
            try:
                text = self.launch_draft(key, label, progress=progress)
                status.update(state="done", message=text)
            except GuiError as exc:
                status.update(state="failed", message=str(exc))
            except Exception as exc:  # noqa: BLE001 -- shown in the popup
                status.update(
                    state="failed",
                    message=f"{type(exc).__name__}: {exc}",
                )

        threading.Thread(target=work, name=f"launch {key}", daemon=True).start()
        return f"launching {key}: checking its studies"

    def launch_status(self, key: str) -> dict | None:
        """How a launch started here stands (None: none was)."""
        status = self._launches.get(key)
        return dict(status) if status else None

    def forget_launch(self, key: str) -> None:
        """Drop a finished launch's status (once it has been shown)."""
        with self._launch_guard:
            status = self._launches.get(key)
            if status and status["state"] != "running":
                del self._launches[key]

    def launch_shown(self, key: str) -> bool:
        """Mark a failed launch's reason as announced; True the first
        time (the GUI then says it once, and keeps showing the reason in
        the popup until it closes)."""
        with self._launch_guard:
            status = self._launches.get(key)
            if not status or status.get("shown"):
                return False
            status["shown"] = True
            return True

    def forget_shown_launches(self) -> None:
        """Drop every failed launch whose reason was shown (the popup
        showing it was closed)."""
        with self._launch_guard:
            for key in [
                k
                for k, s in self._launches.items()
                if s["state"] != "running" and s.get("shown")
            ]:
                del self._launches[key]

    def launch_draft(self, key: str, label: str = "", progress=None) -> str:
        """Queue a whole draft as one campaign (and wait for it)."""
        project, name = self._draft(key)
        try:
            ids = project.launch_draft(name, label, progress=progress)
        except QueueError as exc:
            raise GuiError(str(exc)) from exc
        keys = self.workspace.added(project, ids)
        self._deep.clear()
        left = (
            "; studies added while it launched are still in the draft"
            if (project.drafts_dir / name).is_dir()
            else ""
        )
        return (
            f"launched {len(keys)} {project.words.task}(s) as "
            f"{ids[0].split('-', 1)[0]}: {', '.join(keys)}{left}"
            + _warned(project, ids)
        )

    def discard_draft(self, key: str) -> str:
        """Drop a draft (kept in out/removed/)."""
        project, name = self._draft(key)
        dest = project.discard_draft(name)
        return f"discarded the draft (kept in removed/{dest.name})"

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
            if kind in ("queue", "draft")
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


def _project_values(project, name: str) -> dict | None:
    """The project's values an optional file overrides, if the backend
    says (``project_values``)."""
    hook = getattr(project.backend, "project_values", None)
    if hook is None:
        return None
    try:
        values = hook(name)
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


def _value_sources(project, name: str) -> dict:
    """Which study last wrote each project value, if the backend keeps
    that record (``value_sources``)."""
    hook = getattr(project.backend, "value_sources", None)
    if hook is None:
        return {}
    try:
        return dict(hook(name) or {})
    except Exception:  # noqa: BLE001 -- the column is left out
        return {}


#: Marks a draft in a key: ``~<draft>/<id>``, at the start of a path
#: segment only (a project folder may hold a ``~``, as a Windows short
#: name does: ``CHIP~2``).
DRAFT_MARK = "~"


def _split_draft(key) -> tuple[str, str, str] | None:
    """``(project, draft, id)`` of a draft key, None for any other key.
    ``project`` is "" when the key names none (one project open)."""
    key = str(key)
    if key.startswith(DRAFT_MARK):
        head, rest = "", key[len(DRAFT_MARK) :]
    elif "/" + DRAFT_MARK in key:
        head, rest = key.split("/" + DRAFT_MARK, 1)
        if "/" in head:  # a draft is never deeper than <project>/~<draft>
            return None
    else:
        return None
    draft, _, local = rest.partition("/")
    return head, draft, local


class _Scoped:
    """A project seen through one draft: its ``queue`` is the draft's,
    so every editing method works on draft studies unchanged."""

    def __init__(self, project: Project, queue):
        self._project = project
        self.queue = queue
        self.backend = project.backend
        self.words = project.words
        self.out = project.out
        self.draft_key = ""

    def job(self, job_id: str):
        return load_job(self.queue.folder(job_id), self.words)

    def warnings(self, job_id: str) -> list[str]:
        # Its draft id: the id it gets at launch may differ in length;
        # the launch message warns again with the real one.
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
