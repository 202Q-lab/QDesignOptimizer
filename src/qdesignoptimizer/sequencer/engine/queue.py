"""The folder queue: ``<out>/queue/`` and ``<out>/completed/``.

::

    <out>/queue/order.json                 run order (ids)
    <out>/queue/<id>/job.yaml              one study: its queue entry, plus
                                           its backend files
    <out>/completed/<id>_<start>[_failed]/
        the same files as resolved at start, inputs/ (as queued),
        resolved/ (as filled in from the project, before the solve),
        result.json, and whatever the solve wrote

An id is ``p07-tune_ATS1-ats1_eigenmode`` (measurement plan p07, called
tune_ATS1; the study) or ``x-a7f3-ats1_eigenmode`` (queued on its
own). A completed folder is the id, then its start time, so sorting
``completed/`` by name in Explorer groups each measurement plan's studies, with
the ones run on their own after them. A study folder moves to
``completed/`` when it starts, so the solver writes its outputs straight
into the folder they stay in; a failed one is renamed ``..._failed``.

Every change to the queue is a folder operation plus one rewrite of
``order.json``, which is the only file the GUI and the runner both
write; it is written atomically.
"""

from __future__ import annotations

import re
import secrets
import shutil
import time
from pathlib import Path

import yaml

from . import state as st
from .job import (
    JOB_FILE,
    OLD_JOB_FILE,
    Job,
    JobError,
    completed_name,
    dump_yaml,
    is_completed_name,
    job_file,
    parse_completed,
    parse_job,
)

ORDER_FILE = "order.json"
INPUTS = "inputs"
RESOLVED = "resolved"
#: A draft being launched: ``<out>/queue/.launch-<draft>/`` holds its
#: studies (not listed) and ``launch.json``, how far the launch got.
STAGE = ".launch-"
LAUNCH_FILE = "launch.json"
CLAIMED = ".claimed-"
_PLAN = re.compile(r"^p(\d+)-")


class QueueError(RuntimeError):
    """A queue operation was refused, with the reason."""


def _rename(source: Path, dest: Path, tries: int = 50) -> None:
    """Rename a folder, retrying while Windows refuses.

    A folder whose files were just written is often held open for a
    moment by the virus scanner or the indexer, and the rename then
    fails with "Access is denied" (seen on the first test of this
    queue). Up to ~5 s of retries; the last refusal is raised.
    """
    for attempt in range(tries):
        try:
            source.rename(dest)
            return
        except PermissionError:
            if attempt == tries - 1:
                raise
            time.sleep(0.1)


def slug(text: str) -> str:
    """A folder-name-safe, lower-case form of ``text``."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_").lower()


def plan_tag(name: str, length: int = 32) -> str:
    """A measurement plan's name as it goes into its studies' ids: folder-safe,
    case kept, at most ``length`` characters."""
    tag = re.sub(r"[^0-9a-zA-Z]+", "_", str(name or "")).strip("_")
    return tag[:length].rstrip("_")


class FolderQueue:
    """The queue and completed folders of one project.

    Args:
        out_dir: The project's ``out`` folder.
    """

    def __init__(self, out_dir: Path):
        self.out = Path(out_dir)
        self.queue_dir = self.out / "queue"
        self.completed_dir = self.out / "completed"
        self.removed_dir = self.out / "removed"

    # -- reading ---------------------------------------------------------

    def ids(self) -> list[str]:
        """Queued ids in run order.

        ``order.json`` sets the order. A folder it does not list (made
        by hand) goes at the end, by name; a listed id whose folder is
        gone is dropped.
        """
        if not self.queue_dir.is_dir():
            return []
        folders = sorted(
            p.name
            for p in self.queue_dir.iterdir()
            if p.is_dir() and not p.name.startswith(".") and job_file(p).exists()
        )
        listed = st.read_json(self.queue_dir / ORDER_FILE) or []
        order = [i for i in listed if i in folders]
        return order + [f for f in folders if f not in order]

    def folder(self, job_id: str) -> Path:
        """The queued folder of an id."""
        return self.queue_dir / job_id

    def completed(self, limit: int | None = None) -> list[Path]:
        """Completed study folders, newest first (by start time, not by
        name: the name sorts by measurement plan)."""
        if not self.completed_dir.is_dir():
            return []
        runs = sorted(
            (
                p
                for p in self.completed_dir.iterdir()
                if p.is_dir() and is_completed_name(p.name)
            ),
            key=lambda p: (parse_completed(p.name)[1], p.name),
            reverse=True,
        )
        return runs[:limit] if limit else runs

    def taken(self) -> set[str]:
        """Every id ever used here: queued, being started, completed or
        removed. Ids are never reused, so a measurement plan queued again after
        its studies were removed gets a new number."""
        ids = set(self.ids())
        if self.queue_dir.is_dir():
            ids |= {
                p.name[len(CLAIMED) :]
                for p in self.queue_dir.iterdir()
                if p.name.startswith(CLAIMED)
            }
        for path in self.completed():
            # result.json names the id exactly; the folder name cannot
            # tell a "_failed" suffix from a task that ends in "failed".
            recorded = (st.result_of(path) or {}).get("id")
            ids.add(recorded or parse_completed(path.name)[0])
        if self.queue_dir.is_dir():
            # The ids a launch under way has given out, before it
            # publishes them: a measurement plan started meanwhile must not get
            # the same number.
            for stage in self.queue_dir.iterdir():
                if stage.name.startswith(STAGE) and stage.is_dir():
                    state = st.read_json(stage / LAUNCH_FILE) or {}
                    ids |= {
                        e["new_id"]
                        for e in state.get("entries") or []
                        if e.get("new_id")
                    }
        if self.removed_dir.is_dir():
            for path in self.removed_dir.iterdir():
                parts = path.name.split("_", 1)
                if len(parts) == 2:
                    ids.add(parts[1])
        return ids

    def next_plan_id(self) -> str:
        """``p<NN>``, one past the highest in queue/ and completed/."""
        highest = 0
        for job_id in self.taken():
            match = _PLAN.match(job_id)
            if match:
                highest = max(highest, int(match.group(1)))
        return f"p{highest + 1:02d}"

    def new_id(
        self,
        task: str,
        plan: str = "",
        entry: int = 0,
        plan_name: str = "",
    ) -> str:
        """A fresh id for a study of ``task``: in a measurement plan
        ``p<NN>-<plan>-<study>``, else ``x-<hex>-<study>``. The
        same study twice in one measurement plan gets ``_2``."""
        del entry  # the run order is order.json's, not the id's
        taken = self.taken()
        if plan:
            tag = plan_tag(plan_name)
            base = f"{plan}-{tag}-{slug(task)}" if tag else f"{plan}-{slug(task)}"
            job_id, n = base, 2
            while job_id in taken:
                job_id, n = f"{base}_{n}", n + 1
            return job_id
        while True:
            job_id = f"x-{secrets.token_hex(2)}-{slug(task)}"
            if job_id not in taken:
                return job_id

    # -- editing ---------------------------------------------------------

    def save_order(self, order: list[str]) -> None:
        """Write the run order.

        Raises:
            QueueError: If ``order`` is not exactly the queued ids.
        """
        with st.ORDER_LOCK:
            current = self.ids()
            if sorted(order) != sorted(current) or len(set(order)) != len(order):
                raise QueueError("the queue changed meanwhile; reloaded")
            st.write_json(self.queue_dir / ORDER_FILE, list(order))

    def add(
        self,
        job: Job,
        files: dict[str, str],
        words,
        position: int | None = None,
    ) -> Path:
        """Create a queued study folder and put it in the order.

        Built under a dot-name and renamed into place, so the runner
        never sees half a folder.
        """
        target = self.folder(job.id)
        if target.exists():
            raise QueueError(f"{job.id} is already queued")
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        building = self.queue_dir / f".new-{job.id}"
        if building.exists():
            shutil.rmtree(building)
        building.mkdir()
        (building / JOB_FILE).write_text(dump_yaml(job.to_raw(words)), encoding="utf-8")
        for name, text in files.items():
            (building / name).write_text(text, encoding="utf-8")
        with st.ORDER_LOCK:
            _rename(building, target)
            order = [i for i in self.ids() if i != job.id]
            order.insert(len(order) if position is None else position, job.id)
            st.write_json(self.queue_dir / ORDER_FILE, order)
        return target

    def remove(self, job_id: str) -> Path:
        """Take a study out of the queue. Its folder is moved to
        ``<out>/removed/`` rather than deleted, so an edit is never lost
        by a misclick."""
        source = self.folder(job_id)
        if not source.is_dir():
            raise QueueError(f"{job_id} is not queued")
        self.removed_dir.mkdir(parents=True, exist_ok=True)
        dest = self.removed_dir / f"{st.stamp()}_{job_id}"
        with st.ORDER_LOCK:
            _rename(source, dest)
            st.write_json(
                self.queue_dir / ORDER_FILE,
                [i for i in self.ids() if i != job_id],
            )
        return dest

    def write_file(self, job_id: str, name: str, text: str) -> None:
        """Replace one file of a queued study (the GUI's editor)."""
        folder = self.folder(job_id)
        if not folder.is_dir():
            raise QueueError(f"{job_id} is not queued")
        if "/" in name or "\\" in name or name.startswith("."):
            raise QueueError(f"{name!r} is not a file name")
        path = folder / name
        tmp = folder / f".{name}.tmp"
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)

    def delete_file(self, job_id: str, name: str) -> None:
        """Delete an optional file, so it is pulled from the project."""
        path = self.folder(job_id) / name
        if path.exists():
            path.unlink()

    # -- running ---------------------------------------------------------

    def claim(self, job_id: str) -> Path:
        """Take the study out of the GUI's reach before checking it.

        The folder is renamed to ``.claimed-<id>``: ``ids()`` no longer
        lists it and every edit of it is refused, so what the runner
        checks is what it runs. ``release`` puts it back.

        The runner never writes ``order.json``: ``ids()`` drops a listed
        id whose folder is gone, so the GUI stays its only writer and a
        drag made while a study starts is never overwritten.
        """
        source = self.folder(job_id)
        if not source.is_dir():
            raise QueueError(f"{job_id} is not queued")
        claimed = self.queue_dir / f"{CLAIMED}{job_id}"
        try:
            _rename(source, claimed)
        except FileNotFoundError as exc:
            # Removed in the GUI between the check above and the rename.
            raise QueueError(f"{job_id} is no longer queued") from exc
        return claimed

    def release(self, claimed: Path, job_id: str) -> None:
        """Return a claimed study to the queue, where it was."""
        _rename(Path(claimed), self.folder(job_id))

    def cut_off(self) -> list[Path]:
        """Completed studies a runner left unfinished: ``result.json``
        says running or committing, or was never written although the
        study had started (``inputs/`` exists)."""
        found = []
        for path in self.completed():
            if path.name.endswith("_failed"):
                continue
            record = st.result_of(path)
            status = record.get("status")
            if status in (st.RUNNING, st.COMMITTING) or (
                not record and (path / INPUTS).is_dir()
            ):
                found.append(path)
        return found

    def recover_claimed(self) -> list[str]:
        """Release every claimed study (call only while holding the
        runner lock: then no claim belongs to a live runner)."""
        if not self.queue_dir.is_dir():
            return []
        recovered = []
        for path in sorted(self.queue_dir.iterdir()):
            if path.is_dir() and path.name.startswith(CLAIMED):
                job_id = path.name[len(CLAIMED) :]
                if not self.folder(job_id).exists():
                    self.release(path, job_id)
                    recovered.append(job_id)
        return recovered

    def start(self, claimed: Path, job_id: str) -> Path:
        """Move a claimed study to ``completed/`` under its start time.

        A copy of the files exactly as queued goes to ``inputs/`` first,
        so ``requeue`` can restore them however the run fills them in.
        If the move fails, that copy is removed again, so a later start
        does not carry files deleted meanwhile.
        """
        source = Path(claimed)
        inputs = source / INPUTS
        if inputs.exists():
            shutil.rmtree(inputs)
        inputs.mkdir()
        for path in source.iterdir():
            if path.is_file() and not path.name.startswith("."):
                shutil.copy2(path, inputs / path.name)
        self.completed_dir.mkdir(parents=True, exist_ok=True)
        stamp = st.stamp()
        dest = self.completed_dir / completed_name(job_id, stamp)
        n = 2
        while dest.exists():
            dest = self.completed_dir / completed_name(job_id, f"{stamp}-{n}")
            n += 1
        try:
            _rename(source, dest)
        except OSError:
            shutil.rmtree(inputs, ignore_errors=True)
            raise
        return dest

    def finish(self, folder: Path, ok: bool) -> Path:
        """Rename a failed study ``..._failed``; returns where it is.

        If Windows refuses the rename (a file still held open), the
        name stays and ``result.json`` says ``failed`` all the same.
        """
        folder = Path(folder)
        if ok or folder.name.endswith("_failed"):
            return folder
        dest = folder.with_name(folder.name + "_failed")
        try:
            _rename(folder, dest, tries=30)
            return dest
        except OSError:
            return folder

    def snapshot_resolved(self, folder: Path) -> None:
        """Copy a started study's files, as filled in from the project,
        to ``resolved/`` -- what an exact requeue runs again. Taken
        before the solve, which may write into the same files."""
        folder = Path(folder)
        resolved = folder / RESOLVED
        if resolved.exists():
            shutil.rmtree(resolved)
        resolved.mkdir()
        for path in folder.iterdir():
            if (
                path.is_file()
                and not path.name.startswith(".")
                and path.name != st.RESULT_FILE
            ):
                shutil.copy2(path, resolved / path.name)

    def requeue(self, completed: Path, words, exact: bool = False) -> str:
        """Put a completed study back at the end of the queue as a fresh
        copy. Returns the new id.

        Args:
            completed: The completed study's folder.
            words: The backend's words.
            exact: False: its files as it was queued, so whatever it
                left to the project (design variables, parameter
                targets) is pulled again when the copy starts. True:
                its files as it ran, values from the project included,
                so the copy repeats that run.
        """
        completed = Path(completed)
        inputs = completed / INPUTS
        if not job_file(inputs).exists():
            raise QueueError(f"{completed.name} has no inputs/ to requeue")
        source = inputs
        if exact:
            source = completed / RESOLVED
            if not source.is_dir():
                raise QueueError(
                    f"{completed.name} has no resolved/ (it did not get "
                    f"as far as filling in its values, or ran before "
                    f"they were kept); requeue it fresh instead"
                )
        try:
            old = load_job_from(inputs, words)
        except (JobError, OSError, yaml.YAMLError) as exc:
            raise QueueError(
                f"{completed.name}/inputs cannot be requeued: {exc}"
            ) from exc
        # Strip an "_rN" only from an id that came from a requeue: a task
        # slug can end in "_r2" of its own.
        base = re.sub(r"_r\d+$", "", old.id) if old.raw.get("requeued_from") else old.id
        taken = self.taken()
        n = 2
        new_id = f"{base}_r{n}"
        while new_id in taken:
            n += 1
            new_id = f"{base}_r{n}"
        old.id = new_id
        old.raw["requeued_from"] = completed.name
        old.raw["requeued_as"] = "exact copy" if exact else "fresh"
        files = {
            p.name: p.read_text(encoding="utf-8")
            for p in source.iterdir()
            if p.is_file() and p.name not in (JOB_FILE, OLD_JOB_FILE)
        }
        self.add(old, files, words)
        return new_id


def load_job_from(folder: Path, words) -> Job:
    """``load_job`` without the folder-name check (for inputs/)."""
    raw = yaml.safe_load(job_file(folder).read_text("utf-8"))
    return parse_job(raw, words)
