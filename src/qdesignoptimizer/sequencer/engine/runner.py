"""Run the queue: the first study, then the next, until it is empty.

Per study:

1. **Check** its files (``Project.problems``, deep: the objects are
   rebuilt). A study that cannot run stops the runner and stays in the
   queue, to be fixed there.
2. **Start**: its folder moves to ``completed/<id>_<start>``,
   with ``inputs/`` keeping the files as queued.
3. **Fill in** what the folder leaves to the project
   (``resolve_inputs``): missing parameter targets and design
   variables are pulled now, after everything above it has run.
4. **Run** it, then its named **checks**.
5. **Commit** the tuned values to the project's store -- only when
   every check passed and ``save`` is on. A commit that fails fails the
   study.
6. **Record** ``result.json``; a failed study's folder gets ``_failed``.
   ``on_failure: abort`` (the default) stops the runner there -- also
   when the study was cut off by a crash and a new runner closes it.

Outputs never reach a later study by themselves: a value reaches the
next study only through the store, and only once it was committed.

The queue is re-read before every study, so an edit, a reorder or a new
study made in the GUI while the runner works takes effect for every
study not yet started. ``STOP`` ends it between studies.

The first study that may start runs; the others move up past one that
may not yet:

* one someone is editing (a live lease, ``lease.py``: open in Edit mode
  in the GUI, or being written from the command line) -- it runs once
  the edit is done;
* one whose ``needs`` (studies it uses the results of) have not all run
  successfully -- it runs once they have.

* one of a study sequence that ran in part on older code of its project
  -- it runs once someone says to continue it on the new code or restart
  it (``code_change.py``).

Before each study the runner reloads the project's code if it changed
(``code_files()``), and records the version it ran with.

With nothing that may start but something being edited, the runner
waits for the edit. With nothing that may start and nothing being
edited -- every study left needs one that is on hold, failed or gone --
it finishes and says which wait for what.

A runner runs a workspace: one project, or several sharing one order
(``workspace``). It holds the machine lock (one runner per user at a
time: two never solve at once), the lock of its state folder and the
lock of every project it runs.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

from . import state as st
from .backend import StepResult
from .code_change import (
    changed_files,
    code_hashes,
    code_version,
    hold_reason,
    sequence_of,
    version_of,
)
from .job import JobError, load_job, parse_completed
from .lease import describe, read_lease
from .project import Project
from .queue import INPUTS, QueueError, load_job_from
from .workspace import Workspace, user_home

log = logging.getLogger("qdesignoptimizer.sequencer.run")

#: Seconds between looks while every study it could start is being edited.
WAIT_S = 2.0


class RunRefused(RuntimeError):
    """A run cannot start (another runner holds the lock)."""


class Runner:
    """Runs a queue until it is empty, stopped or aborted.

    Args:
        target: A project, or a workspace of several.
    """

    def __init__(self, target: Project | Workspace):
        self.workspace = (
            target if isinstance(target, Workspace) else Workspace.of(target)
        )
        self.state = st.RunnerState()
        self._locks: list[st.RunnerLock] = []
        self._handler: logging.Handler | None = None
        self._passed: set[tuple[str, str]] = set()  # reasons said once
        self._waiting = ""  # what it waits for, as last said
        self._code: dict[str, dict[str, str]] = {}  # code loaded, per project
        self._broken: dict[str, dict[str, str]] = {}  # code that did not load
        # The study being handled (set per study by _loop).
        self.project = next(iter(self.workspace.projects.values()))

    @property
    def backend(self):
        return self.project.backend

    @property
    def queue(self):
        return self.project.queue

    def _take_locks(self) -> None:
        """Machine lock, then the state folder's, then each project's."""
        out = self.workspace.state_dir
        try:
            self._locks.append(st.acquire_lock(user_home()))
        except st.LockHeld as exc:
            active = st.read_json(user_home() / ACTIVE_FILE) or {}
            where = active.get("state_dir", "")
            raise RunRefused(
                f"a runner is already going on this machine ({exc}"
                + (f", {where}" if where else "")
                + ")"
            ) from exc
        dirs = [out] + [
            p.out
            for p in self.workspace.projects.values()
            if Path(p.out).resolve() != Path(out).resolve()
        ]
        for folder in dirs:
            try:
                self._locks.append(st.acquire_lock(folder))
            except st.LockHeld as exc:
                self._release_locks()
                raise RunRefused(
                    f"a runner is already going ({exc}) in {folder}"
                ) from exc

    def _release_locks(self) -> None:
        while self._locks:
            st.release_lock(self._locks.pop())

    def run(self) -> st.RunnerState:
        """Run every queued study in order; return the final state.

        Raises:
            RunRefused: If another runner is going.
        """
        out = self.workspace.state_dir
        self._take_locks()
        try:
            st.clear_stop(out)
            self.state = st.RunnerState(
                status=st.BUSY,
                pid=os.getpid(),
                started=st.now(),
            )
            self.state.save(out)
            st.write_json(
                user_home() / ACTIVE_FILE,
                {
                    "state_dir": str(out),
                    "pid": os.getpid(),
                    "started": self.state.started,
                    "projects": list(self.workspace.projects),
                },
            )
            self._open_log()
            log.info("runner started in %s", out)
            aborted: list[str] = []
            for project in self.workspace.projects.values():
                self.project = project
                # A runner killed between claim and start (power cut,
                # task manager) leaves .claimed-<id>; this one holds the
                # lock, so no claim can be anyone else's.
                for job_id in self.queue.recover_claimed():
                    log.warning(
                        "%s was left claimed by a runner that died; "
                        "it is back in the queue",
                        job_id,
                    )
                aborted += self._settle_cut_off()
            if aborted:
                # A study cut off by a crash failed as surely as one
                # that raised: under ``on_failure: abort`` the studies
                # below it do not run until the queue is started again.
                self.state.status = st.ABORTED
                self.state.message = (
                    ", ".join(aborted) + " was cut off by a crash "
                    "(on_failure: abort); start the runner again to "
                    "continue the queue"
                )
                log.warning(self.state.message)
            else:
                self._loop()
        except BaseException as exc:  # noqa: BLE001 -- recorded, re-raised
            # BaseException, so Ctrl+C mid-solve is recorded as what it
            # is rather than falling through to "finished".
            log.exception("runner stopped by %s", type(exc).__name__)
            self.state.status = st.ERROR
            self.state.message = (
                f"{type(exc).__name__}: {exc}"
                if str(exc)
                else f"stopped by {type(exc).__name__}"
            )
            raise
        finally:
            if self.state.status == st.BUSY:
                self.state.status = st.FINISHED
            self.state.current = ""
            self.state.finished = st.now()
            try:
                self.state.save(out)
            except OSError:
                # Still close the log and let go of the locks.
                log.exception("could not write the runner's final state")
            log.info("runner %s %s", self.state.status, self.state.message)
            self._close_log()
            st.clear_stop(out)
            self._release_locks()
        return self.state

    def _settle_cut_off(self) -> list[str]:
        """Close every study a dead runner left unfinished.

        Called holding the lock, so none of them is really running. A
        study that died while writing back (``committing``) is checked
        against the store: if every proposed value is there, the write
        completed and the study is recorded done; otherwise nothing was
        written (the write is one atomic file replace) and it failed.
        Everything else is recorded failed and renamed ``_failed``.

        Returns the failed ones whose ``on_failure`` is ``abort`` (a
        study whose job file cannot be read counts as ``abort``, the
        default): the runner then stops before the next study, as it
        would have had the study failed without a crash.
        """
        aborted: list[str] = []
        for path in self.queue.cut_off():
            record = st.result_of(path) or {"id": parse_completed(path.name)[0]}
            updates = record.get("updates") or {}
            if record.get("status") == st.COMMITTING and updates:
                try:
                    campaign = load_job_from(path, self.backend.words).campaign
                except Exception:  # noqa: BLE001 -- the project's store
                    campaign = ""
                # The store the study wrote: its campaign's.
                store = self.project.store_values(campaign)
                if all(store.get(k) == v for k, v in updates.items()):
                    record["status"] = st.DONE
                    record["committed"] = dict(updates)
                    self._note_recovered(record, path)
                    record["message"] = (
                        "recovered: the write-back had completed before "
                        "the runner stopped"
                    )
                    record["finished"] = record.get("finished") or st.now()
                    st.write_json(path / st.RESULT_FILE, record, 30.0)
                    log.warning("%s: %s", path.name, record["message"])
                    continue
            record["status"] = st.FAILED
            record["message"] = (
                "interrupted: the runner stopped while this "
                "study ran; nothing was written back"
            )
            record["finished"] = record.get("finished") or st.now()
            st.write_json(path / st.RESULT_FILE, record, 30.0)
            try:
                on_failure = load_job_from(path, self.backend.words).on_failure
            except Exception:  # noqa: BLE001 -- unreadable: the default
                on_failure = "abort"
            key = self.workspace.key(
                self.workspace.name_of(self.project),
                record.get("id") or path.name,
            )
            self.state.failed.append(key)
            if on_failure == "abort":
                aborted.append(key)
            self.queue.finish(path, ok=False)
            log.warning(
                "%s was cut off with its runner; marked failed "
                "(requeue it to run again)",
                path.name,
            )
        return aborted

    def _source(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self, job_id: str, task, sequence: dict, folder, settings=None, campaign=""
    ) -> dict:
        root = Path(self.backend.out_dir).parent
        try:
            where = Path(folder).relative_to(root).as_posix()
        except ValueError:
            where = str(folder)
        return {
            "study": job_id,
            "task": task,
            "sequence": (sequence or {}).get("id", ""),
            "sequence_name": (sequence or {}).get("name", ""),
            "folder": where,
            "settings": dict(settings or {}),
            **({"campaign": campaign} if campaign else {}),
            "written": st.now(),
        }

    def _note_commit(self, updates, before, job, folder, shown) -> None:
        """Record which study wrote the values back (where the backend
        keeps such a record). Never fails the study: the values are
        written already."""
        hook = getattr(self.backend, "note_commit", None)
        if hook is None or not updates:
            return
        previous = {k: before.get(k) for k in updates}
        try:
            hook(
                updates,
                previous,
                self._source(
                    job.id,
                    job.task,
                    job.sequence,
                    folder,
                    job.settings,
                    job.campaign,
                ),
            )
        except Exception:  # noqa: BLE001 -- logged; the write stands
            log.exception("%s: could not record the write-back", shown)

    def _note_recovered(self, record: dict, path: Path) -> None:
        hook = getattr(self.backend, "note_commit", None)
        if hook is None:
            return
        task = record.get(self.backend.words.task, "")
        try:
            ran = load_job_from(path / INPUTS, self.backend.words)
            sequence, settings, campaign = ran.sequence, ran.settings, ran.campaign
        except Exception:  # noqa: BLE001 -- the record goes without it
            sequence, settings, campaign = {}, {}, ""
        source = {
            **self._source(
                record.get("id", path.name), task, sequence, path, settings, campaign
            ),
            "recovered": True,
        }
        if "before" not in record:
            # No snapshot of the store before it: whether this study
            # changed anything cannot be told, so nothing is credited to
            # it (rather than overwriting the real writer).
            return
        before = record["before"]
        changed = {
            k: v
            for k, v in (record.get("committed") or {}).items()
            if before.get(k) != v
        }
        if not changed:
            return
        try:
            hook(changed, before, source)
        except Exception:  # noqa: BLE001
            log.exception("%s: could not record the write-back", path.name)

    # -- loop ------------------------------------------------------------

    def _next(self, keys: list[str]) -> tuple[str | None, bool]:
        """The first queued study that may start now (None if none), and
        whether to wait for one (something is being edited); with None,
        ``self.state.message`` says what holds them back."""
        edited, waiting, undecided, broken = [], [], {}, []
        versions: dict[str, str] = {}
        for key in keys:
            project, job_id = self.workspace.locate(key)
            name = self.workspace.name_of(project)
            if name in self._broken and code_hashes(project) == self._broken[name]:
                # Its code does not load; the other projects go on.
                if name not in broken:
                    broken.append(name)
                continue
            folder = project.queue.folder(job_id)
            lease = read_lease(folder)
            if lease is not None:
                edited.append(f"{key} (being edited by {describe(lease)})")
                self._say_once(key, f"{key} is being edited by {describe(lease)}")
                continue
            try:
                job = load_job(folder, project.words)
            except JobError:
                return key, False  # its own check stops it
            unmet = project.unmet_needs(job)
            if unmet:
                why = ", ".join(f"{n} ({w})" for n, w in unmet.items())
                waiting.append(f"{key} needs {why}")
                self._say_once(key, f"{key} waits for {why}")
                continue
            if name not in versions:
                versions[name] = code_version(project)
            held = hold_reason(project, job, versions[name])
            if held:
                sequence = self.workspace.key(name, sequence_of(job))
                undecided[sequence] = held
                self._say_once(key, f"{key} waits: {held}")
                continue
            return key, False
        self.state.message = "; ".join(
            (["waiting for code that loads: " + ", ".join(broken)] if broken else [])
            + (["waiting for edits: " + ", ".join(edited)] if edited else [])
            + (["waiting for what they need: " + "; ".join(waiting)] if waiting else [])
            + (
                [
                    "waiting for a decision (the code changed): "
                    + ", ".join(undecided)
                    + " -- continue or restart each"
                ]
                if undecided
                else []
            )
        )
        return None, bool(edited or broken)

    def _reload_if_changed(self, project) -> bool:
        """Reload a project's code before its next study if it changed
        since it was loaded. False if it does not load: nothing of that
        project may run until it does (``self.state.message`` says why);
        it is tried again once the files change again."""
        name = self.workspace.name_of(project)
        now = code_hashes(project)
        before = self._code.setdefault(name, now)
        if now == before:
            return True
        if self._broken.get(name) == now:
            return False
        hook = getattr(project.backend, "reload", None)
        try:
            if hook is not None:
                hook()
        except Exception as exc:  # noqa: BLE001 -- a typo must not end the runner
            self._broken[name] = now
            self.state.message = (
                f"{name}: the changed code does not load "
                f"({type(exc).__name__}: {exc}); waiting until it is fixed"
            )
            log.error(self.state.message)
            return False
        self._broken.pop(name, None)
        self._code[name] = now
        log.info(
            "%s: the code changed (%s); reloaded",
            name,
            ", ".join(changed_files(before, now)),
        )
        return True

    def _say_once(self, key: str, text: str) -> None:
        """Log why a study was passed over, once per reason."""
        if (key, text) not in self._passed:
            self._passed.add((key, text))
            log.info("%s: running the next study first", text)

    def _loop(self) -> None:
        out = self.workspace.state_dir
        self._passed = set()
        self._waiting = ""
        # The code each project was loaded with: a change reloads it.
        self._code = {
            name: code_hashes(project)
            for name, project in self.workspace.projects.items()
        }
        while True:
            if st.stop_requested(out):
                self.state.status = st.STOPPED
                self.state.message = "stopped on request between studies"
                return
            keys = self.workspace.ids()
            if not keys:
                return
            key, wait = self._next(keys)
            if key is None and not wait:
                # Every study left needs one that will not run here (on
                # hold, failed, gone): done, saying which wait for what.
                log.info(self.state.message)
                return
            if key is None:
                # Said once per wait, in runner.json, the log and the GUI.
                if self.state.message != self._waiting:
                    self._waiting = self.state.message
                    log.info(self.state.message)
                    self.state.save(out)
                time.sleep(WAIT_S)
                continue
            if self._waiting:
                self._waiting = ""
                self.state.message = ""
            self.project, job_id = self.workspace.locate(key)
            if not self._reload_if_changed(self.project):
                if self.state.message != self._waiting:
                    self._waiting = self.state.message
                    self.state.save(out)
                time.sleep(WAIT_S)
                continue
            shown = key
            # Claim first, check second: from here on the GUI cannot
            # edit the folder, so the study that runs is the one checked.
            try:
                claimed = self.queue.claim(job_id)
            except QueueError:
                continue  # removed or moved meanwhile; read the queue again
            except PermissionError as exc:
                # Windows refused the rename for ~5 s: a file of the study
                # (or a shell inside its folder) is held open. The study
                # stays queued and nothing below it runs, as for a failed
                # check -- the runner stops with a message that says why.
                self.state.status = st.ERROR
                self.state.message = (
                    f"{shown} could not be taken from the queue: a file or "
                    f"window has its folder open ({exc}); close it and "
                    "start again"
                )
                log.error(self.state.message)
                return
            if read_lease(claimed) is not None:
                # Edit was pressed (or an agent began writing) after it was
                # picked: it goes back, and runs once the edit is done.
                self.queue.release(claimed, job_id)
                continue
            try:
                problems = self.project.problems(job_id, deep=True, folder=claimed)
                if problems:
                    # Back in the queue, where it can be fixed; nothing
                    # below it runs either, since order is all the
                    # dependency the queue has.
                    self.queue.release(claimed, job_id)
                    self.state.status = st.ERROR
                    self.state.message = f"{shown} cannot run: " + ("; ".join(problems))
                    log.error(self.state.message)
                    return
                job = load_job(claimed, self.project.words)
                try:
                    folder = self.queue.start(claimed, job_id)
                except PermissionError as exc:
                    self.queue.release(claimed, job_id)
                    self.state.status = st.ERROR
                    self.state.message = (
                        f"{shown} could not be started: a file of it is held "
                        f"open ({exc}); close it and start again"
                    )
                    log.error(self.state.message)
                    return
            except BaseException:
                # Whatever went wrong between claim and start -- even a
                # Ctrl+C -- the study goes back to the queue rather than
                # staying hidden as .claimed-<id>.
                if claimed.exists():
                    self.queue.release(claimed, job_id)
                raise
            status, folder = self._run_one(job_id, job, folder)
            name = self.workspace.name_of(self.project)
            self.state.done.append(self.workspace.key(name, folder.name))
            if status == st.FAILED:
                self.state.failed.append(self.workspace.key(name, job_id))
            self.state.save(out)
            if status == st.FAILED and job.on_failure == "abort":
                self.state.status = st.ABORTED
                self.state.message = f"{shown} failed (on_failure: abort)"
                return

    def _run_one(self, job_id: str, job, folder: Path) -> tuple[str, Path]:
        out = self.workspace.state_dir
        # The study's key (<project>/<id> when several are open), so the
        # shared log says which project each line is about.
        shown = self.workspace.key(self.workspace.name_of(self.project), job_id)
        self.state.current = self.workspace.key(
            self.workspace.name_of(self.project), folder.name
        )
        self.state.save(out)
        version = version_of(self._code.get(self.workspace.name_of(self.project), {}))
        record = {
            "id": job_id,
            self.backend.words.task: job.task,
            "subject": job.subject,
            **({"campaign": job.campaign} if job.campaign else {}),
            **({"added_by": job.added_by} if job.added_by else {}),
            **({"code": version} if version else {}),
            "status": st.RUNNING,
            "started": st.now(),
            "finished": "",
            "pid": os.getpid(),
            "outputs": {},
            "updates": {},
            "committed": {},
            "checks": {},
            "message": "",
            "files": [],
        }
        st.write_json(folder / st.RESULT_FILE, record, 30.0)
        log.info("%s: %s %s", shown, self.backend.words.task, job.task)

        result: StepResult | None = None
        failures: list[str] = []
        try:
            self.backend.resolve_inputs(folder, job)
            self.queue.snapshot_resolved(folder)
            self.backend.before_step(job)
            result = self.backend.run(folder, job)
        except Exception as exc:  # noqa: BLE001 -- the study fails, not
            log.exception("%s raised", shown)  # the runner
            failures.append(f"{type(exc).__name__}: {exc}")

        if result is not None:
            record["outputs"] = dict(result.outputs)
            record["updates"] = dict(result.updates)
            record["files"] = list(result.files)
            for name in job.checks:
                try:
                    verdict = self.backend.check(name, job, result)
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"{name}: {type(exc).__name__}: {exc}")
                    continue
                record["checks"][name] = {
                    "ok": verdict.ok,
                    "message": verdict.message,
                }
                if not verdict.ok:
                    failures.append(f"{name}: {verdict.message or 'failed'}")
            if not failures and job.save and result.updates:
                # Say "committing" first: if the runner dies between the
                # write-back and the final record, the next runner checks
                # the store and records what really happened.
                try:
                    # For the record only: a store that cannot be read
                    # this instant must not fail the write-back itself.
                    before = self.project.store_values(job.campaign)
                except Exception:  # noqa: BLE001
                    before = None
                record["status"] = st.COMMITTING
                if before is not None:
                    # Kept with the status, so a runner settling a cut-off
                    # write-back can tell what it really changed.
                    record["before"] = {k: before.get(k) for k in result.updates}
                before = before or {}
                st.write_json(folder / st.RESULT_FILE, record, 30.0)
                try:
                    # A campaign's study writes its campaign's store.
                    changed = self.project.commit(dict(result.updates), job.campaign)
                    record["committed"] = dict(result.updates)
                    log.info("%s: wrote back %s", shown, sorted(result.updates))
                    # What the write actually changed (a backend that
                    # does not say: everything it was given).
                    self._note_commit(
                        dict(result.updates if changed is None else changed),
                        before,
                        job,
                        folder,
                        shown,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.exception("%s: write-back failed", shown)
                    failures.append(f"write-back failed: {type(exc).__name__}: {exc}")

        ok = not failures
        record["status"] = st.DONE if ok else st.FAILED
        record["message"] = (
            "; ".join(failures) if failures else (result.message if result else "")
        )
        record["finished"] = st.now()
        # Up to 30 s: a scanner holding result.json must not turn a
        # written-back study into a crashed runner.
        st.write_json(folder / st.RESULT_FILE, record, 30.0)
        final = self.queue.finish(folder, ok)
        log.log(
            logging.INFO if ok else logging.WARNING,
            "%s: %s%s",
            shown,
            record["status"],
            f" -- {record['message']}" if record["message"] else "",
        )
        return record["status"], final

    # -- log -------------------------------------------------------------

    def _open_log(self) -> None:
        # A file handler, not print: pyEPR replaces sys.stdout when the
        # first analysis starts, and every line printed after that is
        # lost (the TsTv2 study sequence of 2026-10-09 lost its stage headers).
        path = self.workspace.state_dir / st.LOG_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s: %(message)s",
                "%Y-%m-%d %H:%M:%S",
            )
        )
        root = logging.getLogger("qdesignoptimizer.sequencer")
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        self._handler = handler
        for project in self.workspace.projects.values():
            # A backend in a worker process logs there; send it here too.
            hook = getattr(project.backend, "log_to", None)
            if callable(hook):
                hook(path)

    def _close_log(self) -> None:
        if self._handler is not None:
            logging.getLogger("qdesignoptimizer.sequencer").removeHandler(self._handler)
            self._handler.close()
            self._handler = None


ACTIVE_FILE = "active.json"


def check_queue(target: Project | Workspace) -> dict[str, list[str]]:
    """Every queued study's problems, objects rebuilt (no solve)."""
    workspace = target if isinstance(target, Workspace) else Workspace.of(target)
    found = {}
    for key in workspace.ids():
        project, job_id = workspace.locate(key)
        found[key] = project.problems(job_id, deep=True)
    return found


def configure_console_logging() -> None:
    """Mirror the runner's log to the real stderr (CLI use)."""
    stream = sys.__stderr__
    if stream is None:  # pythonw: no console at all
        return
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
    root = logging.getLogger("qdesignoptimizer.sequencer")
    root.addHandler(handler)
    root.setLevel(logging.INFO)
