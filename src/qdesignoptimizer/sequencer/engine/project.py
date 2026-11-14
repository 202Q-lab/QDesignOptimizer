"""A project: its backend plus its folder queue.

The one place that turns a study key or a study sequence into queued
study folders, used alike by the CLI, the GUI and the tests.

**Drafts.** A study sequence can be put together before any of it is queued:
``out/drafts/<name>/`` is a queue of its own (the same folders, the
same ``order.json``, plus ``draft.json`` for its label) that the runner
never reads. Studies are added, edited, reordered and removed there;
``launch_draft`` then gives each a study sequence id and moves the whole
draft, in order, to the end of the queue.
"""

from __future__ import annotations

import secrets
import shutil
from contextlib import contextmanager
from pathlib import Path

from . import state as st
from .backend import Backend, SequenceSpec
from .job import (
    CUSTOM,
    JOB_FILE,
    Job,
    JobError,
    completed_name,
    dump_yaml,
    is_custom,
    load_job,
    slug_name,
)
from .loading import load_backend
from .queue import LAUNCH_FILE, STAGE, FolderQueue, QueueError, _rename

DRAFTS = "drafts"
#: ``check_templates``' entry for problems of the templates
#: themselves, which belong to no one study.
TEMPLATES_KEY = "(templates)"
DRAFT_FILE = "draft.json"
#: A staged study's ``job.yaml`` as it was in the draft (put back on
#: an undo).
ORIGINAL = ".job-in-draft.yaml"


#: The longest start time a completed folder's name carries
#: (``state.stamp``, plus the ``-NN`` that keeps two starts apart).
_LONGEST_STAMP = "2026-01-01T00-00-00-99"


class _NotLaunched(Exception):
    """A launch refused after staging (the studies go back)."""


class Project:
    """A backend and the queue under its ``out`` folder.

    Args:
        backend: The backend (from ``open_project`` or injected).
    """

    def __init__(self, backend: Backend):
        self.backend = backend
        self.words = backend.words
        self.queue = FolderQueue(backend.out_dir)

    @property
    def out(self) -> Path:
        """The project's ``out`` folder."""
        return self.queue.out

    def job(self, job_id: str) -> Job:
        """The parsed ``job.yaml`` of a queued study."""
        return load_job(self.queue.folder(job_id), self.words)

    def problems(
        self, job_id: str, deep: bool = False, folder: Path | None = None
    ) -> list[str]:
        """Everything that would stop a queued study from running.

        Args:
            job_id: The study.
            deep: Rebuild the backend's objects (slow; no solve).
            folder: Where it is, when not in the queue (a claimed one).

        Never raises. A check that crashes on a malformed file (a
        missing ``targets:`` key, ``iterations: "2"``) is itself a problem
        with that study: reported, so the runner puts the study back and
        the GUI keeps drawing, instead of losing the study or the page.
        """
        folder = Path(folder) if folder else self.queue.folder(job_id)
        try:
            return self._problems(folder, deep)
        except Exception as exc:  # noqa: BLE001 -- reported as a problem
            return [f"cannot be checked: {type(exc).__name__}: {exc}"]

    def _problems(self, folder: Path, deep: bool) -> list[str]:
        try:
            job = load_job(folder, self.words)
        except JobError as exc:
            return exc.problems
        catalog = self.backend.catalog()
        problems = []
        if catalog.task(job.task) is None and not is_custom(job.task):
            problems.append(f"unknown {self.words.task} {job.task!r}")
            return problems
        bad = sorted(set(job.settings) - set(self.backend.settings_keys()))
        if bad:
            problems.append(
                f"unknown setting(s) {bad}; allowed: "
                f"{sorted(self.backend.settings_keys())}"
            )
        known_checks = set(self.backend.check_names())
        unknown_checks = sorted(set(job.checks) - known_checks)
        if unknown_checks:
            # Caught here, not after a 40-minute solve that then fails
            # its checks and writes nothing back.
            problems.append(
                f"unknown check(s) {unknown_checks}; known: " f"{sorted(known_checks)}"
            )
        problems.extend(self.backend.check_folder(folder, job, deep) or [])
        return problems

    def warnings(self, job_id: str) -> list[str]:
        """What may go wrong with a study without stopping it.

        Today one thing: a path too long for the solver. A backend with a
        ``path_limit`` (characters) names in ``solver_files`` what its
        solver writes into the study's folder; a study whose completed
        folder (the longest form of its name, ``<id>_<start>``) would put
        one of them past the limit is flagged. Never raises.
        """
        limit = getattr(self.backend, "path_limit", None)
        files = tuple(getattr(self.backend, "solver_files", ()) or ())
        if not limit or not files:
            return []
        try:
            folder = self.queue.completed_dir.resolve() / completed_name(
                job_id, _LONGEST_STAMP
            )
            longest = max(files, key=len)
            length = len(str(folder / longest))
        except Exception:  # noqa: BLE001 -- a warning never breaks a study
            return []
        if length <= limit:
            return []
        return [
            f"its folder path will be {length} characters with "
            f"{longest}, past the solver's {limit}: "
            + (
                getattr(self.backend, "path_limit_note", "")
                or "the solver cannot write there"
            )
            + "; a shorter study sequence name or a project nearer the drive "
            "root avoids it"
        ]

    def set_hold(self, job_id: str, hold: bool) -> None:
        """Put a queued study on hold (the runner stops when it reaches
        it) or release it.

        Raises:
            QueueError: If it is not queued, or job.yaml is invalid.
        """
        folder = self.queue.folder(job_id)
        try:
            job = load_job(folder, self.words)
        except JobError as exc:
            raise QueueError("; ".join(exc.problems)) from exc
        job.hold = bool(hold)
        self.queue.write_file(job_id, JOB_FILE, dump_yaml(job.to_raw(self.words)))

    def add(
        self,
        task: str,
        settings: dict | None = None,
        checks: list[str] | None = None,
        save: bool | None = None,
        on_failure: str | None = None,
        notes: str = "",
        sequence: dict | None = None,
        position: int | None = None,
        queue: FolderQueue | None = None,
    ) -> str:
        """Generate one study and queue it. Returns its id.

        Args:
            queue: Where to put it (a draft's queue); the project's own
                queue by default.

        Raises:
            QueueError: For an unknown key, or files that cannot be made.
        """
        queue = queue or self.queue
        spec = self.backend.catalog().task(task)
        if spec is None:
            raise QueueError(f"no {self.words.task} {task!r}")
        try:
            files, generated_by = self.backend.generate(task)
        except Exception as exc:  # noqa: BLE001 -- reported to the user
            raise QueueError(
                f"generating {task} failed: {type(exc).__name__}: {exc}"
            ) from exc
        tag = sequence or {}
        job_id = queue.new_id(
            task,
            tag.get("id", ""),
            int(tag.get("entry", 0)),
            tag.get("name", ""),
        )
        job = Job(
            id=job_id,
            task=task,
            subject=spec.subject,
            settings={
                **self.backend.default_settings(task),
                **(settings or {}),
            },
            checks=list(checks if checks is not None else []),
            save=bool(save) if save is not None else False,
            on_failure=on_failure or "abort",
            notes=notes,
            sequence=dict(tag),
            created=st.now(),
            generated_by=generated_by,
        )
        queue.add(job, files, self.words, position)
        return job_id

    # -- drafts ----------------------------------------------------------

    @property
    def drafts_dir(self) -> Path:
        """``out/drafts/``."""
        return self.out / DRAFTS

    def draft_queue(self, name: str) -> FolderQueue:
        """One draft, as a queue of its own.

        Raises:
            QueueError: If there is no such draft.
        """
        folder = self.drafts_dir / name
        if not (folder / DRAFT_FILE).exists():
            raise QueueError(f"no draft {name!r}")
        queue = FolderQueue(self.out)
        queue.queue_dir = folder
        return queue

    def drafts(self) -> list[dict]:
        """Every draft: name, label, how many studies, where it came
        from."""
        if not self.drafts_dir.is_dir():
            return []
        out = []
        for folder in sorted(d for d in self.drafts_dir.iterdir() if d.is_dir()):
            meta = st.read_json(folder / DRAFT_FILE)
            if meta is None:
                continue
            out.append(
                {
                    **meta,
                    "name": folder.name,
                    "count": len(self.draft_queue(folder.name).ids()),
                }
            )
        return out

    def new_draft(self, label: str = "", from_spec: SequenceSpec | None = None) -> str:
        """Start a draft study sequence, empty or with a study sequence's studies.

        Returns:
            The draft's name.
        """
        name = f"draft-{st.stamp()}-{secrets.token_hex(2)}"
        folder = self.drafts_dir / name
        folder.mkdir(parents=True)
        st.write_json(
            folder / DRAFT_FILE,
            {
                "label": label or (from_spec.label if from_spec else ""),
                "from": from_spec.key if from_spec else "",
                "from_label": from_spec.label if from_spec else "",
                "why_ordered": from_spec.why_ordered if from_spec else "",
                "created": st.now(),
            },
        )
        if from_spec is not None:
            try:
                for entry in from_spec.entries:
                    self.draft_add(
                        name,
                        entry.task,
                        settings=entry.settings,
                        checks=list(entry.checks or []),
                        save=bool(entry.save),
                        on_failure=entry.on_failure,
                        notes=entry.notes,
                    )
            except QueueError:
                shutil.rmtree(folder, ignore_errors=True)
                raise
        return name

    def set_draft_label(self, name: str, label: str) -> None:
        """Rename a draft (the study sequence's name once launched)."""
        with self._draft_locked(name):
            path = self.drafts_dir / name / DRAFT_FILE
            meta = st.read_json(path) or {}
            meta["label"] = label
            st.write_json(path, meta)

    def draft_add(
        self,
        name: str,
        task: str,
        settings: dict | None = None,
        checks: list[str] | None = None,
        save: bool | None = True,
        on_failure: str | None = None,
        notes: str = "",
    ) -> str:
        """Generate one study into a draft. Returns its id there.

        A study added to a study sequence writes back by default (``save``):
        the studies after it build on what it tuned.
        """
        with self._draft_locked(name) as queue:
            return self.add(
                task,
                settings,
                checks,
                save,
                on_failure,
                notes,
                queue=queue,
            )

    def draft_blank(self, name: str, study: str = "") -> str:
        """Add a study made from scratch to a draft: every field of the
        backend's blank study (``blank_files``), no template behind it.
        Its task is ``custom.<study>``. Returns its id there.

        Raises:
            QueueError: If the backend cannot make a blank study.
        """
        maker = getattr(self.backend, "blank_files", None)
        if maker is None:
            raise QueueError(
                f"this backend cannot make a {self.words.task} from scratch"
            )
        label = slug_name(study) or "study"
        task = f"{CUSTOM}.{label}"
        try:
            files = maker()
        except Exception as exc:  # noqa: BLE001 -- reported
            raise QueueError(
                f"a blank {self.words.task} cannot be made: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        with self._draft_locked(name) as queue:
            return self._add_blank(queue, task, files)

    def _add_blank(self, queue: FolderQueue, task: str, files: dict) -> str:
        job = Job(
            id=queue.new_id(task),
            task=task,
            subject=CUSTOM,
            settings=dict(self.backend.default_settings(task)),
            checks=[],
            save=True,
            on_failure="abort",
            notes="",
            created=st.now(),
            generated_by="made from scratch in the GUI",
        )
        queue.add(job, files, self.words)
        return job.id

    # -- launching a draft ---------------------------------------------
    #
    # A launch moves the draft's studies into ONE hidden folder,
    # ``queue/.launch-<draft>/`` (the runner and the GUI list neither it
    # nor what is in it), keeping their ids, so whatever is in there
    # belongs to that launch and nothing can be stranded where no record
    # points. Its ``launch.json`` says how far the launch got; it is not
    # ``draft.json``, which the GUI rewrites whenever the name changes.
    #
    # 1. stage: every study is moved in, then checked THERE (an edit made
    #    after the move cannot reach it: the GUI finds it gone);
    # 2. name: each gets its study sequence id, recorded before it is written;
    # 3. publish: each is renamed into the queue.
    #
    # Any failure before 3 puts every study back. A failure during 3
    # keeps what was published and puts the rest back; launching again
    # continues the same study sequence. The launch lock (``out/drafts/``)
    # makes launches, discards and recovery one at a time; a launch cut
    # off by a crash is finished (if it was publishing) or undone by the
    # next launch, the next runner or the GUI when it opens.

    def _stage_dir(self, name: str) -> Path:
        return self.queue.queue_dir / f"{STAGE}{name}"

    def _launch_lock(self, wait: float):
        self.drafts_dir.mkdir(parents=True, exist_ok=True)
        try:
            return st.acquire_lock(self.drafts_dir, wait=wait)
        except st.LockHeld as exc:
            raise QueueError(
                "a study sequence is being launched; try again when it is done"
            ) from exc

    @contextmanager
    def _draft_locked(self, name: str):
        """Hold the launch lock over a change to a draft, and yield its
        queue: a launch cannot then take the draft away mid-change (it
        would rename the draft's studies out from under the change, and
        a study written after the draft was removed would sit in a
        folder nothing lists).

        Raises:
            QueueError: If the draft is gone, or a launch holds the lock
                for more than a few seconds.
        """
        self.draft_queue(name)
        lock = self._launch_lock(wait=5.0)
        try:
            yield self.draft_queue(name)  # not launched while we waited
        finally:
            st.release_lock(lock)

    def discard_draft(self, name: str) -> Path:
        """Drop a draft; it is moved to ``out/removed/``, not deleted.

        Raises:
            QueueError: If it is gone, or being launched.
        """
        self.draft_queue(name)
        lock = self._launch_lock(wait=5.0)
        try:
            self.draft_queue(name)  # not launched while we waited
            self.queue.removed_dir.mkdir(parents=True, exist_ok=True)
            dest = self.queue.removed_dir / f"{st.stamp()}_{name}"
            _rename(self.drafts_dir / name, dest)
            return dest
        finally:
            st.release_lock(lock)

    def launch_draft(
        self,
        name: str,
        label: str = "",
        deep: bool = True,
        progress=None,
    ) -> list[str]:
        """Queue a whole draft as one study sequence, in its order.

        Every study gets a study sequence id (``s<NN>-<sequence>-<study>``) and goes to
        the end of the queue. The draft is removed, unless studies were
        added to it while it launched: those stay in it, and launching
        it again continues the same study sequence.

        Args:
            deep: Rebuild every study's objects first (the check the
                runner makes before it runs one): a study that cannot
                run would stop the whole queue there.
            progress: Called ``progress(checked, total, study)`` as the
                studies are checked (the slow part), and once more as
                ``(total, total, "")`` when they all are.

        Raises:
            QueueError: If the draft is gone or empty, a study in it
                cannot run (nothing is queued then), another launch is
                going, or the launch was cut short (the message says
                what was queued; the rest is back in the draft).
        """
        self.draft_queue(name)
        lock = self._launch_lock(wait=5.0)
        try:
            self._recover()
            if not (self.drafts_dir / name / DRAFT_FILE).exists():
                raise QueueError("the draft is gone: launched or discarded meanwhile")
            return self._launch(name, label, deep, progress)
        finally:
            st.release_lock(lock)

    def _launch(self, name: str, label: str, deep: bool, progress=None) -> list[str]:
        draft = self.draft_queue(name)
        ids = draft.ids()
        if not ids:
            raise QueueError("the study sequence has no studies yet")
        stage = self._stage_dir(name)
        stage.mkdir(parents=True)
        state: dict = {"draft": name, "phase": "staging", "entries": []}
        st.write_json(stage / LAUNCH_FILE, state)
        try:
            staged = []
            for old_id in ids:
                _rename(draft.folder(old_id), stage / old_id)
                staged.append(old_id)
            bad = {}
            for n, i in enumerate(staged):
                if progress is not None:
                    progress(n, len(staged), i)
                found = self.problems(i, deep=deep, folder=stage / i)
                if found:
                    bad[i] = found
            if progress is not None:
                progress(len(staged), len(staged), "")
            if bad:
                raise _NotLaunched(
                    "not launched; fix first: "
                    + "; ".join(f"{i}: {p[0]}" for i, p in bad.items())
                )
            meta = st.read_json(self.drafts_dir / name / DRAFT_FILE) or {}
            earlier = meta.get("launched") or {}
            # A study sequence launched in parts keeps the name it was first
            # launched with: its ids carry it, and should match. A draft
            # made from a sequence and not renamed is recorded under that
            # sequence's key, as the CLI's ``sequence`` records it (short
            # ids, one name in design_variables.meta.json); a renamed one
            # is a sequence of its own, under its new name.
            label = label or meta.get("label") or ""
            if meta.get("from") and label in ("", meta.get("from_label", "")):
                label = meta["from"]
            label = earlier.get("label") or label or name
            sequence_id = earlier.get("sequence") or self._next_sequence_id()
            first = int(earlier.get("count", 0))
            state.update(sequence=sequence_id, first=first, label=label)
            for n, old_id in enumerate(staged, first + 1):
                folder = stage / old_id
                text = (folder / JOB_FILE).read_text(encoding="utf-8")
                (folder / ORIGINAL).write_text(text, encoding="utf-8")
                job = load_job(folder, self.words)
                job.id = self.queue.new_id(job.task, sequence_id, n, label)
                job.sequence = {"id": sequence_id, "name": label, "entry": n}
                # Recorded before it is written, so the id is taken
                # (``FolderQueue.taken``) and a crash knows where it goes.
                state["entries"].append({"old_id": old_id, "new_id": job.id})
                st.write_json(stage / LAUNCH_FILE, state)
                (folder / JOB_FILE).write_text(
                    dump_yaml(job.to_raw(self.words)), encoding="utf-8"
                )
            state["phase"] = "publishing"
            st.write_json(stage / LAUNCH_FILE, state)
        except BaseException as exc:
            self._unstage(name)
            if isinstance(exc, _NotLaunched):
                raise QueueError(str(exc)) from None
            if isinstance(exc, Exception):
                raise QueueError(
                    f"not launched ({type(exc).__name__}: {exc}); every "
                    f"study is back in the draft"
                ) from exc
            raise
        try:
            published = self._publish(stage, state["entries"])
        except OSError as exc:
            published = [
                e["new_id"]
                for e in state["entries"]
                if self.queue.folder(e["new_id"]).is_dir()
            ]
            self._append_order(published)
            self._unstage(name)
            self._note_launched(name, sequence_id, first + len(published), label)
            raise QueueError(
                f"launched {len(published)} of {len(state['entries'])} as "
                f"{sequence_id} ({type(exc).__name__}: {exc}); the rest "
                f"are back in the draft -- launch it again to finish the "
                f"same study sequence"
            ) from exc
        self._append_order(published)
        self._finish(name, sequence_id, first + len(published), label)
        return published

    def _publish(self, stage: Path, entries: list[dict]) -> list[str]:
        """Rename each staged study to its study sequence id, in order.

        Their order is written first: a folder ``order.json`` does not
        list goes to the end of the queue by name, so a runner picking its
        next study while the renames happen (each may wait seconds on
        Windows) would otherwise take a study sequence's studies alphabetically.
        A listed id whose folder does not exist (yet) is skipped.
        """
        self._append_order([e["new_id"] for e in entries])
        published = []
        for entry in entries:
            source = stage / entry["old_id"]
            final = self.queue.folder(entry["new_id"])
            if source.is_dir():
                _rename(source, final)
            if final.is_dir():
                (final / ORIGINAL).unlink(missing_ok=True)
                published.append(entry["new_id"])
        return published

    def _finish(self, name: str, sequence_id: str, count: int, label: str = "") -> None:
        """Remove the stage folder and the draft -- unless studies were
        added to the draft while it launched: then it stays, and a
        launch of it continues this study sequence."""
        shutil.rmtree(self._stage_dir(name), ignore_errors=True)
        folder = self.drafts_dir / name
        if not folder.is_dir():
            return
        if self.draft_queue(name).ids():
            self._note_launched(name, sequence_id, count, label)
        else:
            # Moved, not deleted: whatever is still in it is kept.
            self.queue.removed_dir.mkdir(parents=True, exist_ok=True)
            _rename(
                folder,
                self.queue.removed_dir / f"{st.stamp()}_{name}_launched-{sequence_id}",
            )

    def _note_launched(self, name: str, sequence_id: str, count: int, label: str = ""):
        path = self.drafts_dir / name / DRAFT_FILE
        if path.exists():  # never recreate a draft that is gone
            meta = st.read_json(path) or {}
            meta["launched"] = {
                "sequence": sequence_id,
                "count": count,
                "label": label
                or (meta.get("launched") or {}).get("label")
                or meta.get("label", ""),
            }
            st.write_json(path, meta)

    def _next_sequence_id(self) -> str:
        """``s<NN>`` past every study sequence queued, completed, being
        launched or reserved by a draft launched in part -- a launch that
        published nothing still owns its number (the draft keeps it)."""
        highest = int(self.queue.next_sequence_id()[1:]) - 1
        try:
            drafts = self.drafts()
        except (OSError, ValueError):  # a damaged draft.json
            drafts = []
        for draft in drafts:
            number = str((draft.get("launched") or {}).get("sequence", ""))
            if number[:1] == "s" and number[1:].isdigit():
                highest = max(highest, int(number[1:]))
        return f"s{highest + 1:02d}"

    def _append_order(self, ids: list[str]) -> None:
        if ids:
            with st.ORDER_LOCK:
                order = [i for i in self.queue.ids() if i not in ids]
                st.write_json(self.queue.queue_dir / "order.json", order + list(ids))

    def _unstage(self, name: str) -> None:
        """Put every study still in a launch's stage folder back into its
        draft, as it was (the draft is made again if it is gone)."""
        stage = self._stage_dir(name)
        if not stage.is_dir():
            return
        draft_dir = self.drafts_dir / name
        if not (draft_dir / DRAFT_FILE).exists():
            draft_dir.mkdir(parents=True, exist_ok=True)
            st.write_json(
                draft_dir / DRAFT_FILE,
                {"label": f"{name} (recovered)", "created": st.now()},
            )
        for folder in sorted(p for p in stage.iterdir() if p.is_dir()):
            original = folder / ORIGINAL
            if original.exists():
                (folder / JOB_FILE).write_text(
                    original.read_text(encoding="utf-8"), encoding="utf-8"
                )
                original.unlink()
            _rename(folder, draft_dir / folder.name)
        shutil.rmtree(stage, ignore_errors=True)

    def _recover(self) -> list[str]:
        """Finish or undo every launch a crash cut short (the launch lock
        held). One line per launch it touched."""
        done: list[str] = []
        if not self.queue.queue_dir.is_dir():
            return done
        for stage in sorted(self.queue.queue_dir.iterdir()):
            if not (stage.is_dir() and stage.name.startswith(STAGE)):
                continue
            name = stage.name[len(STAGE) :]
            state = st.read_json(stage / LAUNCH_FILE) or {}
            if state.get("phase") == "publishing":
                published = self._publish(stage, state.get("entries") or [])
                self._append_order(published)
                self._unstage(name)  # anything it did not get to
                self._finish(
                    name,
                    state.get("sequence", ""),
                    int(state.get("first", 0)) + len(published),
                    state.get("label", ""),
                )
                done.append(f"{name}: finished its launch ({len(published)} queued)")
            else:
                self._unstage(name)
                done.append(
                    f"{name}: its launch was undone; every study is back "
                    f"in the draft"
                )
        return done

    def recover_launches(self) -> list[str]:
        """Finish or undo launches a crash cut short; one line per launch
        it touched. A launch going on now is left alone (the lock is
        taken without waiting)."""
        if not self.queue.queue_dir.is_dir() or not any(
            p.name.startswith(STAGE) for p in self.queue.queue_dir.iterdir()
        ):
            return []
        try:
            lock = self._launch_lock(wait=0)
        except QueueError:
            return []
        try:
            return self._recover()
        finally:
            st.release_lock(lock)

    def check_templates(self, deep: bool = True) -> dict[str, list[str]]:
        """Fill in every queueable study and check it, without queueing:
        ``{key: problems}`` ([] for one that would run).

        Each is generated into a scratch folder exactly as ``add`` would
        and checked the way the runner checks a study before it runs it
        (``deep``: objects and the design built). A template that drifted
        -- a name the design no longer has, a target that does not build
        -- shows up here, not when someone queues it.
        """
        import tempfile

        found = {}
        catalog = self.backend.catalog()
        # What the backend could not make a study of at all (a template
        # it refused, an opt target without a name or with one used
        # twice): those are in no catalog entry, so they are listed here.
        refused = list(getattr(self.backend, "catalog_problems", []) or [])
        if refused:
            found[TEMPLATES_KEY] = refused
        with tempfile.TemporaryDirectory(prefix="qdo-check-") as tmp:
            scratch = FolderQueue(Path(tmp))
            for spec in catalog.tasks:
                try:
                    job_id = self.add(spec.key, queue=scratch)
                except QueueError as exc:
                    found[spec.key] = [str(exc)]
                    continue
                found[spec.key] = self.problems(
                    job_id, deep=deep, folder=scratch.folder(job_id)
                )
                scratch.remove(job_id)
        return found

    def add_sequence(self, spec: SequenceSpec) -> list[str]:
        """Queue every study of a study sequence under a fresh study sequence id.

        All studies are made before any is queued, so one failing
        half-way leaves the queue as it was.
        """
        catalog = self.backend.catalog()
        missing = [e.task for e in spec.entries if catalog.task(e.task) is None]
        if missing:
            raise QueueError(f"{spec.key}: unknown {self.words.task}(s) " f"{missing}")
        generated = []
        for entry in spec.entries:
            try:
                generated.append(self.backend.generate(entry.task))
            except Exception as exc:  # noqa: BLE001
                raise QueueError(
                    f"{spec.key}: generating {entry.task} failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
        ids: list[str] = []
        # Numbered past drafts launched in part (a draft keeps its
        # number) and past a launch under way (its staged ids count as
        # taken). Not under the launch lock: a GUI launch holds it for
        # its whole deep check, minutes for a real design.
        try:
            sequence_id = self._next_sequence_id()
            for n, (entry, (files, by)) in enumerate(zip(spec.entries, generated), 1):
                spec_t = catalog.task(entry.task)
                assert spec_t is not None  # unknown tasks were refused above
                job = Job(
                    id=self.queue.new_id(entry.task, sequence_id, n, spec.key),
                    task=entry.task,
                    subject=spec_t.subject,
                    settings={
                        **self.backend.default_settings(entry.task),
                        **entry.settings,
                    },
                    checks=list(entry.checks or []),
                    save=bool(entry.save),
                    on_failure=entry.on_failure or "abort",
                    notes=entry.notes,
                    sequence={"id": sequence_id, "name": spec.key, "entry": n},
                    created=st.now(),
                    generated_by=by,
                )
                self.queue.add(job, files, self.words)
                ids.append(job.id)
        except BaseException:
            # All or nothing: take back what was queued (to removed/).
            for job_id in ids:
                try:
                    self.queue.remove(job_id)
                except QueueError:
                    pass  # claimed by a runner meanwhile; it runs
            raise
        return ids


def open_project(ref: str | Path) -> Project:
    """The project a ``sequencer.py`` (or ``file.py:factory``) defines."""
    ref = str(ref)
    if ref.endswith(".py") or ":" not in ref.replace(":\\", "").replace(":/", ""):
        target, factory = ref, "make_backend"
    else:
        target, factory = ref.rsplit(":", 1)
    if not target.endswith(".py"):
        # A dotted module (``qdesignoptimizer.sequencer.backends.fake``).
        return Project(load_backend(f"{target}:{factory}", Path.cwd()))
    path = Path(target).resolve()
    return Project(load_backend(f"{path.name}:{factory}", path.parent))
