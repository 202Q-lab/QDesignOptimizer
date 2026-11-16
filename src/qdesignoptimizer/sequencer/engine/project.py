"""A project: its backend plus its folder queue.

The one place that turns a study key or a study sequence into queued
study folders, used alike by the CLI, the GUI and the tests.

**On hold.** ``out/hold/`` is a queue of its own that the runner never
reads: a new study is made there (or parked there from the queue),
edited, and moved into the queue when it is ready.

**Study sequences** are templates: files in the project's
``study_sequences/`` (``save_sequence`` writes one) or built in the
adapter's code. ``add_sequence`` queues every study of one.
"""

from __future__ import annotations

import os
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from . import campaign as cp
from . import state as st
from .backend import Backend, SequenceSpec
from .job import (
    CUSTOM,
    Job,
    JobError,
    completed_name,
    is_custom,
    load_job,
    parse_completed,
    slug_name,
)
from .loading import load_backend
from .queue import HOLD, FolderQueue, QueueError, _rename
from .sequence import (
    SequenceError,
    check_sequence_key,
    load_sequence_dir,
    save_sequence,
)

#: ``check_templates``' entry for problems of the templates
#: themselves, which belong to no one study.
TEMPLATES_KEY = "(templates)"


#: The longest start time a completed folder's name carries
#: (``state.stamp``, plus the ``-NN`` that keeps two starts apart).
_LONGEST_STAMP = "2026-01-01T00-00-00-99"


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

    @property
    def root(self) -> Path:
        """The project's folder (``out``'s parent)."""
        return Path(self.backend.out_dir).parent

    # -- campaigns -------------------------------------------------------

    def campaigns(self) -> list[cp.CampaignInfo]:
        """Every campaign, parents before their sub-campaigns."""
        return [
            replace(c, own_store=self._own_store(c.key))
            for c in cp.campaigns(self.root)
        ]

    def campaign_store(self, key: str) -> Path | None:
        """The store studies of campaign ``key`` use ("" for the
        project's); None when the backend has no campaign stores."""
        store_for = getattr(self.backend, "store_for", None)
        return Path(store_for(key)) if callable(store_for) else None

    def _own_store(self, key: str) -> bool:
        store = self.campaign_store(key)
        return (
            store is not None
            and store.parent.resolve() == cp.folder_of(self.root, key).resolve()
        )

    def store_values(self, campaign: str = "") -> dict:
        """The store a study of ``campaign`` uses, as it stands."""
        if campaign:
            return self.backend.store_values(campaign=campaign)
        return self.backend.store_values()

    def commit(self, updates: dict, campaign: str = ""):
        """Write ``updates`` to the store of ``campaign``."""
        if campaign:
            return self.backend.commit(updates, campaign=campaign)
        return self.backend.commit(updates)

    def new_campaign(
        self, key: str, question: str = "", own_store: bool = True
    ) -> Path:
        """Make campaign ``key`` (see ``campaign.create``).

        Raises:
            QueueError: For a bad key, a missing parent or one that exists.
        """
        try:
            parent = cp.check_key(key).rpartition("/")[0]
            return cp.create(
                self.root,
                key,
                question,
                self.campaign_store(parent) if own_store else None,
            )
        except cp.CampaignError as exc:
            raise QueueError(str(exc)) from exc

    def promote(self, key: str, names: list[str] | None = None) -> dict[str, Any]:
        """Copy a campaign's values one level up: into the store its
        parent campaign uses (the project's, for a top-level one).

        Without ``names``: the values the campaign itself changed since
        it was made (or last promoted), as its ``.promoted.json`` says --
        not every value that differs, which would also undo what the
        parent changed meanwhile. ``names`` promotes exactly those. Only
        values that differ from the parent's are written, and the write
        is recorded like a study's (``promoted_from`` in the record).
        Returns what changed.

        Raises:
            QueueError: If the campaign is not there, has no store of its
                own, or a name is not in its store.
        """
        problems = cp.problems_of(self.root, key)
        if problems:
            raise QueueError(problems[0])
        key = cp.check_key(key)
        parent = key.rpartition("/")[0]
        if not self._own_store(key):
            raise QueueError(
                f"campaign {key!r} has no store of its own: its studies "
                "already write where promoting would"
            )
        values = self.store_values(key)
        base_path = cp.folder_of(self.root, key) / cp.BASE_FILE
        base = st.read_json(base_path)
        if names is not None:
            missing = sorted(set(names) - set(values))
            if missing:
                raise QueueError(f"not in campaign {key!r}'s store: {missing}")
            values = {k: values[k] for k in names}
        elif isinstance(base, dict):
            values = {k: v for k, v in values.items() if base.get(k, object()) != v}
        before = self.store_values(parent)
        updates = {k: v for k, v in values.items() if before.get(k) != v}

        def settle() -> None:
            # Promoted (or already equal above): the new starting point.
            # Written only once the parent holds them, so a failed commit
            # is promoted again next time. A campaign from before
            # .promoted.json gets one from here on.
            if isinstance(base, dict):
                if values:
                    st.write_json(base_path, {**base, **values})
            elif names is None:
                st.write_json(base_path, self.store_values(key))

        if not updates:
            settle()
            return {}
        changed = self.commit(updates, parent)
        changed = updates if changed is None else changed
        settle()
        hook = getattr(self.backend, "note_commit", None)
        if hook is not None and changed:
            source = {
                "study": "",
                "task": "",
                "promoted_from": key,
                "folder": cp.folder_of(self.root, key)
                .relative_to(self.root)
                .as_posix(),
                "written": st.now(),
            }
            if parent:
                source["campaign"] = parent
            hook(dict(changed), {k: before.get(k) for k in changed}, source)
        return dict(changed)

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

    def need_state(self, need: str) -> str:
        """Where a needed study stands: ``done`` (a successful run, of it
        or of a requeued copy ``<id>_rN``), ``queued``, ``hold``,
        ``failed`` (its runs all failed, none queued again) or
        ``missing``."""
        runs = self.runs_of(need)
        if any(st.result_of(p).get("status") == st.DONE for p in runs):
            return "done"
        return self.pending_state(need) or ("failed" if runs else "missing")

    def runs_of(self, job_id: str) -> list[Path]:
        """The completed folders of a study and of its requeued copies
        (``<id>_rN``), oldest start first."""
        same = re.compile(re.escape(job_id) + r"(_r\d+)?$")
        return sorted(
            (
                p
                for p in self.queue.completed()
                if same.match(parse_completed(p.name)[0])
            ),
            key=_started,
        )

    def pending_state(self, job_id: str) -> str:
        """``queued`` (or running) or ``hold`` when the study or a
        requeued copy of it is still to run; "" when none is."""
        same = re.compile(re.escape(job_id) + r"(_r\d+)?$")
        if any(same.match(i) for i in self.queue.ids()) or any(
            same.match(i) for i in self.queue.claimed_ids()
        ):
            return "queued"
        if any(same.match(i) for i in self.hold.ids()):
            return "hold"
        return ""

    def unmet_needs(self, job: Job) -> dict[str, str]:
        """The studies ``job`` needs that have not run successfully yet,
        with where each stands (``need_state``)."""
        out = {}
        for need in job.needs:
            where = self.need_state(need)
            if where != "done":
                out[need] = where
        return out

    def _problems(self, folder: Path, deep: bool) -> list[str]:
        try:
            job = load_job(folder, self.words)
        except JobError as exc:
            return exc.problems
        for need, where in self.unmet_needs(job).items():
            # Queued or on hold: it can still run first. Not so these.
            if where == "missing":
                return [f"needs {need}, which is not queued, on hold or completed"]
            if where == "failed":
                return [f"needs {need}, whose runs failed; requeue it"]
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
        if job.campaign:
            problems.extend(cp.problems_of(self.root, job.campaign))
            if not problems and self.campaign_store(job.campaign) is None:
                problems.append(
                    f"this project's backend keeps no campaign stores; "
                    f"campaign {job.campaign!r} cannot be used"
                )
            if problems:
                return problems
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

    # -- on hold ---------------------------------------------------------

    @property
    def hold(self) -> FolderQueue:
        """The studies on hold (``out/hold/``), as a queue of their own
        that the runner never reads. A study is made there, or parked
        there from the queue, and moved to the queue when it is ready."""
        queue = FolderQueue(self.out)
        queue.queue_dir = queue.hold_dir
        return queue

    def park(self, job_id: str) -> None:
        """Move a queued study on hold: out of the run order, files and
        all, until ``unpark``.

        Raises:
            QueueError: If it is not queued (or the runner just took it).
        """
        self._move(self.queue, self.hold, job_id, None)

    def unpark(self, job_id: str, position: int | None = None) -> None:
        """Move a study on hold into the queue, at ``position`` (the end
        when None).

        Raises:
            QueueError: If it is not on hold.
        """
        self._move(self.hold, self.queue, job_id, position)

    def _move(self, source: FolderQueue, dest: FolderQueue, job_id, position):
        folder = source.folder(job_id)
        if not folder.is_dir():
            where = "on hold" if source.queue_dir == self.out / HOLD else "queued"
            raise QueueError(f"{job_id} is not {where}")
        dest.queue_dir.mkdir(parents=True, exist_ok=True)
        with st.order_guard():
            try:
                _rename(folder, dest.folder(job_id))
            except FileNotFoundError as exc:
                # Claimed by the runner between the check and the rename.
                raise QueueError(
                    f"{job_id} was just taken by the runner; nothing moved"
                ) from exc
            except OSError as exc:
                # A folder of that name there already, or a file held
                # open for the whole retry.
                raise QueueError(
                    f"{job_id} could not be moved ({type(exc).__name__}: "
                    f"{exc}); nothing moved"
                ) from exc
            order = [i for i in dest.ids() if i != job_id]
            order.insert(len(order) if position is None else position, job_id)
            dest.write_order(order)
            source.write_order([i for i in source.ids() if i != job_id])

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
        campaign: str = "",
        hold: bool = False,
        needs: list[str] | None = None,
        added_by: str = "",
        why: str = "",
    ) -> str:
        """Generate one study and queue it. Returns its id.

        Args:
            needs: Ids of studies it needs to have run first.
            added_by: Who queued it ("" for the GUI); ``why``: the reason.
            queue: Where to put it; the project's own queue (or its hold
                folder, with ``hold``) by default.
            campaign: The campaign it belongs to, "" for none.
            hold: Make it on hold (``out/hold/``) rather than in the queue.

        Raises:
            QueueError: For an unknown key, or files that cannot be made.
        """
        queue = queue or (self.hold if hold else self.queue)
        campaign = self._campaign_key(campaign)
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
            campaign=campaign,
            created=st.now(),
            generated_by=generated_by,
            needs=list(needs or []),
            added_by=added_by,
            why=why,
        )
        queue.add(job, files, self.words, position)
        return job_id

    def add_blank(self, study: str = "", hold: bool = True, campaign: str = "") -> str:
        """Make a study from scratch: every field of the backend's blank
        study (``blank_files``), no template behind it. Its task is
        ``custom.<study>``. On hold by default, to be filled in before it
        is queued. Returns its id.

        Raises:
            QueueError: If the backend cannot make a blank study, or the
                campaign does not exist.
        """
        maker = getattr(self.backend, "blank_files", None)
        if maker is None:
            raise QueueError(
                f"this backend cannot make a {self.words.task} from scratch"
            )
        campaign = self._campaign_key(campaign)
        task = f"{CUSTOM}.{slug_name(study) or 'study'}"
        try:
            files = maker()
        except Exception as exc:  # noqa: BLE001 -- reported
            raise QueueError(
                f"a blank {self.words.task} cannot be made: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        queue = self.hold if hold else self.queue
        job = Job(
            id=queue.new_id(task),
            task=task,
            subject=CUSTOM,
            settings=dict(self.backend.default_settings(task)),
            checks=[],
            save=True,
            on_failure="abort",
            notes="",
            campaign=campaign,
            created=st.now(),
            generated_by="made from scratch in the GUI",
        )
        queue.add(job, files, self.words)
        return job.id

    def _next_sequence_id(self) -> str:
        """``s<NN>`` past every study sequence queued, on hold or
        completed."""
        return self.queue.next_sequence_id()

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

    def add_sequence(
        self, spec: SequenceSpec, added_by: str = "", why: str = ""
    ) -> list[str]:
        """Queue every study of a study sequence under a fresh study sequence id.

        ``added_by`` and ``why`` go on each of its studies (``add``).

        All studies are made before any is queued, so one failing
        half-way leaves the queue as it was.
        """
        campaign = self._campaign_key(spec.campaign)
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
                    # What it uses: by default every earlier study of the
                    # sequence that writes back.
                    needs=[
                        ids[k - 1]
                        for k in (
                            entry.needs
                            if entry.needs is not None
                            else [
                                k
                                for k, e in enumerate(spec.entries[: n - 1], 1)
                                if e.save
                            ]
                        )
                    ],
                    notes=entry.notes,
                    added_by=added_by,
                    why=why,
                    sequence={"id": sequence_id, "name": spec.key, "entry": n},
                    campaign=campaign,
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

    # -- study sequence templates ----------------------------------------

    @property
    def sequence_dir(self) -> Path | None:
        """Where study sequence files live (None: the backend keeps none)."""
        folder = getattr(self.backend, "sequence_dir", None)
        return Path(folder) if folder else None

    def sequence_raw(self, key: str) -> dict:
        """A study sequence as its file's mapping, to edit: the file as
        written, or for one built in code, its studies spelled out.

        Raises:
            QueueError: If there is no such study sequence.
        """
        # The file first (a template saved a moment ago is there before
        # every catalog -- a worker's may be a poll old -- has it), unless
        # the adapter's code defines one by that name: the catalog uses
        # that one, so the editor must too.
        folder = self.sequence_dir
        path = folder / f"{key}.yaml" if folder is not None else None
        if path is not None and path.is_file() and not self.sequence_from_code(key):
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                return raw
        spec = self.backend.catalog().sequence(key)
        if spec is None:
            raise QueueError(f"no {self.words.sequence} {key!r}")
        if spec.source and Path(spec.source).is_file():
            raw = yaml.safe_load(Path(spec.source).read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                return raw
        studies = []
        for entry in spec.entries:
            item: dict[str, Any] = {self.words.task: entry.task}
            if entry.settings:
                item["settings"] = dict(entry.settings)
            if entry.checks is not None:
                item["checks"] = list(entry.checks)
            if entry.save is not None:
                item["save"] = entry.save
            if entry.on_failure:
                item["on_failure"] = entry.on_failure
            if entry.notes:
                item["notes"] = entry.notes
            studies.append(item)
        raw = {"name": spec.label, "why_ordered": spec.why_ordered, "studies": studies}
        if spec.campaign:
            raw["campaign"] = spec.campaign
        return raw

    def sequence_from_code(self, key: str) -> bool:
        """Whether study sequence ``key`` is built in the adapter's code
        (the catalog takes it over a file of the same name)."""
        spec = self.backend.catalog().sequence(key)
        return spec is not None and not spec.source

    def rename_sequence(self, key: str, new: str) -> Path:
        """Rename study sequence template ``key``'s file to ``new``.

        Raises:
            QueueError: For a bad name, one in use, a template built in
                code, or no such file.
        """
        folder = self.sequence_dir
        if folder is None:
            raise QueueError(f"this project keeps no {self.words.sequence} files")
        try:
            check_sequence_key(new)
        except SequenceError as exc:
            raise QueueError("; ".join(exc.problems)) from exc
        path, target = folder / f"{key}.yaml", folder / f"{new}.yaml"
        if not path.is_file():
            raise QueueError(f"{key!r} has no file in {folder.name}/ to rename")
        code = self.backend.catalog().sequence(new)
        if target.exists() or (code is not None and not code.source):
            raise QueueError(f"{new!r} is taken; pick another file name")
        try:
            os.rename(path, target)  # refuses one made meanwhile
        except OSError as exc:
            raise QueueError(f"{path.name} could not be renamed: {exc}") from exc
        return target

    def save_sequence(self, key: str, raw: dict, overwrite: bool = False) -> Path:
        """Write a study sequence template (``study_sequences/<key>.yaml``).

        Its studies must be ones the project can queue and its campaign
        (if any) must exist.

        Raises:
            QueueError: Saying what is wrong; nothing is written then.
        """
        folder = self.sequence_dir
        if folder is None:
            raise QueueError(f"this project keeps no {self.words.sequence} files")
        catalog = self.backend.catalog()
        tasks = [
            (
                str(item.get(self.words.task) or item.get("task") or "")
                if isinstance(item, dict)
                else str(item)
            )
            for item in raw.get("studies") or []
        ]
        unknown = sorted({t for t in tasks if t and catalog.task(t) is None})
        if unknown:
            raise QueueError(f"unknown {self.words.task}(s): {', '.join(unknown)}")
        if raw.get("campaign"):
            self._campaign_key(str(raw["campaign"]))
        try:
            check_sequence_key(key)
        except SequenceError as exc:
            raise QueueError("; ".join(exc.problems)) from exc
        # Every template has its own name: the GUI and the queue show it.
        # The files are read afresh (a worker's catalog may lag a save
        # made a moment ago); the code's templates come from the catalog.
        label = str(raw.get("name") or key)
        on_disk, _ = load_sequence_dir(folder, self.words)
        others = [s for s in catalog.sequences if not s.source] + on_disk
        clash = next(
            (s for s in others if s.key != key and s.label == label),
            None,
        )
        if clash is not None:
            raise QueueError(
                f"the name {label!r} is taken by {clash.key!r}; give this "
                f"{self.words.sequence} a name of its own"
            )
        code = catalog.sequence(key)
        if code is not None and not code.source:
            raise QueueError(
                f"{key!r} is built in the adapter's code; save it under " "another name"
            )
        try:
            return save_sequence(folder, key, raw, self.words, overwrite)
        except SequenceError as exc:
            raise QueueError("; ".join(exc.problems)) from exc
        except OSError as exc:
            raise QueueError(f"{key}.yaml could not be written: {exc}") from exc

    def _campaign_key(self, key: str) -> str:
        """``key`` normalised, refused unless the campaign exists."""
        if not key:
            return ""
        problems = cp.problems_of(self.root, key)
        if problems:
            raise QueueError(problems[0])
        return cp.check_key(key)


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


def _started(folder: Path) -> tuple:
    """Sort key of a completed folder: its start time (from its name),
    then when its result was last written -- a study and its requeued
    copy can start in the same second."""
    try:
        written = (folder / st.RESULT_FILE).stat().st_mtime_ns
    except OSError:
        written = 0
    return parse_completed(folder.name)[1], written
