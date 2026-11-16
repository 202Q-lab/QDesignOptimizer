"""When a project's code changes while a study sequence is half run.

A project names its code (``code_files()`` on the backend: for QDO the
chip's ``design.py``, ``names.py``, its ``sequencer.py`` ...). Every run
records the version of that code it ran with (``code`` in result.json, a
hash of the files' contents).

* A study on its own just runs on the code there is when it starts: the
  runner reloads the project's code before the next study whenever it
  changed.
* A study sequence half run on older code is held back -- the studies
  after it move up -- until someone decides, for that sequence:

  - **continue**: the rest runs on the new code;
  - **restart**: the rest goes on hold, and the whole sequence is queued
    again as a new one (``s08`` for ``s07``; same name, same files, each
    study noting ``restarted_from: s07``), to run on the new code from
    its start. The runs of ``s07`` stay, as the record of the old code.

The decisions are kept in ``out/code_decisions.json``: ``{s07: {choice,
code, at}}``. A "continue" holds for the code it was given for: another
change asks again.
"""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from typing import Any

from . import state as st
from .job import JOB_FILE, Job, load_job, parse_completed
from .queue import INPUTS, QueueError, load_job_from

DECISIONS_FILE = "code_decisions.json"
CONTINUE, RESTART = "continue", "restart"


#: How long the list of a project's code files is kept: asking a backend
#: in a worker process is a round trip, and the list hardly changes.
FILES_KEPT_S = 30.0


def code_files(project) -> list[Path]:
    """The files of a project's code (none when its backend names none)."""
    kept = project.__dict__.get("_code_files")
    if kept is not None and time.monotonic() - kept[0] < FILES_KEPT_S:
        return list(kept[1])
    hook = getattr(project.backend, "code_files", None)
    try:
        found = sorted({Path(p) for p in hook() or []}) if hook is not None else []
    except Exception:  # noqa: BLE001 -- no code named, nothing compared
        found = []
    project.__dict__["_code_files"] = (time.monotonic(), found)
    return list(found)


def code_hashes(project) -> dict[str, str]:
    """``{file: short hash of its contents}`` of a project's code. A file
    is read again only when its size or time changed, or it changed in
    the last 2 s."""
    cache: dict[str, tuple] = project.__dict__.setdefault("_code_cache", {})
    out = {}
    for path in code_files(project):
        try:
            stat = path.stat()
        except OSError:
            out[str(path)] = "missing"
            continue
        stamp = (stat.st_mtime_ns, stat.st_size)
        known = cache.get(str(path))
        # A file's time moves in steps (~16 ms on Windows): one written
        # just now may change again under the same time and size.
        recent = time.time() - stat.st_mtime < 2.0
        if known is None or known[0] != stamp or recent:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
            known = cache[str(path)] = (stamp, digest)
        out[str(path)] = known[1]
    return out


def code_version(project) -> str:
    """One short hash of all of a project's code ("" when it names none)."""
    hashes = code_hashes(project)
    if not hashes:
        return ""
    text = "\n".join(f"{Path(k).name} {v}" for k, v in sorted(hashes.items()))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def changed_files(before: dict[str, str], after: dict[str, str]) -> list[str]:
    """The names of the files that differ between two ``code_hashes``."""
    return sorted(
        Path(k).name for k in set(before) | set(after) if before.get(k) != after.get(k)
    )


def sequence_of(job: Job) -> str:
    """The study sequence a study belongs to (``s07``), "" for none."""
    return str((job.sequence or {}).get("id") or "")


def _runs(project, sequence: str) -> list[Path]:
    """The completed folders of a sequence's studies, oldest first."""
    prefix = f"{sequence}-"
    return sorted(
        (
            p
            for p in project.queue.completed()
            if parse_completed(p.name)[0].startswith(prefix)
        ),
        key=_started,
    )


def _started(folder: Path) -> tuple:
    try:
        written = (folder / st.RESULT_FILE).stat().st_mtime_ns
    except OSError:
        written = 0
    return parse_completed(folder.name)[1], written


def ran_with(project, sequence: str) -> set[str]:
    """The code versions a sequence's studies have run with so far."""
    return {
        str(st.result_of(p).get("code"))
        for p in _runs(project, sequence)
        if st.result_of(p).get("code")
    }


def decisions(project) -> dict[str, dict]:
    """``{sequence: {choice, code, at}}``."""
    found = st.read_json(project.out / DECISIONS_FILE)
    return found if isinstance(found, dict) else {}


def hold_reason(project, job: Job, current: str | None = None) -> str:
    """Why a study waits for a decision on its sequence ("" if it may
    run): its sequence ran in part on other code than there is now."""
    sequence = sequence_of(job)
    if not sequence:
        return ""
    current = code_version(project) if current is None else current
    if not current:
        return ""
    before = ran_with(project, sequence)
    if not before or before <= {current}:
        return ""
    decided = decisions(project).get(sequence) or {}
    if decided.get("choice") == CONTINUE and decided.get("code") == current:
        return ""
    return (
        f"the code changed since {sequence} began: continue it on the new "
        f"code or restart it"
    )


def waiting(project) -> dict[str, list[str]]:
    """``{sequence: [queued ids]}`` of the sequences held for a decision."""
    current = code_version(project)
    out: dict[str, list[str]] = {}
    seen: dict[str, str] = {}
    for job_id in project.queue.ids():
        try:
            job = load_job(project.queue.folder(job_id), project.words)
        except Exception:  # noqa: BLE001 -- its own check reports it
            continue
        sequence = sequence_of(job)
        if not sequence:
            continue
        if sequence not in seen:
            seen[sequence] = hold_reason(project, job, current)
        if seen[sequence]:
            out.setdefault(sequence, []).append(job_id)
    return out


def decide(project, sequence: str, choice: str) -> list[str]:
    """Continue or restart a sequence held for a code change. Returns the
    ids of the new sequence's studies for ``restart`` ([] for continue).

    Raises:
        QueueError: For an unknown choice, or a sequence that has
            nothing left to run.
    """
    if choice not in (CONTINUE, RESTART):
        raise QueueError(f"choose {CONTINUE} or {RESTART}, not {choice!r}")
    current = code_version(project)
    record: dict[str, Any] = {"choice": choice, "code": current, "at": st.now()}
    made: list[str] = []
    if choice == RESTART:
        made = restart(project, sequence)
        record["as"] = sequence_of(project.job(made[0])) if made else ""
    with st.order_guard():
        found = decisions(project)
        found[sequence] = record
        st.write_json(project.out / DECISIONS_FILE, found)
    return made


def _files(folder: Path) -> dict[str, str]:
    return {
        p.name: p.read_text(encoding="utf-8")
        for p in folder.iterdir()
        if p.is_file() and p.name != JOB_FILE and not p.name.startswith(".")
    }


def _members(project, sequence: str) -> tuple[dict[int, tuple], list[str], dict]:
    """A sequence's studies by entry: ``{n: (job, files)}`` -- the last
    run for those that ran, else the queued or held one -- the ids still
    queued, and ``{any id it went by: n}``."""
    prefix = f"{sequence}-"
    entries: dict[int, tuple] = {}
    queued: list[str] = []
    known: dict[str, int] = {}

    def take(job: Job, files: dict[str, str]) -> None:
        n = int((job.sequence or {}).get("entry") or 0)
        if n:
            entries[n] = (job, files)
            known[job.id] = n
            known[re.sub(r"_r\d+$", "", job.id)] = n

    for path in _runs(project, sequence):
        try:
            take(load_job_from(path / INPUTS, project.words), _files(path / INPUTS))
        except Exception:  # noqa: BLE001 -- not part of the copy
            continue
    for source, still in ((project.hold, False), (project.queue, True)):
        for job_id in source.ids():
            if not job_id.startswith(prefix):
                continue
            folder = source.folder(job_id)
            try:
                take(load_job(folder, project.words), _files(folder))
            except Exception:  # noqa: BLE001
                continue
            if still:
                queued.append(job_id)
    return entries, queued, known


def restart(project, sequence: str) -> list[str]:
    """Queue a sequence again from its start as a new sequence, and put
    what is left of the old one on hold. Returns the new ids.

    Raises:
        QueueError: If the sequence has no studies, or one cannot be
            queued (then nothing is).
    """
    entries, queued, known = _members(project, sequence)
    if not entries:
        raise QueueError(f"{sequence} has no studies to restart")
    new_sequence = project.queue.next_sequence_id()
    taken = project.queue.taken()
    ids: dict[int, str] = {}
    for n, (job, _) in sorted(entries.items()):
        base = project.queue.new_id(
            job.task, new_sequence, n, (job.sequence or {}).get("name", "")
        )
        new, k = base, 2
        while new in taken or new in ids.values():
            new, k = f"{base}_{k}", k + 1
        ids[n] = new
    made: list[str] = []
    try:
        for n, (job, files) in sorted(entries.items()):
            job.id = ids[n]
            job.sequence = {**(job.sequence or {}), "id": new_sequence}
            job.needs = [ids.get(known.get(i, 0), i) for i in job.needs]
            job.created = st.now()
            for key in ("requeued_from", "requeued_as"):
                job.raw.pop(key, None)
            job.raw["restarted_from"] = sequence
            project.queue.add(job, files, project.words)
            made.append(job.id)
    except BaseException:
        for job_id in made:
            try:
                project.queue.remove(job_id)
            except QueueError:
                pass
        raise
    for job_id in queued:
        try:
            project.park(job_id)
        except QueueError:
            pass  # taken by a runner meanwhile: it runs
    return made
