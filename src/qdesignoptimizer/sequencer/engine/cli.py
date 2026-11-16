"""Command line.

    python -m qdesignoptimizer.sequencer guide     (how an agent works the queue)
    python -m qdesignoptimizer.sequencer <sequencer.py> list
    python -m qdesignoptimizer.sequencer <sequencer.py> add <study> [--settings JSON] [--campaign KEY]
        [--needs ID ...] (studies that must have run successfully first)
        [--by NAME] [--why TEXT] (who queues it and why; the GUI shows both)
    python -m qdesignoptimizer.sequencer <sequencer.py> sequence <file.yaml or key> [--campaign KEY]
        [--by NAME] [--why TEXT]
    python -m qdesignoptimizer.sequencer <sequencer.py> decide [<sNN> continue|restart]
        (a study sequence half run when the project's code changed waits
        for this; without arguments, lists those that wait)
    python -m qdesignoptimizer.sequencer <sequencer.py> wait <id> [<id> ...] [--timeout S]
        (until each has run; prints its result and folder. Exit 0 all done,
        1 one failed or is on hold, 2 one is unknown, 3 timed out)
    python -m qdesignoptimizer.sequencer <sequencer.py> campaign list
    python -m qdesignoptimizer.sequencer <sequencer.py> campaign new <key> [--question TEXT] [--shared]
    python -m qdesignoptimizer.sequencer <sequencer.py> campaign promote <key> [<name> ...]
    python -m qdesignoptimizer.sequencer <sequencer.py> check
    python -m qdesignoptimizer.sequencer <sequencer.py> check-templates
    python -m qdesignoptimizer.sequencer <sequencer.py> run --yes
    python -m qdesignoptimizer.sequencer <sequencer.py> stop
    python -m qdesignoptimizer.sequencer <sequencer.py> requeue <completed folder name>
    python -m qdesignoptimizer.sequencer <sequencer.py> park <id>     (on hold; unpark <id> queues it)
    python -m qdesignoptimizer.sequencer <sequencer.py> move <id> --first | --last | --to N
        (or --before ID / --after ID: next to another queued study)
    python -m qdesignoptimizer.sequencer <sequencer.py> write <id> <file> [--from PATH] [--by NAME]
        (replace one file of a queued or held study; the text is read from
        PATH or standard input. Refused while someone edits the study in
        the GUI; the GUI shows "being edited by NAME" meanwhile.)

Several projects, one queue (``workspace``): name more with ``-p``; a
project may also be given as its folder::

    python -m qdesignoptimizer.sequencer <design A> -p <design B> run --yes

Studies are then ``<project>/<id>`` and catalog keys ``<project>/<key>``
(a bare one does when only one project has it).

Exit codes: 0 ok, 2 something is not valid, 1 refused or failed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

from . import state as st
from .code_change import CONTINUE, RESTART, decide, waiting
from .lease import LeaseHeld, release_lease, take_lease
from .loading import BackendError
from .queue import QueueError
from .runner import Runner, RunRefused, check_queue, configure_console_logging
from .sequence import SequenceError, load_sequence
from .worker import WorkerError
from .workspace import Workspace, open_workspace

#: Who a command line is, in the GUI: an agent sets QDO_SEQUENCER_AGENT.
_AGENT = os.environ.get("QDO_SEQUENCER_AGENT", "the command line")


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m qdesignoptimizer.sequencer``."""
    args_in = sys.argv[1:] if argv is None else list(argv)
    if args_in == ["guide"]:
        # No project: the guide is the package's, for any project.
        sys.stdout.write(guide_text())
        return 0
    parser = argparse.ArgumentParser(prog="python -m qdesignoptimizer.sequencer")
    parser.add_argument(
        "project", help="the project's sequencer.py (its adapter) or folder"
    )
    parser.add_argument(
        "-p",
        "--project",
        dest="more",
        action="append",
        default=[],
        help="another project sharing the queue (repeatable)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="the queue and the latest completed")
    p_add = sub.add_parser("add", help="queue one study")
    p_add.add_argument("task")
    p_add.add_argument("--settings", default="{}")
    p_add.add_argument("--campaign", default="", help="queue it in this campaign")
    p_add.add_argument(
        "--needs",
        action="append",
        default=[],
        metavar="ID",
        help="a study that must have run successfully first (repeatable); "
        "until then the studies after it move up",
    )
    p_sequence = sub.add_parser("sequence", help="queue every study of one")
    p_sequence.add_argument("sequence")
    p_sequence.add_argument(
        "--campaign", default=None, help="queue them in this campaign"
    )
    for p in (p_add, p_sequence):
        p.add_argument(
            "--by",
            default=_AGENT,
            help="who queues it, as the GUI shows it (default: "
            "$QDO_SEQUENCER_AGENT or 'the command line')",
        )
        p.add_argument("--why", default="", help="why it is queued")
    p_decide = sub.add_parser(
        "decide",
        help="continue or restart a study sequence held for a code change",
    )
    p_decide.add_argument("sequence", nargs="?", default="")
    p_decide.add_argument("choice", nargs="?", choices=[CONTINUE, RESTART])
    p_wait = sub.add_parser(
        "wait", help="wait until studies have run; print where their results are"
    )
    p_wait.add_argument("ids", nargs="+")
    p_wait.add_argument(
        "--timeout", type=float, default=None, help="give up after S seconds (exit 3)"
    )
    p_wait.add_argument("--every", type=float, default=5.0, help=argparse.SUPPRESS)
    p_camp = sub.add_parser("campaign", help="list, make or promote campaigns")
    p_camp.add_argument("action", choices=["list", "new", "promote"])
    p_camp.add_argument("key", nargs="?", default="")
    p_camp.add_argument("names", nargs="*", help="promote only these values")
    p_camp.add_argument("--question", default="", help="what the campaign answers")
    p_camp.add_argument(
        "--shared",
        action="store_true",
        help="no own copy of the store: use the parent's (or the project's)",
    )
    sub.add_parser("check", help="rebuild every queued study; no solve")
    sub.add_parser(
        "check-templates",
        help="fill in every template and check it (nothing is queued)",
    )
    p_run = sub.add_parser("run", help="run the queue")
    p_run.add_argument(
        "--yes", action="store_true", help="confirm: this solves for real"
    )
    sub.add_parser("stop", help="stop once the current study is done")
    p_re = sub.add_parser("requeue", help="queue a completed study again")
    p_re.add_argument("folder")
    p_add.add_argument(
        "--hold", action="store_true", help="make it on hold, not in the queue"
    )
    p_park = sub.add_parser("park", help="put a queued study on hold")
    p_park.add_argument("id")
    p_unpark = sub.add_parser("unpark", help="move a study on hold to the queue")
    p_unpark.add_argument("id")
    p_move = sub.add_parser("move", help="move a queued study in the run order")
    p_move.add_argument("id")
    where = p_move.add_mutually_exclusive_group(required=True)
    where.add_argument("--first", action="store_true", help="run it next")
    where.add_argument("--last", action="store_true", help="run it last")
    where.add_argument("--to", type=int, help="to this place (1 = first)")
    where.add_argument("--before", help="just before this queued study")
    where.add_argument("--after", help="just after this queued study")
    p_write = sub.add_parser("write", help="replace one file of a queued or held study")
    p_write.add_argument("id")
    p_write.add_argument("file", help="e.g. job.yaml, ministudy.yaml")
    p_write.add_argument("--from", dest="source", help="read it here (else stdin)")
    p_write.add_argument(
        "--by",
        default=_AGENT,
        help="who is editing, as the GUI shows it (default: "
        "$QDO_SEQUENCER_AGENT or 'the command line')",
    )
    args = parser.parse_args(args_in)

    try:
        workspace = open_workspace([args.project, *args.more])
    except (BackendError, WorkerError, QueueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    try:
        return _command(args, workspace)
    finally:
        workspace.close()


def _command(args, workspace: Workspace) -> int:
    first = next(iter(workspace.projects.values()))
    words = first.words
    state_dir = workspace.state_dir
    try:
        if args.command == "list":
            return _list(workspace)
        if args.command == "add":
            project, task = workspace.locate_task(args.task)
            job_id = project.add(
                task,
                json.loads(args.settings),
                campaign=args.campaign,
                hold=args.hold,
                needs=args.needs,
                added_by=args.by,
                why=args.why,
            )
            if args.hold:
                print(f"on hold: {workspace.key(workspace.name_of(project), job_id)}")
            else:
                print(f"queued {workspace.added(project, [job_id])[0]}")
            return 0
        if args.command == "sequence":
            if Path(args.sequence).is_file():
                if not workspace.single:
                    print(
                        "a study sequence file needs one project; name it by "
                        "<project>/<key> instead",
                        file=sys.stderr,
                    )
                    return 2
                project = first
                spec = load_sequence(Path(args.sequence), words)
            else:
                project, key = workspace.locate_task(args.sequence)
                found = project.backend.catalog().sequence(key)
                if found is None:
                    raise QueueError(f"no {words.sequence} {key!r}")
                spec = found
            if args.campaign is not None:
                spec = replace(spec, campaign=args.campaign)
            ids = workspace.added(
                project, project.add_sequence(spec, added_by=args.by, why=args.why)
            )
            print(f"queued {len(ids)} {words.task}(s): {', '.join(ids)}")
            return 0
        if args.command == "campaign":
            return _campaign(args, workspace)
        if args.command == "check-templates":
            bad = 0
            for name, project in workspace.projects.items():
                for key, problems in project.check_templates().items():
                    shown = workspace.key(name, key)
                    if problems:
                        bad += 1
                        print(f"{shown}:")
                        for problem in problems:
                            print(f"  - {problem}")
                    else:
                        print(f"{shown}: ok")
            print("every template checks out" if not bad else f"{bad} do not")
            return 2 if bad else 0
        if args.command == "check":
            blocked = {k: v for k, v in check_queue(workspace).items() if v}
            for job_id, problems in blocked.items():
                print(f"{job_id}:")
                for problem in problems:
                    print(f"  - {problem}")
            print(
                "all queued studies can run"
                if not blocked
                else f"{len(blocked)} cannot run"
            )
            return 2 if blocked else 0
        if args.command == "run":
            if not args.yes:
                print("run solves for real; add --yes", file=sys.stderr)
                return 1
            configure_console_logging()
            final = Runner(workspace).run()
            print(f"runner {final.status}: {final.message}")
            failed = list(final.failed)
            if failed:
                # on_failure: continue ran on past them; still a failure.
                print(f"{len(failed)} failed: " + ", ".join(failed))
            ended_well = (st.FINISHED, st.STOPPED)
            return 0 if final.status in ended_well and not failed else 1
        if args.command == "stop":
            runner = st.RunnerState.load(state_dir)
            if not runner.alive(state_dir):
                print("no runner is going")
                return 1
            st.request_stop(state_dir)
            print(f"stop requested; {runner.current or 'the study'} " f"finishes first")
            return 0
        if args.command == "park":
            project, job_id = workspace.locate(args.id)
            project.park(job_id)
            print(f"{args.id}: on hold")
            return 0
        if args.command == "write":
            return _write(args, workspace)
        if args.command == "wait":
            return _wait(args, workspace)
        if args.command == "decide":
            return _decide(args, workspace)
        if args.command == "move":
            order = workspace.move(
                args.id,
                to=1 if args.first else (10**9 if args.last else args.to),
                before=args.before,
                after=args.after,
            )
            moved = workspace.queued_key(args.id, order)
            print(f"{moved} is now {order.index(moved) + 1} of {len(order)}")
            return 0
        if args.command == "unpark":
            project, job_id = _locate_held(workspace, args.id)
            project.unpark(job_id)
            print(f"queued {workspace.added(project, [job_id])[0]}")
            return 0
        project, folder = workspace.locate(args.folder)
        path = project.queue.completed_dir / folder
        new_id = project.queue.requeue(path, words)
        print(f"queued {workspace.added(project, [new_id])[0]}")
        return 0
    except (QueueError, RunRefused) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (SequenceError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


def _list(workspace: Workspace) -> int:
    keys = workspace.ids()
    print(f"queue ({len(keys)}):")
    for n, key in enumerate(keys, 1):
        project, job_id = workspace.locate(key)
        problems = project.problems(job_id)
        flag = f"  !! {problems[0]}" if problems else ""
        print(f"  {n:2d}. {key}{flag}")
    held = [
        workspace.key(name, i)
        for name, project in workspace.projects.items()
        for i in project.hold.ids()
    ]
    print(f"on hold ({len(held)}):")
    for key in held:
        print(f"      {key}")
    print("completed (latest 10):")
    for project, path in workspace.completed(10):
        result = st.result_of(path)
        name = workspace.key(workspace.name_of(project), path.name)
        print(
            f"  {result.get('status', '?'):8s} {name}  " f"{result.get('message', '')}"
        )
    out = workspace.state_dir
    runner = st.RunnerState.load(out)
    print(
        f"runner: {runner.status}"
        f"{' (alive)' if runner.alive(out) else ''} {runner.message}"
    )
    return 0


def _campaign(args, workspace: Workspace) -> int:
    if not workspace.single:
        print("campaign commands take one project", file=sys.stderr)
        return 2
    project = next(iter(workspace.projects.values()))
    if args.action == "list":
        found = project.campaigns()
        if not found:
            print("no campaigns (make one: campaign new <key>)")
        for item in found:
            print(
                f"{'  ' * item.depth}{item.key}  {item.title}  "
                f"(store: {_shown(project, project.campaign_store(item.key))})"
            )
        return 0
    if not args.key:
        print(f"campaign {args.action} needs a campaign key", file=sys.stderr)
        return 2
    if args.action == "new":
        folder = project.new_campaign(args.key, args.question, not args.shared)
        print(f"made {folder}; say in its README.md what it is for")
        return 0
    changed = project.promote(args.key, args.names or None)
    parent = args.key.replace("\\", "/").strip("/").rpartition("/")[0]
    where = _shown(project, project.campaign_store(parent))
    if not changed:
        print(f"nothing to promote: {where} has the same values")
    else:
        print(f"promoted to {where}: " + ", ".join(sorted(changed)))
    return 0


def _shown(project, path: Path | None) -> str:
    """A store's path, relative to the project where it can be."""
    if path is None:
        return "the project's"
    try:
        return Path(path).resolve().relative_to(project.root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _write(args, workspace: Workspace) -> int:
    """Replace one file of a queued or held study under a lease, then
    say whether the study can still run."""
    try:
        project, job_id = workspace.locate(args.id)
        queue = project.queue
        if not queue.folder(job_id).is_dir():
            raise QueueError(f"{args.id} is not queued")
    except QueueError:
        project, job_id = _locate_held(workspace, args.id)
        queue = project.hold
    folder = queue.folder(job_id)
    if not folder.is_dir():
        raise QueueError(f"{args.id} is not queued or on hold")
    text = (
        Path(args.source).read_text(encoding="utf-8")
        if args.source
        else sys.stdin.read()
    )
    try:
        take_lease(folder, args.by, seconds=60.0)
    except LeaseHeld as exc:
        print(f"{args.id} is {exc}; nothing written", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        raise QueueError(f"{args.id} moved meanwhile; nothing written") from exc
    try:
        queue.write_file(job_id, args.file, text)
    finally:
        release_lease(folder, args.by)
    problems = project.problems(job_id, folder=folder)
    print(f"wrote {args.file} of {args.id}")
    if problems:
        print("cannot run yet:")
        for problem in problems:
            print(f"  - {problem}")
        return 2
    return 0


def _decide(args, workspace: Workspace) -> int:
    """List the sequences held for a code change, or decide one."""
    held = {
        workspace.key(name, sequence): (project, sequence, ids)
        for name, project in workspace.projects.items()
        for sequence, ids in waiting(project).items()
    }
    if not args.sequence:
        if not held:
            print("no study sequence waits for a decision")
        for key, (_, _, ids) in held.items():
            print(f"{key}: the code changed since it began; {len(ids)} to run")
        return 0
    if not args.choice:
        print(f"say {CONTINUE} or {RESTART}", file=sys.stderr)
        return 2
    found = [v for k, v in held.items() if args.sequence in (k, v[1])]
    if len(found) != 1:
        print(
            (
                f"{args.sequence} does not wait for a decision"
                if not found
                else f"{args.sequence} is ambiguous: name it <project>/{args.sequence}"
            ),
            file=sys.stderr,
        )
        return 1
    project, sequence, _ = found[0]
    made = decide(project, sequence, args.choice)
    if made:
        keys = workspace.added(project, made)
        print(
            f"restarted {sequence} as {project.job(made[0]).sequence.get('id')}: "
            f"{', '.join(keys)}; the rest of {sequence} is on hold"
        )
    else:
        print(f"{sequence} continues on the new code")
    return 0


def guide_text() -> str:
    """The agent guide that ships with the package (``AGENT_GUIDE.md``)."""
    return (Path(__file__).resolve().parent.parent / "AGENT_GUIDE.md").read_text(
        encoding="utf-8"
    )


def _wait(args, workspace: Workspace) -> int:
    """Wait until each study has run -- neither it nor a requeued copy
    is queued any more -- and print its latest run as it lands."""
    pending = {key: _locate_any(workspace, key) for key in args.ids}
    deadline = None if args.timeout is None else time.monotonic() + args.timeout
    worst = 0
    said_runner = False
    first = True
    while pending:
        for key, (project, local) in list(pending.items()):
            where = project.pending_state(local)
            if where == "queued":
                continue
            del pending[key]
            if where == "hold":
                print(f"{key}: on hold; nothing runs it until it is queued", flush=True)
                worst = max(worst, 1)
                continue
            runs = project.runs_of(local)
            if not runs:
                print(f"{key}: not queued, on hold or completed", flush=True)
                worst = 2
                continue
            if not _print_run(workspace, project, runs[-1]):
                worst = max(worst, 1)
        if not pending:
            break
        if first:
            print("waiting for " + ", ".join(pending), flush=True)
            first = False
        state = workspace.state_dir
        if not said_runner and not st.RunnerState.load(state).alive(state):
            print("(no runner is going: waiting for one to start)", flush=True)
            said_runner = True
        if deadline is not None and time.monotonic() >= deadline:
            print(f"timed out; still queued: {', '.join(pending)}", flush=True)
            return 3
        time.sleep(args.every)
    return worst


def _print_run(workspace: Workspace, project, folder: Path) -> bool:
    """Print one run's result and where it is; whether it succeeded."""
    result = st.result_of(folder)
    status = result.get("status", "?")
    print(f"{status} {workspace.key(workspace.name_of(project), folder.name)}")
    print(f"  folder: {folder.resolve()}")
    for label, key in (
        ("message", "message"),
        ("outputs", "outputs"),
        ("wrote back", "committed"),
        ("checks", "checks"),
        ("files", "files"),
    ):
        value = result.get(key)
        if value:
            shown = value if isinstance(value, str) else json.dumps(value)
            print(f"  {label}: {shown}")
    sys.stdout.flush()
    return status == st.DONE


def _locate_any(workspace: Workspace, key: str):
    """The project and id of a study wherever it is (queued, on hold,
    completed); a bare id with several projects open must be in one."""
    if "/" in key or workspace.single:
        return workspace.locate(key)
    found = [
        (project, key)
        for project in workspace.projects.values()
        if project.pending_state(key) or project.runs_of(key)
    ]
    if len(found) != 1:
        raise QueueError(
            f"{key!r}: name the project as <project>/{key}"
            if found
            else f"{key!r} is in none of the open projects"
        )
    return found[0]


def _locate_held(workspace: Workspace, key: str):
    """The project and id of a study on hold (``<project>/<id>`` with
    several projects open, the bare id when it is unique)."""
    found = [
        (project, i)
        for name, project in workspace.projects.items()
        for i in project.hold.ids()
        if key in (i, workspace.key(name, i), f"{name}/{i}")
    ]
    if len(found) != 1:
        raise QueueError(
            f"{key} is not on hold" if not found else f"{key} is ambiguous"
        )
    return found[0]
