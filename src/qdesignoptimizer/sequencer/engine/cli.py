"""Command line.

    python -m qdesignoptimizer.sequencer <sequencer.py> list
    python -m qdesignoptimizer.sequencer <sequencer.py> add <study> [--settings JSON]
    python -m qdesignoptimizer.sequencer <sequencer.py> measurement plan <file.yaml or key>
    python -m qdesignoptimizer.sequencer <sequencer.py> check
    python -m qdesignoptimizer.sequencer <sequencer.py> check-templates
    python -m qdesignoptimizer.sequencer <sequencer.py> run --yes
    python -m qdesignoptimizer.sequencer <sequencer.py> stop
    python -m qdesignoptimizer.sequencer <sequencer.py> requeue <completed folder name>
    python -m qdesignoptimizer.sequencer <sequencer.py> hold <id>     (or release <id>)

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
import sys
from pathlib import Path

from . import state as st
from .loading import BackendError
from .plan import PlanError, load_plan
from .queue import QueueError
from .runner import Runner, RunRefused, check_queue, configure_console_logging
from .worker import WorkerError
from .workspace import Workspace, open_workspace


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m qdesignoptimizer.sequencer``."""
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
    p_plan = sub.add_parser("plan", help="queue every study of one")
    p_plan.add_argument("plan")
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
    p_hold = sub.add_parser("hold", help="the runner stops at this study")
    p_hold.add_argument("id")
    p_rel = sub.add_parser("release", help="take a study off hold")
    p_rel.add_argument("id")
    args = parser.parse_args(argv)

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
            job_id = project.add(task, json.loads(args.settings))
            print(f"queued {workspace.added(project, [job_id])[0]}")
            return 0
        if args.command == "plan":
            if Path(args.plan).is_file():
                if not workspace.single:
                    print(
                        "a measurement plan file needs one project; name it by "
                        "<project>/<key> instead",
                        file=sys.stderr,
                    )
                    return 2
                project = first
                spec = load_plan(Path(args.plan), words)
            else:
                project, key = workspace.locate_task(args.plan)
                found = project.backend.catalog().plan(key)
                if found is None:
                    raise QueueError(f"no {words.plan} {key!r}")
                spec = found
            ids = workspace.added(project, project.add_plan(spec))
            print(f"queued {len(ids)} {words.task}(s): {', '.join(ids)}")
            return 0
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
            ended_well = (st.FINISHED, st.STOPPED, st.HELD)
            return 0 if final.status in ended_well else 1
        if args.command == "stop":
            runner = st.RunnerState.load(state_dir)
            if not runner.alive(state_dir):
                print("no runner is going")
                return 1
            st.request_stop(state_dir)
            print(f"stop requested; {runner.current or 'the study'} " f"finishes first")
            return 0
        if args.command in ("hold", "release"):
            project, job_id = workspace.locate(args.id)
            project.set_hold(job_id, args.command == "hold")
            print(f"{args.id}: {'on hold' if args.command == 'hold' else 'released'}")
            return 0
        project, folder = workspace.locate(args.folder)
        path = project.queue.completed_dir / folder
        new_id = project.queue.requeue(path, words)
        print(f"queued {workspace.added(project, [new_id])[0]}")
        return 0
    except (QueueError, RunRefused) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (PlanError, json.JSONDecodeError) as exc:
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
