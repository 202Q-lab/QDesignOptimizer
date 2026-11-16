# Working the simulation sequencer as an agent

This is how Claude (or another agent) works on a chip project's simulation
queue next to the person who watches it in the GUI. Copy this file into
your own chip project (next to `simulations/`) so it applies there too.

The person checks what you do in the GUI. Everything you do must be
visible and explained there: who queued a study, why, and what it waits
for.

## Ground rules

- **Use the command line**, never the files under `out/` directly:
  `python -m qdesignoptimizer.sequencer <project> <command>`.
  `<project>` is the chip's folder (or its `simulations/sequencer.py`).
- **Open the same projects the GUI has.** With several connected in the
  GUI, name all of them: `<project A> -p <project B> ...`. Otherwise
  your studies sit in a different order from the one the person sees.
  Studies are then `<project>/<id>`.
- **Say who you are and why.** Set `QDO_SEQUENCER_AGENT=claude` in the
  environment, or pass `--by claude`, and pass `--why "<one line>"` on
  every `add` and `sequence`. The GUI tags the study "by claude"; the
  reason shows on hover and in the study editor.
- **Never start or stop the runner** unless the person asks: `run`
  solves in Ansys. Queue studies; the person presses Start. A runner
  already going picks up new studies before its next one.
- **Never touch a study someone is editing.** `write` refuses then
  (exit 1, "being edited by ..."); wait and try again, or ask.

## Commands

| What | Command |
|---|---|
| See the queue, on hold, latest results, runner | `list` |
| Queue a study | `add <study> --why "..." [--needs <id>] [--campaign KEY] [--settings JSON]` |
| Make it on hold (to fill in before it runs) | `add <study> --hold --why "..."` |
| Queue a study sequence | `sequence <name or file.yaml> --why "..."` |
| Reorder | `move <id> --first / --last / --to N / --before <id> / --after <id>` |
| Change one file of a queued or held study | `write <id> <file> --from <path>` (or the text on standard input) |
| Put on hold / queue again | `park <id>` / `unpark <id>` |
| Check what would stop a queued study (no solve) | `check` |
| Run a completed study again | `requeue <completed folder name>` |
| Wait for results | `wait <id> [<id> ...] [--timeout S]` |

`write` says afterwards whether the study can still run (exit 2 and the
problems if not). Fix them before moving on.

## Order and dependencies

- A study that uses what another one writes back (`save: true`) lists it
  with `--needs <id>` (repeatable). The runner passes it over until the
  needed study has run successfully, and runs the ones after it
  meanwhile. A study sequence fills this in itself.
- A need that failed or is missing is a problem the person sees; requeue
  the failed one (`requeue`) rather than editing `needs` away.

## Waiting for results

Run `wait <id>` in the background (Bash `run_in_background`): it returns
when the study has run and prints its status, its folder, the message,
outputs, the values it wrote back, its checks and its files. You are
notified when it exits, so you can analyse as soon as the result lands
instead of polling. Exit codes: 0 done, 1 failed or on hold, 2 unknown
id, 3 `--timeout` reached. A requeued copy (`<id>_r2`) counts: `wait`
follows the latest run.

## Reading results

A completed study is `out/completed/<id>_<start time>/`:

- `result.json`: status, message, outputs, `updates` (what it found),
  `committed` (what it wrote back to the store), checks;
- `study.log`, `convergence.png`, the picture of what it simulated;
- `inputs/` (the files as queued) and `resolved/` (as run, every value
  filled in).

The store is `design_variables.json`; `design_variables.meta.json` says
which study wrote each value and when. A campaign's study writes its
campaign's store instead (`campaigns/<name>/`).

When you report back, name the studies by id, quote the numbers from
`result.json`, and say what you would queue next and why. Queue it only
with `--why`, so the person can check it in the GUI.
