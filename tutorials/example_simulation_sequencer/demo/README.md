# Sequencer demo: no Ansys needed

Three tiny projects whose "simulation" takes **5 seconds**, to try the
queue, the runner and the GUI without Ansys:

| Project | Backend | Resonators | Studies |
| --- | --- | --- | --- |
| `chip_qdo` | QDO, dummy solver | R1, R2 | `<R>.frequency`, `<R>.linewidth`; study sequences `tune_<R>` (in code) and `tune_all` (a file) |
| `chip_a` | fake | R1, R2 | the same names |
| `chip_b` | fake | R3 | the same names |

**`chip_qdo` is the real QDO path** with a dummy in place of Ansys
(`dummy_solver.py`). What it can queue comes from its templates:

| File | What it shows |
| --- | --- |
| `ministudies/frequency.yaml` | a MiniStudy and nothing else. `{subject}` makes it one study per resonator of the adapter's `subjects`: `R1.frequency`, `R2.frequency` |
| `ministudies/linewidth.yaml` | a chained `ResonatorDecayIntoWaveguideStudy`; its `mode_freq_GHz` is `"=p['resonator_{subject}_freq'] / 1e9"`, read from `parameter_targets.py` when queued |
| `opt_targets/resonators.yaml` | the project's opt targets, each with a `name` (unique in the project, over however many files) and a `tolerance` |
| `study_sequences/tune_all.yaml` | a study sequence as a file: four chained studies, `defaults` for all of them, one entry with its own settings and notes, `on_failure`, `why_ordered` |
| `simulations/sequencer.py` | the adapter; its `study_sequences()` builds `tune_R1`, `tune_R2` in code |

A study is made when it is queued, and lives only in `out/`: the
MiniStudy, filled in, plus every opt target whose modes are its modes
(here the frequency and the linewidth of its resonator). The one its
simulation produces is switched on; the other is kept in the study's
`opt_targets.yaml` with `active: false`, one flag to flip. `job.yaml`
in the folder is its queue entry (study sequence, run settings, checks).

Edit a template and the Add menu has it at once; save any study's
MiniStudy as a new one with "Save as template"; check them all with

```bash
python -m qdesignoptimizer.sequencer tutorials/example_simulation_sequencer/demo/chip_qdo check-templates
```

The dummy measures a toy resonator model, moves each target's design
variable the way QDO does, spends the study's `seconds` over its
`iterations` (default 5 s over 3), and writes the tuned values back into
`chip_qdo/design_variables.json` -- it starts as a copy of
`design_variables.start.json`. Edit `parameter_targets.py` to move a
target; edit a study's settings, nested capacitance study or opt
targets in the GUI before it runs.

`chip_a` and `chip_b` use the fake backend: a study "proposes" the
`value` in its `task.yaml` and writes it back to the project's
`store.json` when `save` is on and its checks pass. Set `fail: true` to
see a failed study, `converged: false` to see a failed check.

Run from the QDesignOptimizer repo root, with QDO installed (`poetry install`).

## GUI

One project:

```bash
python -m qdesignoptimizer.sequencer.app tutorials/example_simulation_sequencer/demo/chip_qdo
```

All three, one shared queue (each project in its own worker process):

```bash
python -m qdesignoptimizer.sequencer.app tutorials/example_simulation_sequencer/demo
```

Then: **Add** a study, or a study sequence such as `tune_R1` (a popup
lists its studies; **Launch** queues them), drag rows to reorder, click
a study to edit its files, **Start** (it asks you to confirm), and watch
the runner banner, the Completed list and the runner log. Open
<http://127.0.0.1:8051>. Each study "solves" for 5 s; a `tune_R1`
study sequence plus one more study takes about 15 s.

### Things to try

- **A chained sequence**: Add `tune_all`, Launch, Start. Each linewidth
  study starts from the frequency its frequency study wrote back; click
  a completed study for its files and convergence plot, and see
  `chip_qdo/design_variables.meta.json` for which study wrote each value.
- **A failure that does not stop the rest**: in `tune_all`'s popup, give
  the first study (`R1.frequency`) `iterations: 1` before you Launch. It
  ends off target, fails its `converged` check and writes nothing back;
  under `on_failure: continue` the other three still run. Under `abort`
  (the default) the runner stops at the failed study. (In `chip_a`,
  `fail: true` in a queued study's `task.yaml` makes it raise instead.)
- **Per-study values**: click a queued chip_qdo study and set its
  *Parameter targets* or *Design variables*: they hold only for that
  study (written to its folder) and are never written back unless a
  target tunes them.
- **Hold**: put a queued study on hold; the runner stops there until you
  release it (CLI: `hold <id>`, `release <id>`).
- **Notes**: every study has a notes field; `tune_all` sets one.
- **Requeue** a completed study: *fresh* pulls its values from the
  project again, *exact* reruns with the values it ran with (GUI;
  CLI `requeue <completed folder>` is the fresh one).
- **New study sequence**: build one in the popup from scratch or from
  an existing sequence, then Launch it.
- **Check before running**: *Check queue* in the GUI, or `check` in the
  CLI, rebuilds every queued study without solving.
- **Stop** between studies: *Stop after this study*, or `stop` in the CLI.

## Command line

```bash
python -m qdesignoptimizer.sequencer tutorials/example_simulation_sequencer/demo/chip_a sequence tune_R1
```

```bash
python -m qdesignoptimizer.sequencer tutorials/example_simulation_sequencer/demo/chip_a -p tutorials/example_simulation_sequencer/demo/chip_b add chip_b/R3.frequency
```

```bash
python -m qdesignoptimizer.sequencer tutorials/example_simulation_sequencer/demo/chip_a -p tutorials/example_simulation_sequencer/demo/chip_b list
```

```bash
python -m qdesignoptimizer.sequencer tutorials/example_simulation_sequencer/demo/chip_a -p tutorials/example_simulation_sequencer/demo/chip_b run --yes
```

## Reset

Delete each project's `out/`, `chip_a/store.json`, `chip_b/store.json`,
`chip_qdo/design_variables.json` and the `*.meta.json` next to them (all
git-ignored). With several
projects open, the shared order and runner state live in your user
folder (`%LOCALAPPDATA%\qdo_sequencer\workspaces\`), which is safe to delete while no runner is going.
