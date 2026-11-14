# Sequencer demo: no Ansys needed

Three tiny projects whose "simulation" takes **5 seconds**, to try the
queue, the runner and the GUI without Ansys:

| Project | Backend | Resonators | Studies |
| --- | --- | --- | --- |
| `chip_qdo` | QDO, dummy solver | R1, R2 | `<R>.frequency`, `<R>.linewidth`; measurement plan `tune_<R>` |
| `chip_a` | fake | R1, R2 | the same names |
| `chip_b` | fake | R3 | the same names |

**`chip_qdo` is the real QDO path** with a dummy in place of Ansys
(`dummy_solver.py`). What it can queue comes from its templates:

| File | What it shows |
| --- | --- |
| `ministudies/frequency.yaml` | a MiniStudy and nothing else. `{subject}` makes it one study per resonator of the adapter's `subjects`: `R1.frequency`, `R2.frequency` |
| `ministudies/linewidth.yaml` | a chained `ResonatorDecayIntoWaveguideStudy`; its `mode_freq_GHz` is `"=p['resonator_{subject}_freq'] / 1e9"`, read from `parameter_targets.py` when queued |
| `opt_targets/resonators.yaml` | the project's opt targets, each with a `name` (unique in the project, over however many files) and a `tolerance` |

A study is made when it is queued, and lives only in `out/`: the
MiniStudy, filled in, plus every opt target whose modes are its modes
(here the frequency and the linewidth of its resonator). The one its
simulation produces is switched on; the other is kept in the study's
`opt_targets.yaml` with `active: false`, one flag to flip. `job.yaml`
in the folder is its queue entry (measurement plan, run settings, checks).

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

Then: **Add** a study or a `tune_R1` measurement plan, drag rows to reorder, click
a study to edit its files, **Start** (it asks you to confirm), and watch
the runner banner, the Completed list and the runner log. Open
<http://127.0.0.1:8051>. Each study "solves" for 5 s; a `tune_R1`
measurement plan plus one more study takes about 15 s.

## Command line

```bash
python -m qdesignoptimizer.sequencer tutorials/example_simulation_sequencer/demo/chip_a plan tune_R1
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

Delete each project's `out/`, `chip_a/store.json`, `chip_b/store.json`
and `chip_qdo/design_variables.json` (all git-ignored). With several
projects open, the shared order and runner state live in your user
folder (`%LOCALAPPDATA%\qdo_sequencer\workspaces\`), which is safe to delete while no runner is going.
