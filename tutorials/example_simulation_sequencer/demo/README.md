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
| `campaigns/tune_resonators/README.md` | a campaign: a lab-log folder for one question, here sharing the chip's store (see *Campaigns* below) |

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
variable the way QDO does, spends 5 s over the study's `iterations`
(default 3), and writes the tuned values back into
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

Open <http://127.0.0.1:8051>. Three areas, each scrolling on its own;
drag the divider between left and right to share the width. Top left,
the **Shared queue** of every connected project: **Start**, **Stop
after this study**, **Check queue**, the queue (drag rows to reorder;
**Add** a study or a study sequence to it), the studies on hold, the
completed ones and the runner log. Right, the **Study editor**: a study
you pick opens read only (*Inspecting*); press **Edit** to change one
that has not run yet (*Editing*, with **Done editing**, and for one on
hold **Add to queue ▶**). A study that has run stays read only: **Edit
copy** makes a copy of it on hold, opened for editing. Bottom left, the
**Projects** area, one project at a time. The project picker's last entry, **+ Connect
or disconnect project folders**, lists the connected ones (✕
disconnects; type a folder and **Connect** to add one; the list is
remembered, so `python -m qdesignoptimizer.sequencer.app` with no
folder opens them again). Below it, the project worked on (its folder
relative to the repository; hover for the full path), then its study
sequence templates, study templates and campaigns, each heading with
**+ New**. Pick a campaign to work in: it opens in the editor, and
**+ New study** makes studies in it, from a study template or blank,
on hold or straight into the queue. Below, the editor of whatever you
pick opens in the Study editor. Each study "solves" for 5 s; a `tune_R1`
study sequence plus one more study takes about 15 s.

### Things to try

- **A chained sequence**: Add `tune_all`, then Start. Each linewidth
  study starts from the frequency its frequency study wrote back; click
  a completed study for its files and convergence plot, and see
  `chip_qdo/design_variables.meta.json` for which study wrote each value.
- **A failure that does not stop the rest**: open `tune_all` in the
  browser, give its first study (`R1.frequency`) the settings
  `iterations: 1`, save the template and add it to the queue. It
  ends off target, fails its `converged` check and writes nothing back;
  under `on_failure: continue` the other three still run. Under `abort`
  (the default) the runner stops at the failed study. (In `chip_a`,
  `fail: true` in a queued study's `task.yaml` makes it raise instead.)
- **Per-study values**: click a queued chip_qdo study and set its
  *Parameter targets* or *Design variables*: they hold only for that
  study (written to its folder) and are never written back unless a
  target tunes them.
- **On hold**: put a queued study on hold (⏸) and it moves to the On hold
  list, out of the run order: the runner skips it, and you edit it there
  until you queue it again (▶). It lives in `out/hold/` (CLI: `park <id>`,
  `unpark <id>`, `add <study> --hold`).
- **Notes**: every study has a notes field; `tune_all` sets one.
- **Requeue** a completed study: *fresh* pulls its values from the
  project again, *exact* reruns with the values it ran with (GUI;
  CLI `requeue <completed folder>` is the fresh one).
- **New study sequence**: + New study sequence opens an empty template:
  give it a file name, defaults for every study, and its studies in
  order; **Save template** writes `study_sequences/<name>.yaml`, and
  **Add to queue** queues its studies. A template built in the
  adapter's code (`tune_R1`) opens read-only: save it under a new file
  name and a new name to change it. Every template has a name of its
  own; saving one under a name another has is refused.
- **New study in the Study editor**: its header has **+ New blank study**
  (a MiniStudy from scratch) and **+ New study from template**; both
  make it in the project and campaign picked on the left, on hold or
  straight into the queue.
- **New study**: + New study (next to the campaign picker) makes one
  from a study template or a blank MiniStudy, in the campaign picked,
  on hold (edit it, then ▶) or straight into the queue.
- **New study template**: its + New starts one as a study on hold (a
  copy of a template, or blank); edit it, then **Save as template** at
  the bottom of its page.
- **Check before running**: *Check queue* in the GUI, or `check` in the
  CLI, rebuilds every queued study without solving. For one study,
  **Validate** in the Study editor checks the same: that its files read,
  its objects build, the names it gives are in the design and its
  targets and values resolve.
- **Pre-figures**: **Generate pre-figures** draws what each solver run
  of a study simulates (its components, with the airbridges the solve
  adds), the chip built with the design variables the study would
  fetch now (its campaign's or the project's, plus its own). The project
  needs a `design(variables)` hook (`chip_qdo` has none; the tutorial's
  `coupled_transmon_chip` has one).
- **Rename a template**: a study template's page has **Rename** (its
  file in `ministudies/`); a study sequence template is renamed by
  changing its file name and saving it. Studies already made keep their
  own copy.
- **Notes** at the top go away after a few seconds; an error stays
  until you click it.
- **Reload templates** (in the Projects area) reads every project's templates, study
  sequences and adapter (`sequencer.py`, and what it imports) again:
  after editing them outside the GUI.
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

Move a study (here to the front), or replace one of its files the way
an agent would; with the GUI open, watch both show up there. `write`
is refused while that study is open in Edit mode in the GUI:

```bash
python -m qdesignoptimizer.sequencer tutorials/example_simulation_sequencer/demo/chip_qdo move s04-tune_R1_file-r1_linewidth --first
```

```bash
python -m qdesignoptimizer.sequencer tutorials/example_simulation_sequencer/demo/chip_qdo write s04-tune_R1_file-r1_linewidth job.yaml --from my_job.yaml --by claude
```

## Campaigns

A campaign is a folder under `chip_qdo/campaigns/` for one question,
with a `README.md` as its lab log. `tune_resonators` is one; try a
sub-campaign of it with its own copy of the design variables, so its
studies change nothing in the chip until you promote them:

```bash
python -m qdesignoptimizer.sequencer tutorials/example_simulation_sequencer/demo/chip_qdo campaign new tune_resonators/r2_check --question "Does R2 converge from the start values?"
```

```bash
python -m qdesignoptimizer.sequencer tutorials/example_simulation_sequencer/demo/chip_qdo sequence tune_R2 --campaign tune_resonators/r2_check
```

```bash
python -m qdesignoptimizer.sequencer tutorials/example_simulation_sequencer/demo/chip_qdo run --yes
```

The two studies wrote to
`chip_qdo/campaigns/tune_resonators/r2_check/design_variables.json`
only; `chip_qdo/design_variables.json` is as it was. List the campaigns
(each with the store it uses), then promote the sub-campaign's values
one level up: to the store `tune_resonators` uses, which is the chip's.
The write is recorded in `chip_qdo/design_variables.meta.json` as
`promoted_from`:

```bash
python -m qdesignoptimizer.sequencer tutorials/example_simulation_sequencer/demo/chip_qdo campaign list
```

```bash
python -m qdesignoptimizer.sequencer tutorials/example_simulation_sequencer/demo/chip_qdo campaign promote tune_resonators/r2_check
```

## Reset

Delete each project's `out/`, `chip_a/store.json`, `chip_b/store.json`,
`chip_qdo/design_variables.json` and the `*.meta.json` next to them, and
any sub-campaign you made under `chip_qdo/campaigns/tune_resonators/`
(all git-ignored). With several
projects open, the shared order and runner state live in your user
folder (`%LOCALAPPDATA%\qdo_sequencer\workspaces\`), which is safe to delete while no runner is going.
