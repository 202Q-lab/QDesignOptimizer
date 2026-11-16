# Simulation sequencer: a queue of QDO studies

The notebooks in `examples_coupled_transmon_chip/` run one study at a
time, by hand: build a `MiniStudy` and its `OptTarget`s in Python, make a
`DesignAnalysis`, call `optimize_target` a few times, copy the result
into `design_variables.json`. `qdesignoptimizer.sequencer` turns that
into a queue:

- a **study** is a folder of plain files (`ministudy.yaml`,
  `opt_targets.yaml`, `job.yaml`) made from the project's templates when
  it is queued, editable until it runs and kept, with what it found,
  after;
- studies run **one at a time**, in the order of the queue, unattended;
- a study that converged **writes its tuned design variables back** into
  the project's `design_variables.json`, so the next study starts from
  them, and records each write in `design_variables.meta.json`;
- a **study sequence** queues a named sequence of studies in one go;
- a **campaign** is a lab-log folder for one question (tune up the
  chip, a Purcell filter, mesh sensitivity), optionally with its own
  copy of the design variables, nested as deep as the question needs;
- a Dash **GUI** shows the queue, the running study, the completed ones
  with their convergence plot, and lets you add, reorder, edit and
  requeue studies.

This folder holds two things:

| Folder | Ansys? | What it is |
| --- | --- | --- |
| `demo/` | no | Three tiny projects that "solve" in 5 s. Start here to see the queue, the runner and the GUI work. `demo/chip_qdo` is the full QDO path with a dummy in place of Ansys. |
| `coupled_transmon_chip/` | yes | The chip of `examples_coupled_transmon_chip/`, set up for the sequencer: the same design, its studies as templates. |

## Try it without Ansys

From the repository root:

```bash
python -m qdesignoptimizer.sequencer.app tutorials/example_simulation_sequencer/demo
```

Open <http://127.0.0.1:8051>. Top left is the **Shared queue** of every
connected project (the queue, the studies on hold, the completed ones,
the log); bottom left the **Projects** area, with the project worked on, one at a time (the picker's last entry,
**+ Connect or disconnect project folders**, does that, remembered for
the next start), with its study sequence templates, study templates
and campaigns, each with **+ New**; right, the **Study editor**, where a
study opens read only until you press **Edit** (a study that has run
offers **Edit copy** instead).
**Add** the study sequence `tune_R1` of `chip_qdo` to the queue, then
**Start**. Two studies run (about 10 s), frequency then
linewidth, and each writes the design variable it tuned back into
`demo/chip_qdo/design_variables.json`; click one under **Completed** to
see its files and convergence plot. `tune_all` chains four studies
from a sequence file, `chip_qdo/study_sequences/tune_all.yaml`. More in `demo/README.md` (in the online docs: *Sequencer demo*).

## A project on top of QDO

`coupled_transmon_chip/` is what you add to a QDO design to queue its
studies. Nothing in `design.py` changes: the project imports it from
`../../examples_coupled_transmon_chip/`.

```
coupled_transmon_chip/
├── simulations/sequencer.py      the adapter: HOW a study is solved
├── ministudies/                  WHAT can be queued: one MiniStudy per file
│   ├── resonator_kappa.yaml        -> Q1.resonator_kappa, Q2.resonator_kappa
│   ├── qubit_charge_line.yaml      -> Q1.qubit_charge_line, Q2.qubit_charge_line
│   └── Q1_qubit_resonator.yaml     -> Q1.qubit_resonator
├── opt_targets/                  the project's OptTargets, each with a name
│   ├── qubit_resonator.yaml
│   └── charge_line.yaml
├── study_sequences/tune_Q1.yaml        a named sequence of studies
├── parameter_targets.py          PARAM_TARGETS, as in every QDO project
└── design_variables.json         the store (made on first use, git-ignored)
```

### The adapter: `simulations/sequencer.py`

A file with a `make_backend()` that returns `QDOBackend(project)`. The
project object says where the design lives and how one study is solved;
its `solve` is the loop of the notebooks:

```python
def solve(self, study, targets, folder, settings, variables, params):
    chip = self.design(variables)              # build with the study's values
    analysis = DesignAnalysis(
        DesignAnalysisState(chip, render, params),
        mini_study=study,                      # from the study's ministudy.yaml
        opt_targets=list(targets),             # from its opt_targets.yaml
        save_path=get_save_path(str(folder) + "/", n.CHIP_NAME),
        update_design_variables=False,         # the sequencer writes back
        ...
    )
    for _ in range(settings["iterations"]):
        analysis.optimize_target({}, {})
    return analysis
```

`subjects` names the subsystems a template is made for: here `Q1` and
`Q2`, each with a value `n` that fills `{n}` in the templates. The full
list of hooks (all optional beyond `solve`, `parameter_targets`,
`settings_keys`, `default_settings`, `before_step`) is the `QDOProject`
protocol in `src/qdesignoptimizer/sequencer/backends/qdo.py`.

### MiniStudy templates: `ministudies/*.yaml`

A template is a `MiniStudy` written as its constructor call, nothing
else:

```yaml
class: qdesignoptimizer.design_analysis_types:MiniStudy
args:
  qiskit_component_names: ["name_resonator_{n}_", "name_tee_{n}_"]
  modes: ["resonator_{n}"]
  run_capacitance_studies_only: true
  nbr_passes: 7
  capacitance_matrix_studies:
  - class: qdesignoptimizer.sim_capacitance_matrix:ResonatorDecayIntoWaveguideStudy
    args:
      mode: "resonator_{n}"
      mode_freq_GHz: "=p['resonator_{n}_freq'] / 1e9"
      ...
```

- `{n}` makes the file a family: one study per subsystem that sets `n`
  (`Q1.resonator_kappa`, `Q2.resonator_kappa`).
- A string starting with `=` is an expression of `p` (the parameter
  targets) and `v` (the design variables), evaluated when the study is
  queued.
- A file without placeholders named `<subsystem>_<rest>.yaml` is the one
  study `<subsystem>.<rest>`: `Q1_qubit_resonator.yaml` is
  `Q1.qubit_resonator`. It is written out in full because its
  `jj_setup` is keyed by the junction's name, and keys are not filled in.
- The solver settings (`nbr_passes`, `delta_f`) are set here and nowhere
  else.

### Opt targets: `opt_targets/*.yaml`

The project's targets, not a study's. Each has a `name`, unique in the
project, and either calls a QDO helper or spells the target out:

```yaml
- name: "Q{n}_resonator_freq"
  helper: qdesignoptimizer.utils.optimization_targets:get_opt_target_res_freq_via_length
  args: {resonator: "resonator_{n}"}
  tolerance: {abs: 10.0e+6}    # Hz; or a number: relative
- name: "Q{n}_charge_line_t1"
  target_param_type: charge_line_limited_t1
  involved_modes: ["qubit_{n}"]
  design_var: "design_var_cl_pos_x_qubit_{n}"
  prop_to: "-v['design_var_cl_pos_x_qubit_{n}'] ** 3"
  tolerance: 0.2
```

A queued study takes **every target whose modes are all modes of its
MiniStudy** (a `capacitance` target, whose conductors belong to the
components of one of its chained capacitance studies), and switches on
those whose parameter it measures, in the kind of simulation the target
is tuned in: `kappa`, `charge_line_limited_t1` and `capacitance` in a
capacitance study, the rest in an eigenmode solve, unless the target says
`simulation:`. The eigenmode solve measures each mode's frequency and
linewidth and the nonlinearities; a chained decay study measures one
parameter, of its type, for its mode. The others are written into the
study with `active: false`, and two switched-on targets that tune the
same design variable are refused before anything runs. That is why
`opt_targets/qubit_resonator.yaml` has the resonator linewidth twice:
`Q1_resonator_kappa` is tuned in `Q1.resonator_kappa` (capacitance),
`Q1_resonator_kappa_eigenmode` (`simulation: eigenmode`) in
`Q1.qubit_resonator`, from the feedline ports. A MiniStudy that does
both takes the linewidth from its capacitance study, as QDO does.

`tolerance` decides the study's `converged` check; a study writes back
only when its checks pass (with `save: true`).

### Study sequences: `study_sequences/*.yaml`

```yaml
name: Tune qubit 1 and its resonator
defaults: {settings: {iterations: 3}, checks: [converged], save: true}
studies:
- study: Q1.resonator_kappa
- study: Q1.qubit_charge_line
- {study: Q1.qubit_resonator, settings: {iterations: 4}}
```

A file may name a campaign (`campaign: tuneup_chip`) to queue its
studies there; `sequence <key> --campaign <key>` does the same for any
study sequence.

### Campaigns: `campaigns/<name>/`

A campaign is a folder that answers one question, with a `README.md`
as its lab log: the question, what was run (under which study ids) and
the conclusion. A campaign inside another is a sub-campaign; it is just
a campaign, nested:

```
coupled_transmon_chip/campaigns/
  tuneup_chip/
    README.md
    design_variables.json      its own copy of the store
    q1_readout/                a sub-campaign
      README.md                no store of its own: tuneup_chip's
```

- **Names**: lower-case words joined by `_`, no dates (the README and the
  studies have those). The key is the path: `tuneup_chip/q1_readout`.
- **Studies**: `add <study> --campaign tuneup_chip/q1_readout` (or
  `campaign:` in a study sequence file) queues a study in it.
  `job.yaml` says `campaign: tuneup_chip/q1_readout`; the id and the
  folder under `out/` stay as they are, so Ansys paths stay short.
- **Store**: a campaign's studies read and write the nearest
  `design_variables.json` from their campaign up, the project's when
  none on the way has one. `campaign new` gives a new campaign a copy
  of the store it would otherwise use (`--shared` skips the copy), so
  it changes nothing outside itself until you `campaign promote` its
  values one level up. Promote writes only what the campaign itself
  changed since it was made (or last promoted; its `.promoted.json`
  keeps that starting point), so a value its parent changed meanwhile
  is not overwritten; `campaign promote <key> <name> ...` writes those
  names regardless. The write is recorded in that store's
  `design_variables.meta.json` as `promoted_from`.

```bash
python -m qdesignoptimizer.sequencer tutorials/example_simulation_sequencer/coupled_transmon_chip campaign new tuneup_chip --question "Can every mode reach its target?"
```

```bash
python -m qdesignoptimizer.sequencer tutorials/example_simulation_sequencer/coupled_transmon_chip campaign new tuneup_chip/q1_readout --shared
```

```bash
python -m qdesignoptimizer.sequencer tutorials/example_simulation_sequencer/coupled_transmon_chip sequence tune_Q1 --campaign tuneup_chip/q1_readout
```

```bash
python -m qdesignoptimizer.sequencer tutorials/example_simulation_sequencer/coupled_transmon_chip campaign list
```

```bash
python -m qdesignoptimizer.sequencer tutorials/example_simulation_sequencer/coupled_transmon_chip campaign promote tuneup_chip
```

The demo walks through it without Ansys: *Campaigns* in
`demo/README.md` (in the online docs: *Sequencer demo*).

### Running it

Check the templates (no Ansys; builds the chip to check the component
names):

```bash
python -m qdesignoptimizer.sequencer tutorials/example_simulation_sequencer/coupled_transmon_chip check-templates
```

Queue the study sequence and look at what it made, under
`coupled_transmon_chip/out/queue/`:

```bash
python -m qdesignoptimizer.sequencer tutorials/example_simulation_sequencer/coupled_transmon_chip sequence tune_Q1
```

```bash
python -m qdesignoptimizer.sequencer tutorials/example_simulation_sequencer/coupled_transmon_chip list
```

Run the queue (Ansys), or do all of it in the GUI:

```bash
python -m qdesignoptimizer.sequencer tutorials/example_simulation_sequencer/coupled_transmon_chip run --yes
```

```bash
python -m qdesignoptimizer.sequencer.app tutorials/example_simulation_sequencer/coupled_transmon_chip
```

A completed study is in `out/completed/<id>_<start time>/`: the files it
ran from (`inputs/` as queued, `resolved/` as run), `result.json`,
`study.log`, `convergence.png`, a picture of what it simulated, and QDO's
own `.npy` and `_design_variables.json`.

Before the first Ansys run:

- `Q1.resonator_kappa` (capacitance) has been run in Ansys 2022 R2
  from this folder; on a new machine run that cheap study first.
- One Ansys at a time: the runner takes a lock in your user folder, so
  two runners of yours never solve at once; a notebook driving Ansys, or
  another user's runner, is not stopped by it.
- A `mode_identity` check on a study of this project fails on purpose:
  its `solve` does not check the modes (`modes_checked` says so).
- Ansys refuses a project path longer than 259 characters; queueing
  warns when a study's folder gets close. Keep the repository near the
  root of a drive.

### Working next to an agent

The GUI and the command line work on the same files, so you can watch
and steer in the GUI while an agent (Claude) adds studies, reorders the
queue and edits studies from a terminal. The GUI shows what it does
within a poll (2 s); a runner already going picks up new studies before
its next one.

- Add: `add <study>` / `sequence <name>` (straight into the queue; with
  `--hold` on hold).
- Reorder: `move <id> --first`, `--last`, `--to 3`, `--before <id>`,
  `--after <id>`. Every order write, from the GUI or the command line,
  is made under one machine-wide lock, so two moves never undo each
  other.
- Edit: `write <id> <file> --by claude` replaces one file of a queued or
  held study, read from `--from <path>` or standard input, and says
  whether the study can still run. While it writes, and while you are in
  Edit mode on that study in the GUI, the study holds a lease
  (`.editing.json` in its folder, renewed every poll, gone 15 s after
  the GUI closes): the other side is refused, and the GUI tags the study
  "being edited by claude". A GUI save also checks that nothing changed
  the study since you pressed Edit, and refuses if something did.
- The runner passes over a study being edited and runs the next one,
  coming back to it once the edit is done -- unless the study is
  *needed by the next study* (`needed_by_next: true` in its `job.yaml`,
  the box of that name in the GUI): then it waits for the edit, running
  nothing past it. A study sequence marks this itself on every study
  that writes back and has studies after it (an entry can say
  `needed_by_next: false`).
- Read the results in `out/completed/<id>_<start>/` and the store's
  `design_variables.meta.json`.

With several projects open in the GUI, give the agent the same set
(`<project> -p <other> ...`), so its studies join the shared order.

## Where things are kept

- The queue, the studies on hold and the results: `<project>/out/` (`queue/`, `hold/`, `completed/`; git-ignored).
- The store: `<project>/design_variables.json`, and its metadata
  `design_variables.meta.json` (which study wrote each value, and when).
  Commit both in a real project; this tutorial ignores them, as its
  store is a generated copy.
- Campaigns: `<project>/campaigns/`, each with its README and, if it
  has one, its own store and metadata. Commit them: they are the lab log.
- Machine-wide state (the runner lock, a multi-project queue order):
  `%LOCALAPPDATA%\qdo_sequencer\`, or the folder `QDO_SEQUENCER_HOME`
  names. Safe to delete while no runner is going.
