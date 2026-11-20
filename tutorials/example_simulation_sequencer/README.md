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
└── design_variables.json         the store (made on first use, git-ignored:
                                  the chip's values, junction inductances
                                  and resonator lengths estimated from
                                  the targets)
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
  targets) and `v` (the design variables). In a MiniStudy it stays in
  the queued file and is evaluated when the study starts, with the
  values the study runs with, so a target changed while it waits is the
  one used. The completed folder holds the value; `inputs/` the
  expression. Type a number over it to fix the value instead.
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

### What a study simulates

Open a queued study in the GUI:

- **Generate pre-figures** draws each solver run as it is saved: only
  that run's components, with the airbridges the solver adds.
- **Components and pins** lists every component of the built chip, with
  its class and pins. For each run of the study (the HFSS eigenmode
  model, then each chained capacitance study):
  - Tick the components the run includes. A filter box narrows the list
    by name or class.
  - Set each loose pin to **short**, **open** or **port**. Short is the
    default: qiskit-metal runs the trace into ground. Open cuts an
    endcap. Port is a lumped port, 50 Ohm unless you change it, and HFSS
    only. A pin joined to another included component is a connection
    inside the model, so it has no choice.
  - **Draw** shows the choice before it is saved, in two pictures. The
    first is the included components over the rest of the chip in grey,
    with every pin marked. The second is the whole chip, with every
    component named.
  - **Save to the study** writes `qiskit_component_names`, `open_pins`
    and `port_list` into the study's `ministudy.yaml`, then checks it as
    any edit is checked.

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

### Reports

**Report ...** under Completed (or `report` on the command line) makes
a report of chosen completed studies; by default the latest run of each
study, and you can tick any others to compare:

- **brief**: one line per study, how many targets it had and its
  worst miss, and whether it wrote its values back;
- **full**: per study what it was (task, campaign, sequence, settings,
  who queued it and why), every target per iteration, its design
  variables from first to final, and its checks;
- **a table for Notion**: one row per qubit, with its simulated
  frequency, anharmonicity, readout resonator, linewidth and dispersive
  shift, and its designed Lj and Ej. Copy it as Markdown (Notion makes a
  table of it) or tab-separated (a spreadsheet, a Notion database).

The table's last columns are for fabrication. Give each junction as it
was made, as Ej (`12.5 GHz`), its normal resistance Rn (`8.1 kOhm`,
converted by Ambegaokar-Baratoff with aluminium's gap; set `gap_ueV` to
change it) or Lj (`14 nH`). The table then estimates each qubit's
frequency with that junction, rescaling the simulated one:
`f01' = (f01 + α) · sqrt(Ej' / Ej) − α`. What you give is kept in the
project's `fabrication.yaml`, so every report after uses it:

    python -m qdesignoptimizer.sequencer <project> report --fab qubit_1=8.1kOhm --show table

Every report is also written to `out/reports/<when>/`: `report_brief.md`,
`report.md`, `table.md` and `table.tsv`.

### Working next to an agent

The GUI and the command line work on the same files, so you can watch
and steer in the GUI while an agent (Claude) adds studies, reorders the
queue and edits studies from a terminal. The GUI shows what it does
within a poll (2 s); a runner already going picks up new studies before
its next one.

The steps an agent should follow ship with the package: `python -m
qdesignoptimizer.sequencer guide` prints them. A chip project's own
`CLAUDE.md` points to it and adds that project's rules; this tutorial's
[`CLAUDE.md`](CLAUDE.md) is an example to copy.

- Add: `add <study>` / `sequence <name>` (straight into the queue; with
  `--hold` on hold). `--by claude` (or `QDO_SEQUENCER_AGENT=claude`) and
  `--why "..."` say who queued it and why: the GUI tags the study
  "by claude", with the reason on hover and in the study editor.
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
  "being edited by claude". Each browser tab and each command holds its
  own lease, so a second tab, or a second agent with the same name, is
  kept out too. Park, Remove and queueing a held study wait for someone
  else's edit as well. A GUI save also checks that nothing changed the
  study since you pressed Edit, and refuses if something did.
- **What a study needs.** A study that uses what another one writes
  back lists it: `needs: [<id>, ...]` in its `job.yaml` (the *needs*
  field in the GUI, `add <study> --needs <id>` on the command line). The
  runner starts the first study that may start; one whose needs have
  not all run successfully (a requeued copy, `<id>_r2`, counts) is
  passed over. A need that is queued again, or on hold, is waited for
  even if an earlier run of it succeeded: the study runs on its new
  result, not the old one. A study passed over stays where it is, and
  the studies after it move up. So is one being edited;
  it runs once the edit is done. With nothing left that may start but
  something being edited, the runner waits for it; with nothing at all,
  it finishes and says which study waits for what (a need on hold, or
  failed). The queue tags such a study "waits for ...".
- A study sequence fills `needs` in itself: each study needs the
  earlier ones of its sequence that write back. An entry can say
  otherwise by number: `needs: [1]`, or `needs: []` for none.
- **When the code changes.** The project names its code
  (`code_files()` in `sequencer.py`; by default the `.py` files among
  `design_files()` and `sequencer.py` itself). Before each study the
  runner reloads it if it changed, and every run records the version it
  used (`code` in `result.json`); the GUI reloads too, and clears
  pre-figures drawn with the old code. A study on its own just runs on
  the new code. A study sequence that ran in part on the old code waits
  -- the rest of the queue moves up -- until you choose, in the GUI
  (its studies are tagged "code changed: continue or restart?") or with
  `decide <sNN> continue|restart`: continue it on the new code, or
  restart it: the whole sequence is queued again as a new sequence
  (`s008`, same name, each study noting `restarted_from: s007`) and the
  rest of the old one goes on hold; its results stay.
- Wait: `wait <id> ...` returns once each has run and prints its
  status, folder, outputs and what it wrote back (exit 0 done, 1 failed
  or on hold, 2 unknown, 3 `--timeout`). An agent runs it in the
  background and is told when it exits.
- Read the results in `out/completed/<id>_<start>/` and the store's
  `design_variables.meta.json`.

With several projects open in the GUI, give the agent the same set
(`<project> -p <other> ...`), so its studies join the shared order.
Connecting another project keeps that order: the new project's studies
come after it.

## Where things are kept

- The queue, the studies on hold and the results: `<project>/out/` (`queue/`, `hold/`, `completed/`; git-ignored).
- The store: `<project>/design_variables.json`, and its metadata
  `design_variables.meta.json` (which study wrote each value, and when).
  Commit both in a real project; this tutorial ignores them, as its
  store is a generated copy.
- Campaigns: `<project>/campaigns/`, each with its README and, if it
  has one, its own store and metadata. Commit them: they are the lab log.
- Reports: `<project>/out/reports/`; what fabrication gave:
  `<project>/fabrication.yaml` (commit it).
- Machine-wide state (the runner lock, a multi-project queue order):
  `%LOCALAPPDATA%\qdo_sequencer\`, or the folder `QDO_SEQUENCER_HOME`
  names. Safe to delete while no runner is going.

## Known limits

- Edits are saved one file at a time. A change spread over two files
  (the MiniStudy, then the opt targets) can be cut by the runner taking
  the study in between; the GUI then says the second save came too
  late.
- Write-backs and `campaign promote` hold a machine-wide lock on the
  store, so they never drop each other's values. A hand edit of the
  store in a text editor is not covered, and a value a study tunes
  overwrites a hand edit of the same value.
- Template `=` expressions drop units without converting them: keep the
  variables that one expression combines in one unit.
- Capacitance targets are matched to studies by qiskit-metal's
  conductor names (`<part>_<component>`). Without the built design, a
  component whose name ends another's can be mistaken for it (QDO then
  only warns).
- The deviation shown for a `_freq` target, when a project's
  `converged` hook reports it, is read as Hz (the hook's contract).
- Projects open in one process share their `code_modules` allowance:
  a study file of one may name a module the other allows. Both are your
  own code.
- The tests run the whole queue against a fake Ansys. They show that
  studies are made, run, checked and written back correctly; they do
  not show that the physics of a target is right, since the fake
  answers with the same relations the targets assume.

## Open questions

- **A reusable `solve`.** A base class doing QDO's loop (build the chip,
  `DesignAnalysis`, `optimize_target` for each iteration), so that an
  adapter is paths, subsystems and a meshing map, about 20 lines
  instead of about 190. Open sub-questions: a `stop_when_converged`
  setting (stop once every target is within tolerance), an
  `after_iteration` hook for mode checks, and whether to recover from
  Ansys dropping out.

## Notes on readout linewidth (kappa)

Kappa sets how the readout resonator passes noise to the qubit, so a
wrong value has consequences:

- Purcell T1 goes as kappa (g/Delta)^2. The tutorial's targets give a
  few tens of microseconds without a Purcell filter.
- Dephasing from thermal photons goes as
  n_th kappa (2chi)^2 / (kappa^2 + (2chi)^2), largest at kappa = 2chi,
  which is also where readout SNR is best.
- Photon noise has a bandwidth of about kappa, so a small kappa can make
  it look like 1/f noise in echo or CPMG.

In QDO the eigenmode kappa (f/Q) counts every port and loss in the
model; the capacitance decay study counts only the named waveguide. A
disagreement between the two can be real leakage. Check whether QDO's
`nonlin` chi is the full or the half dispersive shift (pyEPR's is
believed to be the full one) before using these numbers.
