# qdesignoptimizer.sequencer -- handover

Where the port of the simulation sequencer into QDO stands, for whoever
picks it up next (person or Claude session). Last updated 2026-11-14.
Delete this file before the PR is merged, or move what is still useful
into the docs.

## Where it lives

- Branch `claude/simulation-framework-b9c89e`, pushed to
  `202Q-lab/QDesignOptimizer`. One commit on `main`: `ef15d73`, plus the
  commit adding this file.
- PR: https://github.com/202Q-lab/QDesignOptimizer/pull/56 -- open, not
  merged, no reviews. **CI fails** (see Open items, 1).
- Origin of the code: the `qdo_sequencer/` package, commit `f3dd9de5e`
  on branch `qdo_sequencer_redesign` of the local qiskit-metal clone
  (worktree `C:\Users\marer\code\qiskit-metal\.claude\worktrees\strange-goldberg-d977e8`,
  not pushed). That copy is now superseded by this one; it is still on
  qiskit-metal's `pyproject.toml` package list (`qdo_sequencer*`).

## What was done

- **Package** `src/qdesignoptimizer/sequencer/` (engine, backends, Dash
  app), renamed from `qdo_sequencer`. CLI `python -m
  qdesignoptimizer.sequencer`, GUI `python -m qdesignoptimizer.sequencer.app`.
  The user-state folder keeps its old name: `%LOCALAPPDATA%\qdo_sequencer\`
  (`QDO_SEQUENCER_HOME` overrides it).
- **Tests** `tests/test_sequencer/` (238 tests in the repo pass locally).
- **Tutorial** `tutorials/example_simulation_sequencer/` (start with its
  README): `demo/` runs without Ansys; `coupled_transmon_chip/` sets up the
  chip of `examples_coupled_transmon_chip/` (design reused unchanged) with
  templates `Q1/Q2.resonator_kappa`, `Q1/Q2.qubit_charge_line`,
  `Q1.qubit_resonator` and a measurement plan `tune_Q1`.
- **Dependencies** added: `dash>=2.16`, `psutil` (`poetry.lock` regenerated
  with Poetry 2.0.1; only additions).
- **Three independent reviews**, all findings fixed with tests:
  - study files may name only QDO's study/target classes, opt-target
    helpers and naming functions, by top-level name (a `Class.method`
    reference could start any program);
  - a target is switched on only when its study measures its parameter:
    eigenmode frequency; eigenmode kappa only with ports; nonlinearities
    only with a non-linear junction (EPR); a chained decay study's own
    parameter (wins over the eigenmode value, as in QDO); capacitance
    targets by Ansys conductor name; never two on one design variable;
  - the deep check refuses: a target with no target value, an eigenmode
    mode with no target frequency, a design-variable override in another
    unit, a switched-on target the study does not measure, two targets on
    one variable or parameter;
  - `study.log` survives QDO's logger import; template `=v[...]` sees
    values without units; YAML 1.1 exponent numbers reach QDO as numbers;
    measurement plan order written before publishing; `runner.json` waits 30 s.

## Open items, in order

1. **CI fails on PR #56; `main` passes.** Both the `Lint-Test-Mypy` job
   (step "Analyze the code with pylint") and the `Docs` job (step
   "Compile documentation") exit 1. Neither reproduces locally:
   - pylint 3.3.4 / astroid 3.3.8 under Python 3.12 (`qdesignenv-new`)
     reports no E/F messages except "Unable to import 'dash'" (Dash is not
     installed there);
   - a Sphinx 7.3.7 build of a copy of the branch (notebooks excluded, no
     Pandoc here) succeeds with warnings only.

   So the cause is in CI's environment. Read the logs (needs a GitHub
   login): https://github.com/202Q-lab/QDesignOptimizer/actions/runs/43312013612
   (lint) and .../runs/43312013597 (docs). Suspects: Dash 4.5.0 from the
   lock (local runs used 4.4.1) giving pylint errors in `app/`; the
   environment having drifted since main's last CI run (2026-10-07).
2. **Nothing from `coupled_transmon_chip` has run in Ansys.** First run:
   one cheap capacitance study, `Q1.resonator_kappa`; then
   `Q1.qubit_resonator`. One Ansys at a time; never close Ansys with
   `taskkill` (that is why the adapter's `before_step` does nothing).
3. **GUI not clicked through** in this repo.
4. **Local Python is 3.10** (`qdesignenv`); the project requires >=3.11.
   The test suite ran on 3.10 only.
5. **`gh` is not logged in** on this machine (`gh auth login`), so PRs and
   CI logs are not reachable from a session.

## Decisions still open (owner's call)

- **Reusable `solve`**: a base class `DesignAnalysisProject` in
  `backends/qdo_project.py` doing QDO's loop (build chip, DesignAnalysis,
  `optimize_target` x iterations), so a project's adapter is paths,
  `subjects`, `render` and a meshing map (~20 lines instead of ~190).
  Sketched, not written. Sub-questions: a `stop_when_converged` setting
  (stop iterating once all targets are within tolerance; proposed off by
  default); an `after_iteration` hook for mode checks; no Ansys-drop
  recovery in a first version. Alternative discussed earlier: no adapter,
  a `design.py` convention.

## Decided

- The write-back record is `design_variables.meta.json` (was
  `.writes.json`) and is committed next to the store in a real project.
  The tutorial git-ignores both, since its store is a generated copy.
- After a crash, a restarted runner obeys the cut-off study's
  `on_failure`: under `abort` it closes the study as failed and stops
  (status `aborted`); started again, it continues the queue.
- Possible later: a `needs:` field on the dependent study (skip it when
  what it needs failed, let independent studies run). Not planned.
- What used to be called a *campaign* (a named sequence of studies,
  queued in one go) is now a **measurement plan**: a reusable template,
  `<project>/measurement_plans/*.yaml`, CLI `plan <name>`, queued study
  ids `p<NN>-...`, `job.yaml` key `plan:`, adapter hook
  `measurement_plans(tasks)`. The queue stays the ground truth: a
  launched plan is copied into it and edited there.
- The word **campaign** is kept free for a new concept, not built yet:
  a lab log answering one question over a few days (tune up the chip, a
  Purcell filter, mesh sensitivity), in `<project>/campaigns/<name>/`,
  with its own copy of the design variables (copied back to the chip on
  demand). It can run several measurement plans, and one plan can serve
  several campaigns. Results stay in the chip's `out/`, tagged with the
  campaign. A conclusion may later become a rule or a skill.

## Known limits, left as they are

- Projects open in one process share their `code_modules` allowance
  (`qdo_files.ALLOWED_MODULES` is global); both are the user's own code.
- The deviation line reads a `_freq` deviation from a project's
  `converged` hook as Hz (the hook's documented contract).
- The measurement plan order can still lose to an `order.json` rewrite by
  another process at the same instant (`ORDER_LOCK` is a thread lock).
- Template `=...` expressions drop units without converting: keep the
  variables one expression combines in one unit.
- Capacitance targets are matched to studies by qiskit-metal's conductor
  naming (`<part>_<component>`); without the built design a component
  whose name ends another's can be mistaken for it (QDO then only warns).

## Physics notes from the session (readout kappa)

Kappa sets how the readout resonator feeds noise to the qubit, so it can
be misassigned: Purcell T1 ~ kappa (g/Delta)^2 (tutorial targets: a few
tens of us without a Purcell filter); thermal-photon dephasing
~ n_th kappa (2chi)^2 / (kappa^2 + (2chi)^2), largest at kappa = 2chi
(also the readout-SNR optimum); photon noise has a bandwidth ~kappa, so a
small kappa makes it look like 1/f noise in echo/CPMG. In QDO the
eigenmode kappa (f/Q) counts every port and loss in the model, the
capacitance decay study only the named waveguide: a disagreement can be
real leakage. Check whether QDO's `nonlin` chi is the full or the half
dispersive shift (pyEPR's is believed to be the full one) before using
these numbers.

## Running it

From the repo root, QDO installed (or `src` on `PYTHONPATH`):

```bash
python -m pytest tests/test_sequencer -q -p no:cacheprovider
python -m qdesignoptimizer.sequencer tutorials/example_simulation_sequencer/coupled_transmon_chip check-templates
python -m qdesignoptimizer.sequencer.app tutorials/example_simulation_sequencer/demo
```

Formatting is the repo's pre-commit (black, isort, pycln, pyupgrade); CI
also runs `pylint src` (E/F fail) and `mypy src`.
