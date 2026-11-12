"""Run a queue of simulation studies (or measurements) one at a time.

Ported from the qiskit-metal repo's ``qdo_sequencer`` package. The engine
imports nothing of QDO, qiskit-metal or a design project; the QDO backend
imports them only inside functions. A project plugs in through one ``sequencer.py`` (its
adapter, how it solves) and its templates, which say what can be
queued: ``ministudies/`` (one MiniStudy per file, and nothing else),
``opt_targets/`` (the project's opt targets, each with a name unique in
the project) and ``campaigns/``; the adapter builds a backend.

A study is not a template. It exists only as a folder of editable files
under the project's ``out/queue/``: a MiniStudy, the opt targets whose
modes are its modes, and ``job.yaml``, its queue entry. When it starts,
the folder moves to ``out/completed/<id>_<start>[_failed]/`` and keeps
what ran (a name sort groups a campaign's studies: ids are
``c<NN>-<campaign>-<study>``).
A campaign is a named sequence of studies that enqueues one folder per
entry.

Layers, lowest first (``tests/test_architecture.py`` holds them):

* ``engine``   -- job.yaml, the folder queue, campaigns, the runner,
  state files, the CLI. Pure Python + PyYAML.
* ``backends`` -- one per kind of work. ``qdo`` drives QDO/Ansys through
  a project's hooks and writes a MiniStudy and its OptTargets as YAML
  (``qdo_files``); ``fake`` serves the tests and a demo. Heavy imports
  stay inside functions.
* ``app``      -- the Dash queue GUI, the ONLY package that imports Dash.

Run::

    python -m qdesignoptimizer.sequencer <sequencer.py> add ATS1.kappa
    python -m qdesignoptimizer.sequencer <sequencer.py> campaign revalidate
    python -m qdesignoptimizer.sequencer <sequencer.py> check
    python -m qdesignoptimizer.sequencer <sequencer.py> run --yes
    python -m qdesignoptimizer.sequencer.app <sequencer.py>
"""

__version__ = "0.4.0"
