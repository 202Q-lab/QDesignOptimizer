"""Adapter: the tunable-coupler tutorial chip, tuned partition by partition.

The chip is the one of ``tutorials/examples_coupled_transmons_tunable_coupler_chip``
(``design.py``, ``names.py``): two transmons, each with a quarter-wave
readout resonator, and a tunable coupler between them -- five modes. Its
notebook ``main_eigenmode_two_qubit_resonator_partitioned.ipynb`` splits
the chip into three smaller eigenmode solves; this project does the same
through the sequencer (``../tunable_coupler_chip`` solves all five modes
at once):

* ``ministudies/Q1_readout.yaml``        qubit 1, resonator 1, the coupler
                                         (the notebook's ``res_qb_cplr``);
* ``ministudies/Q2_readout.yaml``        qubit 2, resonator 2, the coupler
                                         (``cplr_qb_res``);
* ``ministudies/C_coupler.yaml``         both qubits and the coupler
                                         (``qb_cplr_qb``);
* ``opt_targets/partitions.yaml``        the notebook's OptTargets, each
                                         taken by the one partition that
                                         tunes it (its ``include_param_in_update``);
* ``study_sequences/tune_partitioned.yaml``  three rounds of the three;
* ``parameter_targets.py``               the notebook's PARAM_TARGETS;
* this file                              how a study is solved.

``design_variables.json`` is the project's store. It starts as a copy of
the tutorial chip's file, the values the notebook starts from; delete it
to start over.

A design folder is not a package: its modules import each other by bare
name, so this folder and the tutorial chip's go on the import path here.
"""

import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHIP = ROOT.parents[1] / "examples_coupled_transmons_tunable_coupler_chip"
for _folder in (CHIP, ROOT):  # ROOT first: its parameter_targets wins
    if str(_folder) in sys.path:
        sys.path.remove(str(_folder))
    sys.path.insert(0, str(_folder))

from qdesignoptimizer.sequencer.backends.qdo import QDOBackend  # noqa: E402


class _NoGui:
    """What ``design.render_qiskit_metal_design`` asks of the Metal GUI,
    without opening one: a study runs in a worker process."""

    def __init__(self, design):
        self.design = design

    def rebuild(self):
        self.design.rebuild()

    def autoscale(self):
        pass


class PartitionedProject:
    """The ``QDOProject`` protocol (see ``qdesignoptimizer.sequencer.backends.qdo``)."""

    root = ROOT
    design_variables_file = ROOT / "design_variables.json"

    #: The partitions. ``<subsystem>_<rest>.yaml`` is the study
    #: ``<subsystem>.<rest>``, and takes the opt targets whose
    #: ``{placeholders}`` its subsystem's values fill in: ``{n}`` for a
    #: qubit and its resonator, ``{coupler}`` for the coupler. That is
    #: what keeps a partition to its own targets.
    subjects = {
        "Q1": ("Qubit 1 and its resonator, with the coupler", "res_qb_cplr", {"n": 1}),
        "Q2": ("Qubit 2 and its resonator, with the coupler", "cplr_qb_res", {"n": 2}),
        "C": (
            "The coupler, with both qubits",
            "qb_cplr_qb",
            {"coupler": "coupler_1to2"},
        ),
    }

    #: Ansys refuses a project path longer than this; queueing warns when
    #: a study's folder would get close.
    path_limit = 259

    def __init__(self):
        if not self.design_variables_file.exists():
            self.design_variables_file.write_text(
                json.dumps(self.initial_design_variables(), indent=4) + "\n",
                encoding="utf-8",
            )

    def initial_design_variables(self) -> dict:
        """The store a new project starts with: the tutorial chip's
        values, as the notebook starts from them."""
        return json.loads((CHIP / "design_variables.json").read_text("utf-8"))

    # -- what a study's job.yaml may set -------------------------------------

    def settings_keys(self) -> set:
        return {"iterations"}

    def default_settings(self, task) -> dict:
        return {"iterations": 3}

    # -- the project's values ------------------------------------------------

    def parameter_targets(self) -> dict:
        import parameter_targets

        importlib.reload(parameter_targets)  # edits count without a restart
        return dict(parameter_targets.PARAM_TARGETS)

    def design_files(self) -> list:
        """What ``design()`` is built from: a change rebuilds it."""
        return [CHIP / "design.py", CHIP / "names.py", self.design_variables_file]

    def code_files(self) -> list:
        """The chip's code: a change reloads it before the next study."""
        return [
            CHIP / "design.py",
            CHIP / "names.py",
            ROOT / "parameter_targets.py",
            Path(__file__),
        ]

    def design(self, variables=None):
        """The chip, built with ``variables`` (the store when None)."""
        import design as d
        import names as n

        from qdesignoptimizer.utils.chip_generation import create_chip_base
        from qdesignoptimizer.utils.names_design_variables import (
            add_design_variables_to_design,
        )

        if variables is None:
            variables = json.loads(
                self.design_variables_file.read_text(encoding="utf-8")
            )
        chip, _ = create_chip_base(
            n.CHIP_NAME, d.chip_type, open_gui=False, design_variables_file=None
        )
        add_design_variables_to_design(chip, dict(variables))
        d.render_qiskit_metal_design(chip, _NoGui(chip))
        return chip

    # -- solving -------------------------------------------------------------

    def before_step(self) -> None:
        """Nothing to reset. QDO's ``close_ansys`` kills every Ansys on the
        machine (``taskkill``), which would take down another session's
        study; close Ansys yourself when you need a clean start."""

    def solve(self, study, targets, folder, settings, variables, params):
        """Run QDO's optimization loop for one study, outputs in ``folder``:
        the notebook's loop, with the passes and ``delta_f`` the
        MiniStudy's. Returns the DesignAnalysis."""
        import design as d
        import names as n

        from qdesignoptimizer.design_analysis import (
            DesignAnalysis,
            DesignAnalysisState,
        )
        from qdesignoptimizer.utils.utils import get_save_path

        chip = self.design(variables)
        gui = _NoGui(chip)

        def render(design, **kwargs):
            d.render_qiskit_metal_design(design, gui, **kwargs)

        analysis = DesignAnalysis(
            DesignAnalysisState(chip, render, dict(params)),
            mini_study=study,
            opt_targets=list(targets),
            save_path=get_save_path(str(Path(folder)) + "/", n.CHIP_NAME),
            # The sequencer writes the tuned values back, after its checks.
            update_design_variables=False,
            plot_settings=None,
        )
        for _ in range(int(settings.get("iterations", 3))):
            analysis.optimize_target({}, {})
        return analysis

    def modes_checked(self, settings) -> bool:
        """``solve`` does not check each iteration's mode order."""
        return False

    def reload(self) -> None:
        """The GUI's Reload: pick up edits to the chip's modules."""
        for name in ("names", "design", "parameter_targets"):
            if name in sys.modules:
                importlib.reload(sys.modules[name])


def make_backend():
    """The factory the sequencer calls."""
    return QDOBackend(PartitionedProject())
