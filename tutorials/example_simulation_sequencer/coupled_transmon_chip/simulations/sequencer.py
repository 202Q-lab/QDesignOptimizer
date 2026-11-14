"""Adapter: the coupled-transmon tutorial chip, queued and solved in Ansys.

The chip itself is the one of ``tutorials/examples_coupled_transmon_chip``
(``design.py``, ``names.py``): this project reuses it and adds only what
the sequencer needs to queue and run studies of it:

* ``ministudies/*.yaml``   the MiniStudies that can be queued;
* ``opt_targets/*.yaml``   the project's OptTargets, each with a name;
* ``measurement_plans/*.yaml``     named sequences of studies;
* ``parameter_targets.py`` the target values (PARAM_TARGETS);
* this file                HOW a study is solved: build the chip with the
                           study's design variables, hand QDO's
                           DesignAnalysis the MiniStudy and OptTargets,
                           iterate.

``design_variables.json`` next to this folder is the project's store: the
studies read their design variables from it and write the tuned ones
back. It starts as a copy of the tutorial chip's file; delete it to start
over.

A design folder is not a package: its modules import each other by bare
name, so this folder and the tutorial chip's go on the import path here.
"""

import importlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHIP = ROOT.parents[1] / "examples_coupled_transmon_chip"
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


class CoupledTransmonProject:
    """The ``QDOProject`` protocol (see ``qdesignoptimizer.sequencer.backends.qdo``)."""

    root = ROOT
    design_variables_file = ROOT / "design_variables.json"

    #: The subsystems. ``{n}`` in a template is filled in with each one's
    #: value, so a template using it makes one study per subsystem:
    #: ``ministudies/resonator_kappa.yaml`` -> ``Q1.resonator_kappa``,
    #: ``Q2.resonator_kappa``. A template without placeholders named
    #: ``Q1_<rest>.yaml`` is the study ``Q1.<rest>``.
    subjects = {
        "Q1": ("Qubit 1, resonator 1, charge line 1", "group 1", {"n": 1}),
        "Q2": ("Qubit 2, resonator 2, charge line 2", "group 2", {"n": 2}),
    }

    #: Ansys refuses a project path longer than this; queueing warns when
    #: a study's folder would get close.
    path_limit = 259

    def __init__(self):
        if not self.design_variables_file.exists():
            shutil.copy(CHIP / "design_variables.json", self.design_variables_file)

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
        """What ``design()`` is built from: a change rebuilds it (after
        the GUI's Reload for an edit to the Python files)."""
        return [
            CHIP / "design.py",
            CHIP / "names.py",
            self.design_variables_file,
        ]

    def design(self, variables=None):
        """The chip, built with ``variables`` (the store when None).

        The sequencer uses it to check the component names a study file
        gives and to draw a picture of each study before it runs.
        """
        import json

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
        machine (``taskkill``), which would take down another user's or
        session's study; close Ansys yourself when you need a clean start.
        """

    def solve(self, study, targets, folder, settings, variables, params):
        """Run QDO's optimization loop for one study, outputs in ``folder``.

        ``study`` and ``targets`` are the MiniStudy and OptTargets built
        from the study's files; the solver settings (passes, ``delta_f``)
        are the MiniStudy's. Returns the DesignAnalysis: the sequencer reads
        each iteration's results and the proposed design variables from it.
        """
        import design as d
        import names as n
        from qiskit_metal.qlibrary.couplers.coupled_line_tee import (
            CoupledLineTee,
        )

        from qdesignoptimizer.design_analysis import (
            DesignAnalysis,
            DesignAnalysisState,
        )
        from qdesignoptimizer.design_analysis_types import MeshingMap
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
            # The sequencer draws convergence.png itself; no plot windows.
            plot_settings=None,
            meshing_map=[
                MeshingMap(
                    component_class=CoupledLineTee,
                    mesh_names=d.CoupledLineTee_mesh_names,
                )
            ],
        )
        for _ in range(int(settings.get("iterations", 3))):
            analysis.optimize_target({}, {})
        return analysis

    def modes_checked(self, settings) -> bool:
        """``solve`` does not check that each iteration's modes are the
        intended ones, so a ``mode_identity`` check on a study fails
        rather than passing unchecked."""
        return False

    def reload(self) -> None:
        """The GUI's Reload: pick up edits to the chip's modules."""
        for name in ("names", "design", "parameter_targets"):
            if name in sys.modules:
                importlib.reload(sys.modules[name])


def make_backend():
    """The factory the sequencer calls."""
    return QDOBackend(CoupledTransmonProject())
