"""Demo project chip_qdo: QDO studies, a 5 s dummy in place of Ansys.

The adapter a real design has, with ``solve`` handed to
``dummy_solver`` instead of QDO + Ansys. Everything else is the real
QDO path: the MiniStudy templates in ``ministudies/`` and the project's
opt targets in ``opt_targets/`` filled in to a study's files, queued as
forms, judged by each target's ``tolerance``, the write-back into
``design_variables.json``. Measurement plan files go in ``measurement_plans/``.

The design folder is not a package: its modules import each other by
bare name, so it goes on the import path here, as in every design.
``design_variables.json`` (the store, git-ignored) starts as a copy of
``design_variables.start.json``; delete it to start over.
"""

import importlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dummy_solver  # noqa: E402

from qdesignoptimizer.sequencer.backends.qdo import QDOBackend  # noqa: E402
from qdesignoptimizer.sequencer.engine.backend import (  # noqa: E402
    PlanEntry,
    PlanSpec,
)


class DemoQDOProject:
    """The QDOProject protocol (see ``backends.qdo``), solved by a dummy."""

    root = ROOT
    design_variables_file = ROOT / "design_variables.json"
    #: The subsystems: ``{subject}`` in a template is filled in with
    #: each key, so every template here makes one study per resonator.
    subjects = {
        "R1": ("Resonator R1", "chip_qdo"),
        "R2": ("Resonator R2", "chip_qdo"),
    }

    def __init__(self):
        if not self.design_variables_file.exists():
            shutil.copy(
                ROOT / "design_variables.start.json",
                self.design_variables_file,
            )

    def settings_keys(self) -> set:
        return {"iterations", "seconds"}

    def default_settings(self, task) -> dict:
        return {"iterations": 3, "seconds": 5}

    def parameter_targets(self) -> dict:
        import parameter_targets

        importlib.reload(parameter_targets)  # edits count without a restart
        return dict(parameter_targets.PARAM_TARGETS)

    def before_step(self) -> None:
        """Nothing to reset: there is no Ansys."""

    def solve(self, study, targets, folder, settings, variables, params):
        return dummy_solver.solve(study, targets, folder, settings, variables, params)

    def measurement_plans(self, tasks) -> list:
        """``tune_R<n>``: frequency, then linewidth, each written back."""
        out = []
        for name in self.subjects:
            out.append(
                PlanSpec(
                    f"tune_{name}",
                    f"Tune {name}: frequency, then linewidth",
                    tuple(
                        PlanEntry(f"{name}.{kind}", save=True, checks=("converged",))
                        for kind in ("frequency", "linewidth")
                    ),
                    why_ordered="the linewidth depends on the frequency "
                    "the first study wrote back",
                )
            )
        return out


def make_backend():
    """The factory qdesignoptimizer.sequencer calls."""
    return QDOBackend(DemoQDOProject())
