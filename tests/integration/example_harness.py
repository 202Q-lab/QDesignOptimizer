"""Run the tutorial notebooks end-to-end against faked simulators.

The notebook code cells are executed unmodified (IPython magics are dropped), so
the tests always follow what the examples actually do. The qiskit-metal design is
built for real; only the GUI and the Ansys/pyEPR/Q3D boundary are replaced (see
``fake_ansys.py``).

Snapshots
---------
Each run produces a JSON snapshot containing

* ``simulator_calls``: every call into HFSS/pyEPR/Q3D with its settings,
* ``optimization_results``: per iteration the design variables, optimized
  parameters, eigenmode results, cross-Kerr/participation matrices, capacitance
  matrices and which targets the optimizer grouped together, and
* ``surface_participation_ratios`` when the example computes them,

which is compared against ``snapshots/<name>.json`` (floats with a relative
tolerance). When a change *intentionally* alters what is sent to the simulators,
regenerate the snapshots and review the diff::

    QDO_UPDATE_SNAPSHOTS=1 pytest tests/integration
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional
from unittest.mock import MagicMock

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import qiskit_metal  # noqa: E402

import qdesignoptimizer.design_analysis as design_analysis_module  # noqa: E402
import qdesignoptimizer.sim_capacitance_matrix as sim_capacitance_matrix_module  # noqa: E402
import qdesignoptimizer.utils.utils as utils_module  # noqa: E402

from .fake_ansys import (  # noqa: E402
    SimulationSession,
    parse_value_with_unit,
    to_jsonable,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TUTORIALS = REPO_ROOT / "tutorials"
SNAPSHOT_DIR = Path(__file__).resolve().parent / "snapshots"
UPDATE_SNAPSHOTS = os.environ.get("QDO_UPDATE_SNAPSHOTS", "") not in ("", "0")

# Local modules imported by the notebooks with plain ``import names as n`` etc.
# They exist in every example folder, so they must be purged between examples.
TUTORIAL_LOCAL_MODULES = (
    "names",
    "design",
    "mini_studies",
    "optimization_targets",
    "parameter_targets",
    "plot_settings",
)

FLOAT_REL_TOL = 1e-6
FLOAT_ABS_TOL = 1e-12


def notebook_code_cells(notebook: Path) -> list[str]:
    """Return the code cells of a notebook with IPython magics removed."""
    content = json.loads(notebook.read_text(encoding="utf8"))
    cells = []
    for cell in content["cells"]:
        if cell["cell_type"] != "code":
            continue
        lines = [
            line
            for line in "".join(cell["source"]).splitlines()
            if not line.lstrip().startswith(("%", "!"))
        ]
        cells.append("\n".join(lines))
    return cells


def run_example_notebook(
    notebook: Path,
    workdir: Path,
    monkeypatch,
    skip_cells_containing: Iterable[str] = (),
):
    """Execute a tutorial notebook with faked simulators.

    Args:
        notebook: path to the ``.ipynb`` file inside ``tutorials/``.
        workdir: empty directory used as working directory (outputs are written here).
        monkeypatch: the pytest ``monkeypatch`` fixture.
        skip_cells_containing: cells containing any of these strings are not run.

    Returns:
        (session, namespace): the recording session and the notebook's globals.
    """
    session = prepare_example(notebook.parent, workdir, monkeypatch)

    namespace: dict = {"__name__": "__main__"}
    try:
        for idx, source in enumerate(notebook_code_cells(notebook)):
            if any(marker in source for marker in skip_cells_containing):
                continue
            code = compile(source, f"{notebook.name} [code cell {idx}]", "exec")
            exec(code, namespace)  # pylint: disable=exec-used
    finally:
        plt.close("all")
        for module in TUTORIAL_LOCAL_MODULES:
            sys.modules.pop(module, None)
    return session, namespace


def prepare_example(example_dir: Path, workdir: Path, monkeypatch) -> SimulationSession:
    """Make an example folder importable from ``workdir`` and install the fakes.

    Afterwards the example's local modules (``names``, ``design``, ``mini_studies``,
    ...) can be imported. Their ``sys.modules`` entries are removed again by
    ``monkeypatch`` at teardown.
    """
    shutil.copy(
        example_dir / "design_variables.json", workdir / "design_variables.json"
    )
    monkeypatch.chdir(workdir)
    monkeypatch.syspath_prepend(str(example_dir))
    for module in TUTORIAL_LOCAL_MODULES:
        monkeypatch.delitem(sys.modules, module, raising=False)
        # register for removal at teardown even if never imported
        monkeypatch.setitem(sys.modules, module, None)
        monkeypatch.delitem(sys.modules, module)

    session = SimulationSession()
    _install_fakes(session, monkeypatch)
    return session


def _install_fakes(session: SimulationSession, monkeypatch):
    monkeypatch.setattr(
        design_analysis_module, "EPRanalysis", session.epranalysis_factory
    )
    monkeypatch.setattr(design_analysis_module, "epr", session.epr_module())
    monkeypatch.setattr(
        sim_capacitance_matrix_module, "LOManalysis", session.lomanalysis_factory
    )
    # create_chip_base imports MetalGUI from qiskit_metal when open_gui=True. Patch the
    # module dict directly: reading the attribute would trigger quantum-metal's lazy
    # GUI import and with it Qt, which the tests must not depend on.
    monkeypatch.setitem(vars(qiskit_metal), "MetalGUI", MagicMock(name="MetalGUI"))
    monkeypatch.setattr(utils_module, "close_ansys", lambda: None)

    original_init = design_analysis_module.DesignAnalysis.__init__

    def recording_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        session.register_analysis(self)
        render = self.render_qiskit_metal

        def logged_render(design, **render_kwargs):
            session.log("render_qiskit_metal", kwargs=render_kwargs)
            return render(design, **render_kwargs)

        self.render_qiskit_metal = logged_render

    monkeypatch.setattr(
        design_analysis_module.DesignAnalysis, "__init__", recording_init
    )


# ----------------------------------------------------------------------- snapshots
def build_snapshot(
    session: SimulationSession, optimization_results: list, analysis=None
) -> dict:
    snapshot = {
        "simulator_calls": session.events,
        "optimization_results": [
            _snapshot_iteration(result) for result in optimization_results
        ],
    }
    surface_p_ratio = getattr(analysis, "surface_p_ratio", None)
    if surface_p_ratio is not None:
        snapshot["surface_participation_ratios"] = to_jsonable(surface_p_ratio)
    return snapshot


def _snapshot_iteration(result: dict) -> dict:
    iteration = {
        "design_variables": to_jsonable(result["design_variables"]),
        "system_optimized_params": to_jsonable(result["system_optimized_params"]),
        "minimization_groups": [
            group["targets_to_minimize_for"]
            for group in result.get("minimization_results", [])
        ],
    }
    if "eig_results" in result:
        iteration["eig_results"] = to_jsonable(result["eig_results"])
    if "cross_kerrs" in result:
        iteration["cross_kerrs"] = to_jsonable(result["cross_kerrs"])
    if "capacitance_matrix" in result:
        iteration["capacitance_matrix"] = to_jsonable(result["capacitance_matrix"])
    return iteration


def assert_matches_snapshot(name: str, actual: dict):
    """Compare ``actual`` with ``snapshots/<name>.json`` (or rewrite it when updating)."""
    actual = json.loads(
        json.dumps(actual)
    )  # normalise tuples/keys exactly like a reload
    path = SNAPSHOT_DIR / f"{name}.json"
    if UPDATE_SNAPSHOTS or not path.exists():
        SNAPSHOT_DIR.mkdir(exist_ok=True)
        path.write_text(json.dumps(actual, indent=1, sort_keys=True) + "\n", "utf8")
        if not UPDATE_SNAPSHOTS:
            raise AssertionError(
                f"Snapshot {path.name} did not exist and was created; review and commit it."
            )
        return

    expected = json.loads(path.read_text("utf8"))
    differences: list[str] = []
    _diff(expected, actual, "$", differences)
    if differences:
        # Into the system temp folder, not the repository: a test run
        # leaves nothing behind in the checkout.
        actual_path = Path(tempfile.gettempdir()) / f"qdo-snapshot-{name}.actual.json"
        actual_path.write_text(
            json.dumps(actual, indent=1, sort_keys=True) + "\n", "utf8"
        )
        shown = "\n  ".join(differences[:25])
        more = (
            f"\n  ... and {len(differences) - 25} more" if len(differences) > 25 else ""
        )
        raise AssertionError(
            f"Example run differs from snapshot {path.name} "
            f"({len(differences)} differences):\n  {shown}{more}\n"
            f"Full actual output written to {actual_path}. If the change is intended, "
            "regenerate with QDO_UPDATE_SNAPSHOTS=1."
        )


def _diff(expected, actual, path: str, out: list[str]):
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            if key not in actual:
                out.append(f"{path}.{key}: missing (expected {_short(expected[key])})")
            elif key not in expected:
                out.append(f"{path}.{key}: unexpected {_short(actual[key])}")
            else:
                _diff(expected[key], actual[key], f"{path}.{key}", out)
        return
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            out.append(f"{path}: length {len(actual)} != expected {len(expected)}")
        for idx, (exp, act) in enumerate(zip(expected, actual)):
            _diff(exp, act, f"{path}[{idx}]", out)
        return
    if not _scalars_equal(expected, actual):
        out.append(f"{path}: {_short(actual)} != expected {_short(expected)}")


def _scalars_equal(expected, actual) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected is actual
    if expected == actual:
        return True
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(
            expected, actual, rel_tol=FLOAT_REL_TOL, abs_tol=FLOAT_ABS_TOL
        )
    if isinstance(expected, str) and isinstance(actual, str):
        # Design variables such as "401.2345 um" are compared numerically.
        parsed_expected = parse_value_with_unit(expected)
        parsed_actual = parse_value_with_unit(actual)
        if parsed_expected is None or parsed_actual is None:
            return False
        return parsed_expected[1] == parsed_actual[1] and math.isclose(
            parsed_expected[0],
            parsed_actual[0],
            rel_tol=FLOAT_REL_TOL,
            abs_tol=FLOAT_ABS_TOL,
        )
    return False


def _short(value) -> str:
    text = json.dumps(value)
    return text if len(text) < 120 else text[:117] + "..."


# ---------------------------------------------------------------------- helpers
def events(session: SimulationSession, name: str) -> list[dict]:
    """Payloads of all recorded events with the given name, in call order."""
    return [payload for event, payload in session.events if event == name]


def relative_errors_to_targets(analysis, optimization_results=None) -> dict:
    """Relative deviation from target of every optimization target in the last iteration."""
    from qdesignoptimizer.anmod_optimizer import ANModOptimizer

    final = (optimization_results or analysis.optimization_results)[-1][
        "system_optimized_params"
    ]
    errors = {}
    for target in analysis.opt_targets:
        current = ANModOptimizer._get_parameter_value(target, final)
        wanted = ANModOptimizer._get_parameter_value(
            target, analysis.system_target_params
        )
        errors[f"{target.target_param_type} {target.involved_modes}"] = abs(
            current / wanted - 1
        )
    return errors
