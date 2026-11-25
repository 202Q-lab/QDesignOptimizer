from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Tuple

import pytest

from .example_harness import TUTORIALS, run_example_notebook
from .fake_ansys import SimulationSession

COUPLED_TRANSMON = TUTORIALS / "examples_coupled_transmon_chip"
TUNABLE_COUPLER = TUTORIALS / "examples_coupled_transmons_tunable_coupler_chip"


def _results_of_design_analysis(namespace: dict) -> list:
    return namespace["design_analysis"].optimization_results


@dataclass(frozen=True)
class ExampleCase:
    """A tutorial notebook exercised by the integration tests."""

    name: str
    notebook: Path
    # Whether the notebook runs enough iterations to reach its targets.
    converges: bool = True
    # Cells containing any of these strings are not executed (document why!).
    skip_cells_containing: Tuple[str, ...] = ()
    get_optimization_results: Callable[[dict], list] = field(
        default=_results_of_design_analysis
    )


EXAMPLE_CASES = [
    ExampleCase(
        "coupled_transmon__capacitance_target_resonator",
        COUPLED_TRANSMON / "main_capacitance_target_resonator.ipynb",
    ),
    ExampleCase(
        "coupled_transmon__charge_line_decay_single_qubit",
        COUPLED_TRANSMON / "main_charge_line_decay_single_qubit.ipynb",
    ),
    ExampleCase(
        "coupled_transmon__eigenmode_single_qubit_resonator",
        COUPLED_TRANSMON / "main_eigenmode_single_qubit_resonator.ipynb",
    ),
    ExampleCase(
        "coupled_transmon__eigenmode_single_qubit_resonator_exponent_study",
        COUPLED_TRANSMON / "main_eigenmode_single_qubit_resonator_exponent_study.ipynb",
    ),
    ExampleCase(
        "coupled_transmon__eigenmode_single_qubit_resonator_pratio",
        COUPLED_TRANSMON / "main_eigenmode_single_qubit_resonator_pratio.ipynb",
        converges=False,  # runs a single iteration only
    ),
    ExampleCase(
        "coupled_transmon__eigenmode_two_qubit_resonator",
        COUPLED_TRANSMON / "main_eigenmode_two_qubit_resonator.ipynb",
    ),
    ExampleCase(
        "coupled_transmon__kappa_target_resonator",
        COUPLED_TRANSMON / "main_kappa_target_resonator.ipynb",
    ),
    ExampleCase(
        "tunable_coupler__eigenmode_two_qubit_resonator",
        TUNABLE_COUPLER / "main_eigenmode_two_qubit_resonator.ipynb",
    ),
    ExampleCase(
        "tunable_coupler__eigenmode_two_qubit_resonator_model_inaccuracy",
        TUNABLE_COUPLER / "main_eigenmode_two_qubit_resonator_model_inaccuracy.ipynb",
    ),
    ExampleCase(
        "tunable_coupler__eigenmode_two_qubit_resonator_model_inaccuracy_qb_res",
        TUNABLE_COUPLER
        / "main_eigenmode_two_qubit_resonator_model_inaccuracy_qb_res.ipynb",
    ),
    ExampleCase(
        "tunable_coupler__eigenmode_two_qubit_resonator_partitioned",
        TUNABLE_COUPLER / "main_eigenmode_two_qubit_resonator_partitioned.ipynb",
        # The partitions are created with save_path=None, so the notebook's final
        # overwrite_parameters() cell raises "A path must be specified".
        skip_cells_containing=("design_analysis.overwrite_parameters()",),
        get_optimization_results=lambda namespace: namespace["optimization_results"],
    ),
]

EXAMPLES_BY_NAME = {case.name: case for case in EXAMPLE_CASES}


@dataclass
class ExampleRun:
    case: ExampleCase
    session: SimulationSession
    namespace: dict
    workdir: Path

    @property
    def analysis(self):
        return self.namespace["design_analysis"]

    @property
    def optimization_results(self) -> list:
        return self.case.get_optimization_results(self.namespace)


@pytest.fixture(scope="session")
def run_example(tmp_path_factory) -> Callable[[str], ExampleRun]:
    """Run each example notebook at most once per test session and share the result."""
    cache: Dict[str, ExampleRun] = {}
    failures: Dict[str, BaseException] = {}

    def get(name: str) -> ExampleRun:
        if name in failures:
            pytest.fail(f"Example {name} failed to run: {failures[name]!r}")
        if name not in cache:
            case = EXAMPLES_BY_NAME[name]
            workdir = tmp_path_factory.mktemp(name)
            try:
                with pytest.MonkeyPatch.context() as monkeypatch:
                    session, namespace = run_example_notebook(
                        case.notebook,
                        workdir,
                        monkeypatch,
                        skip_cells_containing=case.skip_cells_containing,
                    )
            except BaseException as error:
                failures[name] = error
                raise
            cache[name] = ExampleRun(case, session, namespace, workdir)
        return cache[name]

    return get
