"""The partitioned tunable-coupler tutorial project, short of Ansys.

``tutorials/example_simulation_sequencer/tunable_coupler_partitioned``
splits the chip as ``main_eigenmode_two_qubit_resonator_partitioned.ipynb``
does. What a test can hold: each partition takes only the targets the
notebook updates from it (its ``include_param_in_update``), the study
sequence queues, and every study deep-checks against the built chip.
"""

import shutil
from pathlib import Path

import yaml

from qdesignoptimizer.sequencer.engine.workspace import open_workspace

_TUTORIALS = Path(__file__).resolve().parents[2] / "tutorials"
_PROJECT = _TUTORIALS / "example_simulation_sequencer" / "tunable_coupler_partitioned"
_CHIP = _TUTORIALS / "examples_coupled_transmons_tunable_coupler_chip"


def _copy(tmp_path: Path) -> Path:
    """The project and the chip it builds on, in the same layout."""
    shutil.copytree(_CHIP, tmp_path / _CHIP.name)
    root = tmp_path / "example_simulation_sequencer" / _PROJECT.name
    shutil.copytree(
        _PROJECT,
        root,
        ignore=shutil.ignore_patterns(
            "out", "design_variables.json", "*.meta.json", "__pycache__"
        ),
    )
    return root


def _targets(folder: Path) -> dict:
    entries = yaml.safe_load((folder / "opt_targets.yaml").read_text("utf-8"))
    return {e["name"]: e.get("active", True) for e in entries["targets"]}


def test_each_partition_tunes_only_its_own_targets(tmp_path):
    root = _copy(tmp_path)
    workspace = open_workspace([root], isolate=True)
    try:
        (project,) = workspace.projects.values()
        catalog = project.backend.catalog()
        assert project.backend.catalog_problems == []
        assert {t.key for t in catalog.tasks} == {
            "Q1.readout",
            "Q2.readout",
            "C.coupler",
        }
        ids = project.add_sequence(catalog.sequence("tune_partitioned"))
        assert len(ids) == 9  # three rounds of the three partitions
        # Round 3 may fail to converge partition by partition: each needs
        # rounds 1 and 2, not the other round-3 studies.
        for job_id in ids[6:]:
            assert set(project.job(job_id).needs) == set(ids[:6])
        for job_id in ids[:3]:
            assert project.problems(job_id, deep=True) == []

        q1, q2, coupler = (_targets(project.queue.folder(i)) for i in ids[:3])
        # Q1.readout has the coupler in its model, but not its targets.
        assert q1 == dict.fromkeys(
            ["Q1_freq", "Q1_anharmonicity", "R1_freq", "Q1_R1_chi"], True
        )
        assert q2 == dict.fromkeys(
            ["Q2_freq", "Q2_anharmonicity", "R2_freq", "Q2_R2_chi"], True
        )
        # C.coupler has both qubits in its model, but not their targets.
        assert coupler == dict.fromkeys(
            ["C_freq", "C_anharmonicity", "C_Q1_chi", "C_Q2_chi"], True
        )
    finally:
        workspace.close()
