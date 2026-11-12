"""An in-process backend with no solver, for tests and the GUI demo.

Two subjects (``A``, ``B``), a ``calibrate`` and a ``measure`` task each,
and one campaign. A study's own file is ``task.yaml``::

    value: 3.0         # output ``value``, proposed as <subject>_value
    fail: false        # true: the task raises
    converged: true    # false: the ``converged`` check fails
    step: 0.0          # not used by the engine; kept for edits

``settings: {seconds: 2}`` makes a study take that long (GUI demo).
The store is ``store.json`` in the project folder; ``design_variables
.json`` in a study folder holds only what that study sets, the rest
pulled from the store when it starts -- the QDO backend's rule.

A demo project is one file::

    from pathlib import Path
    from qdesignoptimizer.sequencer.backends.fake import FakeBackend
    def make_backend():
        return FakeBackend(Path(__file__).parent)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import yaml

from ..engine.backend import (
    CampaignEntry,
    CampaignSpec,
    Catalog,
    CheckResult,
    FileSpec,
    StepResult,
    SubjectSpec,
    TaskSpec,
    Words,
)
from ..engine.job import dump_yaml
from ._store import read_json, record_writes, sources_text, update_json

TASK_FILE, VARIABLES = "task.yaml", "design_variables.json"
_TASK_KEYS = {"value", "fail", "converged", "step"}


class FakeBackend:
    """See the module docstring."""

    name = "fake"
    words = Words()

    def __init__(self, root: Path):
        self.root = Path(root)
        self.store_path = self.root / "store.json"
        if not self.store_path.exists():
            self.root.mkdir(parents=True, exist_ok=True)
            self.store_path.write_text(
                json.dumps({"A_value": 0.0, "B_value": 0.0}), encoding="utf-8"
            )
        self.calls: list[tuple[str, str]] = []
        self.fail_commit = False

    @property
    def out_dir(self) -> Path:
        return self.root / "out"

    def catalog(self) -> Catalog:
        return Catalog(
            tasks=[
                TaskSpec(f"{s}.{kind}", s, f"{s} {kind}", family=kind, minutes=0.1)
                for s in ("A", "B")
                for kind in ("calibrate", "measure")
            ],
            campaigns=[
                CampaignSpec(
                    "full_A",
                    "Calibrate then measure A",
                    (
                        CampaignEntry("A.calibrate", save=True),
                        CampaignEntry("A.measure"),
                    ),
                    why_ordered="measure reads what calibrate wrote back",
                )
            ],
            subjects=[
                SubjectSpec("A", "Subject A"),
                SubjectSpec("B", "Subject B"),
            ],
        )

    def files(self) -> list[FileSpec]:
        return [
            FileSpec(TASK_FILE, "Task", about="What the fake task does"),
            FileSpec(
                VARIABLES,
                "Design variables",
                optional=True,
                about="Values for this study only",
            ),
        ]

    def settings_keys(self) -> set[str]:
        return {"seconds"}

    def check_names(self) -> set[str]:
        return {"converged"}

    def default_settings(self, task: str) -> dict:
        return {}

    def generate(self, task: str) -> tuple[dict[str, str], str]:
        self.calls.append(("generate", task))
        body = {"value": 1.0, "fail": False, "converged": True, "step": 0.0}
        return (
            {TASK_FILE: dump_yaml(body, f"fake task {task}")},
            f"fake study {task}",
        )

    def check_folder(self, folder: Path, job, deep: bool) -> list[str]:
        problems = []
        try:
            data = (
                yaml.safe_load((Path(folder) / TASK_FILE).read_text(encoding="utf-8"))
                or {}
            )
        except FileNotFoundError:
            return [f"{TASK_FILE} is missing"]
        except yaml.YAMLError as exc:
            return [f"{TASK_FILE} is not valid YAML: {exc}"]
        bad = sorted(set(data) - _TASK_KEYS)
        if bad:
            problems.append(f"{TASK_FILE}: unknown key(s) {bad}")
        path = Path(folder) / VARIABLES
        if path.exists():
            try:
                partial = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                return problems + [f"{VARIABLES}: {exc}"]
            unknown = sorted(set(partial) - set(self.store_values()))
            if unknown:
                problems.append(f"{VARIABLES}: the project has no " f"{unknown}")
        return problems

    def resolve_inputs(self, folder: Path, job) -> None:
        path = Path(folder) / VARIABLES
        partial = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        path.write_text(
            json.dumps({**self.store_values(), **partial}, indent=4),
            encoding="utf-8",
        )

    def before_step(self, job) -> None:
        self.calls.append(("before_step", job.id))

    def run(self, folder: Path, job) -> StepResult:
        self.calls.append(("run", job.id))
        if job.settings.get("seconds"):
            time.sleep(float(job.settings["seconds"]))
        data = (
            yaml.safe_load((Path(folder) / TASK_FILE).read_text(encoding="utf-8")) or {}
        )
        if data.get("fail"):
            raise RuntimeError(f"{job.task} failed as asked")
        variables = json.loads((Path(folder) / VARIABLES).read_text(encoding="utf-8"))
        value = float(data.get("value", 1.0))
        key = f"{job.subject}_value"
        return StepResult(
            outputs={
                "value": value,
                "converged": bool(data.get("converged", True)),
                "started_from": variables.get(key),
            },
            updates={key: value},
            message=f"value {value}",
        )

    def check(self, name: str, job, result: StepResult) -> CheckResult:
        if name == "converged":
            ok = bool(result.outputs.get("converged"))
            return CheckResult(ok, "" if ok else "did not converge")
        return CheckResult(False, f"no check named {name!r}")

    def commit(self, updates: dict[str, Any]) -> dict[str, Any]:
        """Write the values; return those that changed."""
        if self.fail_commit:
            raise PermissionError("store is locked")
        return update_json(self.store_path, updates, allow_new=True)

    def store_values(self) -> dict[str, Any]:
        return read_json(self.store_path)

    def note_commit(self, updates, previous, source) -> None:
        """Record which study wrote the values (``store.writes.json``)."""
        record_writes(self.store_path, updates, previous, source)

    def value_sources(self, name: str) -> dict:
        """Which study last wrote each value of the optional file."""
        return sources_text(self.store_path) if name == VARIABLES else {}

    def project_values(self, name: str):
        """The project's values a study's optional file overrides."""
        return self.store_values() if name == VARIABLES else None

    def describe(self, folder: Path, job) -> dict[str, Any]:
        return {"task": job.task, "settings": job.settings}
