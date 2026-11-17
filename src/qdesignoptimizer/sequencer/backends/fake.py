"""An in-process backend with no solver, for tests and the GUI demo.

Two subjects (``A``, ``B``), a ``calibrate`` and a ``measure`` task each,
and one study sequence. A study's own file is ``task.yaml``::

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
    Catalog,
    CheckResult,
    FileSpec,
    SequenceEntry,
    SequenceSpec,
    StepResult,
    SubjectSpec,
    TaskSpec,
    Words,
)
from ..engine.job import dump_yaml
from ..engine.sequence import load_sequence_dir
from ._store import (
    SwitchableStore,
    read_json,
    record_writes,
    sources_text,
    update_json,
    uses_job_store,
)

TASK_FILE, VARIABLES = "task.yaml", "design_variables.json"
_TASK_KEYS = {"value", "fail", "converged", "step"}


class FakeBackend(SwitchableStore):
    """See the module docstring."""

    name = "fake"
    words = Words()
    store_file_name = "store.json"

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
    def default_store(self) -> Path:
        return self.store_path

    @property
    def project_root(self) -> Path:
        return self.root

    @property
    def out_dir(self) -> Path:
        return self.root / "out"

    @property
    def sequence_dir(self) -> Path:
        """Study sequence files, as the QDO backend keeps them."""
        return self.root / "study_sequences"

    def catalog(self) -> Catalog:
        catalog = Catalog(
            tasks=[
                TaskSpec(f"{s}.{kind}", s, f"{s} {kind}", family=kind, minutes=0.1)
                for s in ("A", "B")
                for kind in ("calibrate", "measure")
            ],
            sequences=[
                SequenceSpec(
                    "full_A",
                    "Calibrate then measure A",
                    (
                        SequenceEntry("A.calibrate", save=True),
                        SequenceEntry("A.measure"),
                    ),
                    why_ordered="measure reads what calibrate wrote back",
                )
            ],
            subjects=[
                SubjectSpec("A", "Subject A"),
                SubjectSpec("B", "Subject B"),
            ],
        )
        files, self.catalog_problems = load_sequence_dir(self.sequence_dir, self.words)
        known = {c.key for c in catalog.sequences}
        catalog.sequences.extend(f for f in files if f.key not in known)
        return catalog

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

    def code_files(self) -> list[Path]:
        """``code.py`` in the project, when there is one (tests)."""
        path = self.root / "code.py"
        return [path] if path.exists() else []

    def reload(self) -> None:
        self.calls.append(("reload", ""))

    def check_names(self) -> set[str]:
        return {"converged"}

    def default_settings(self, task: str) -> dict:
        return {}

    def generate(self, task: str, campaign: str = "") -> tuple[dict[str, str], str]:
        del campaign  # its files have no values to fill in
        self.calls.append(("generate", task))
        body = {"value": 1.0, "fail": False, "converged": True, "step": 0.0}
        return (
            {TASK_FILE: dump_yaml(body, f"fake task {task}")},
            f"fake study {task}",
        )

    @uses_job_store
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

    @uses_job_store
    def resolve_inputs(self, folder: Path, job) -> None:
        path = Path(folder) / VARIABLES
        partial = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        path.write_text(
            json.dumps({**self.store_values(), **partial}, indent=4),
            encoding="utf-8",
        )

    def before_step(self, job) -> None:
        self.calls.append(("before_step", job.id))

    @uses_job_store
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

    def commit(
        self, updates: dict[str, Any], campaign: str | None = None
    ) -> dict[str, Any]:
        """Write the values; return those that changed."""
        if self.fail_commit:
            raise PermissionError("store is locked")
        with self.store_of(campaign):
            return update_json(self.store_file, updates, allow_new=True)

    def store_values(self, campaign: str | None = None) -> dict[str, Any]:
        with self.store_of(campaign):
            return read_json(self.store_file)

    def note_commit(self, updates, previous, source) -> None:
        """Record which study wrote the values (``store.meta.json``)."""
        with self.store_of(source.get("campaign") or ""):
            record_writes(self.store_file, updates, previous, source)

    def value_sources(self, name: str, campaign: str | None = None) -> dict:
        """Which study last wrote each value of the optional file."""
        with self.store_of(campaign):
            return sources_text(self.store_file) if name == VARIABLES else {}

    def project_values(self, name: str, campaign: str | None = None):
        """The project's (or a campaign's) values a study's optional
        file overrides."""
        return self.store_values(campaign) if name == VARIABLES else None

    def describe(self, folder: Path, job) -> dict[str, Any]:
        return {"task": job.task, "settings": job.settings}
