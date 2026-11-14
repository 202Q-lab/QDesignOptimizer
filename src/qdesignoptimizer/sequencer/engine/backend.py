"""What a backend must provide, and the plain records it trades in.

A backend turns a study key (``ATS1.eigenmode``) into the files of
a queued study, checks those files, and runs a study from them. The
engine owns the folders, the order, the locks and the rule that a
store is written only after every check passed; it never touches
Ansys, an instrument or a store itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class Words:
    """The names a repo's users know the engine's concepts by.

    The engine says task / measurement plan / subject / store. The simulation
    backend says study / measurement plan / subsystem / design variables; a
    measurement backend would say experiment / measurement plan / element /
    calibration database. ``job.yaml`` accepts the backend's word as
    a key (``study: ATS1.kappa``) as well as ``task:``, and the GUI
    prints the backend's word in every header.
    """

    task: str = "task"
    plan: str = "measurement plan"
    subject: str = "subject"
    store: str = "store"
    live: str = "live"

    def title(self, word: str) -> str:
        """``self.<word>`` with its first letter capitalised."""
        text = getattr(self, word)
        return text[:1].upper() + text[1:]


@dataclass(frozen=True)
class TaskSpec:
    """One study the backend can queue (for QDO: one template, filled
    in for one subject).

    Attributes:
        key: Stable id (``ATS1.kappa``).
        subject: The subject key it belongs to.
        label: Human name.
        family: Free word for grouping (``eigenmode``, ``capacitance``).
        summary: One line on what it answers / which code generates it.
        minutes: Rough wall time per iteration, None when unmeasured.
    """

    key: str
    subject: str
    label: str
    family: str = ""
    summary: str = ""
    minutes: float | None = None


@dataclass(frozen=True)
class PlanEntry:
    """One study of a measurement plan, with what it changes from the defaults."""

    task: str
    settings: dict = field(default_factory=dict)
    checks: tuple[str, ...] | None = None
    save: bool | None = None
    on_failure: str | None = None
    notes: str = ""


@dataclass(frozen=True)
class PlanSpec:
    """A named sequence of studies. Enqueueing it writes one study
    folder per entry, tagged with the measurement plan's id.

    Attributes:
        key: Stable id (``tune_ATS1``, or a measurement plan file's stem).
        label: Human name.
        entries: The studies, in order.
        why_ordered: Why that order; shown in the GUI.
        source: The file it came from, or "" for one defined in code.
    """

    key: str
    label: str
    entries: tuple[PlanEntry, ...]
    why_ordered: str = ""
    source: str = ""


@dataclass(frozen=True)
class SubjectSpec:
    """A thing tuned or characterised on its own (one ATS, one qubit)."""

    key: str
    label: str
    group: str = ""


@dataclass(frozen=True)
class FileSpec:
    """One editable file of a queued study.

    Attributes:
        name: File name inside the study folder.
        label: Tab title in the GUI.
        optional: True when a missing file is filled in from the
            project when the study starts.
        about: One line shown above the editor.
    """

    name: str
    label: str
    optional: bool = False
    about: str = ""


@dataclass
class Catalog:
    """Everything a backend can enqueue; the GUI's Add menu."""

    tasks: list[TaskSpec] = field(default_factory=list)
    plans: list[PlanSpec] = field(default_factory=list)
    subjects: list[SubjectSpec] = field(default_factory=list)

    def task(self, key: str) -> TaskSpec | None:
        """The task with this key, or None."""
        return next((t for t in self.tasks if t.key == key), None)

    def plan(self, key: str) -> PlanSpec | None:
        """The measurement plan with this key, or None."""
        return next((c for c in self.plans if c.key == key), None)


@dataclass
class StepResult:
    """What one study produced.

    ``outputs`` and ``updates`` are deliberately apart. Outputs are what
    was simulated or measured and never reach the store; updates are the
    store values the study proposes, and the engine commits them only
    after every check passed.
    """

    outputs: dict[str, Any] = field(default_factory=dict)
    updates: dict[str, Any] = field(default_factory=dict)
    files: list[str] = field(default_factory=list)
    message: str = ""


@dataclass
class CheckResult:
    """The verdict of one named check."""

    ok: bool
    message: str = ""


@runtime_checkable
class Backend(Protocol):
    """The one object a repo implements to be driven by the engine."""

    name: str
    words: Words

    @property
    def out_dir(self) -> Path:
        """Where ``queue/`` and ``completed/`` live."""

    def catalog(self) -> Catalog:
        """Every study and measurement plan. Must be cheap (no Ansys)."""

    def files(self) -> list[FileSpec]:
        """The backend's files in a study folder, in display order."""

    def settings_keys(self) -> set[str]:
        """What ``job.yaml``'s ``settings`` may hold."""

    def check_names(self) -> set[str]:
        """The named checks ``job.yaml``'s ``checks`` may list."""

    def default_settings(self, task: str) -> dict[str, Any]:
        """The settings a new study of this task starts with."""

    def generate(self, task: str) -> tuple[dict[str, str], str]:
        """Make a study's files: ``({file name: text}, generated_by)``.
        May import QDO; must not start Ansys."""

    def check_folder(self, folder: Path, job, deep: bool) -> list[str]:
        """Problems with a study's files. ``deep`` rebuilds the objects
        (imports QDO); shallow only parses. Empty means runnable."""

    def resolve_inputs(self, folder: Path, job) -> None:
        """At start: fill in optional files from the project, so the
        folder records exactly what ran."""

    def before_step(self, job) -> None:
        """Put the resource in a known state (close and wait for Ansys)."""

    def run(self, folder: Path, job) -> StepResult:
        """Run the study from its folder."""

    def check(self, name: str, job, result: StepResult) -> CheckResult:
        """A named check (``converged``, ``mode_identity``)."""

    def commit(self, updates: dict[str, Any]) -> dict[str, Any] | None:
        """Write accepted updates to the project's store."""

    def store_values(self) -> dict[str, Any]:
        """The project's store as it stands."""

    def describe(self, folder: Path, job) -> dict[str, Any]:
        """What the study would hand its solver, as plain data."""
