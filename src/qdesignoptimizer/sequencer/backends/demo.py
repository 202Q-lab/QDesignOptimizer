"""A demo backend: simulation-shaped studies, a 5 s sleep for a solve.

``FakeBackend`` with names that read like a design project -- per
resonator a ``frequency`` study (tunes its length) and a ``linewidth``
study (tunes its coupling length), and a "tune" study sequence per resonator
-- and every study taking ``seconds`` (default 5) to "solve". Nothing is
simulated: a study proposes the value in its ``task.yaml``.

Used by ``tutorials/example_simulation_sequencer/demo/``; see its README.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from ..engine.backend import (
    Catalog,
    SequenceEntry,
    SequenceSpec,
    SubjectSpec,
    TaskSpec,
    Words,
)
from .fake import FakeBackend

#: Seconds a demo "solve" takes.
SOLVE_SECONDS = 5

_STUDIES = {
    "frequency": ("frequency (tunes the length)", "eigenmode"),
    "linewidth": ("linewidth (tunes the coupling)", "capacitance"),
}


class DemoBackend(FakeBackend):
    """See the module docstring.

    Args:
        root: The demo project's folder (``out/`` and ``store.json``
            are made there).
        resonators: Subject keys, e.g. ``("R1", "R2")``.
    """

    name = "demo"
    words = Words(
        task="study",
        sequence="study sequence",
        subject="subsystem",
        store="design variables",
        live="solve",
    )

    def __init__(self, root: Path, resonators: Iterable[str]):
        self.resonators = tuple(resonators)
        store = Path(root) / "store.json"
        if not store.exists():
            # Before FakeBackend's own seeding, which would name its
            # A and B subjects.
            store.parent.mkdir(parents=True, exist_ok=True)
            store.write_text(
                json.dumps({f"{r}_value": 0.0 for r in self.resonators}),
                encoding="utf-8",
            )
        super().__init__(root)

    def catalog(self) -> Catalog:
        return Catalog(
            tasks=[
                TaskSpec(
                    f"{r}.{kind}",
                    r,
                    f"{r} {label}",
                    family=family,
                    minutes=SOLVE_SECONDS / 60,
                )
                for r in self.resonators
                for kind, (label, family) in _STUDIES.items()
            ],
            sequences=[
                SequenceSpec(
                    f"tune_{r}",
                    f"Tune {r}: frequency, then linewidth",
                    (
                        SequenceEntry(
                            f"{r}.frequency",
                            save=True,
                            checks=("converged",),
                        ),
                        SequenceEntry(
                            f"{r}.linewidth",
                            save=True,
                            checks=("converged",),
                        ),
                    ),
                    why_ordered="the linewidth study starts from the "
                    "length the frequency study wrote back",
                )
                for r in self.resonators
            ],
            subjects=[
                SubjectSpec(r, f"Resonator {r}", "demo") for r in self.resonators
            ],
        )

    def default_settings(self, task: str) -> dict:
        return {"seconds": SOLVE_SECONDS}
