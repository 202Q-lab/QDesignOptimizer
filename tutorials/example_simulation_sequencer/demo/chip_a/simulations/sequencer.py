"""Demo project chip_a: the adapter qdesignoptimizer.sequencer opens.

A real design's sequencer.py wraps its QDO project in QDOBackend; this
one uses the demo backend, whose "solve" is a 5 s sleep.
"""

from pathlib import Path

from qdesignoptimizer.sequencer.backends.demo import DemoBackend

#: The design folder (out/ and store.json go here).
ROOT = Path(__file__).resolve().parents[1]


def make_backend():
    """The factory qdesignoptimizer.sequencer calls."""
    return DemoBackend(ROOT, ("R1", "R2"))
