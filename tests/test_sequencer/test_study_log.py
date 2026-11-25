"""study.log survives QDO's logger module being imported during a solve.

``qdesignoptimizer.logger`` removes the first StreamHandler on the root
logger when first imported, and a FileHandler is one. A project's solve
imports DesignAnalysis (and with it that module) inside ``study_log``, so
the backend must have imported it already. Run in a fresh interpreter:
in this one the module was imported long ago.
"""

import subprocess
import sys
import textwrap

_SCRIPT = textwrap.dedent(
    """
    import logging, sys
    from pathlib import Path
    from qdesignoptimizer.sequencer.backends import qdo_record
    from qdesignoptimizer.sequencer.backends.qdo import QDOBackend

    assert "qdesignoptimizer.logger" not in sys.modules
    QDOBackend(object())
    folder = Path(sys.argv[1])
    with qdo_record.study_log(folder) as path:
        import qdesignoptimizer.logger  # what a solve's imports do
        logging.getLogger("qdesignoptimizer.sequencer.x").info("after import")
    print(path.read_text(encoding="utf-8"))
    """
)


def test_study_log_keeps_its_handler_when_a_solve_imports_qdo(tmp_path):
    done = subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert "after import" in done.stdout
