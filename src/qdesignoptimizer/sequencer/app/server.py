"""Dash entry point for the queue GUI.

Run with::

    python -m qdesignoptimizer.sequencer.app <project> [<project> ...] [--port 8051]

A project is its ``sequencer.py`` or its folder (searched for one); a
parent folder opens every design below it. Several projects share one
queue (``engine.workspace``).

``build_app()`` builds the app (layout + callbacks) without starting a
server, so it can be imported and smoke-tested. Port 8051 by default,
so it does not collide with setup-analyzer's 8050.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from dash import Dash

from ..engine.project import Project
from ..engine.workspace import Workspace
from . import layout, theme
from .callbacks import (
    register_callbacks,
    register_draft_callbacks,
    register_template_callbacks,
)
from .queue_state import Bridge

_INDEX = (
    """<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
<style>"""
    + theme.CSS
    + """</style>
</head>
<body>
{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>"""
)


def build_app(
    ref: str | list = "",
    project: Project | None = None,
    workspace: Workspace | None = None,
) -> Dash:
    """Build the queue GUI for one project or several.

    Args:
        ref: A project's ``sequencer.py`` or folder, or a list of them.
        project: Injected project (tests); else opened from ``ref``.
        workspace: Injected workspace (tests).

    Returns:
        The Dash app, not started.
    """
    bridge = Bridge(ref, project=project, workspace=workspace)
    try:
        name = " + ".join(bridge.workspace.projects)
    except Exception:  # noqa: BLE001 -- the page shows the error itself
        name = Path(str(ref[0] if isinstance(ref, list) else ref)).name
    app = Dash(
        __name__,
        title=f"Queue - {name}",
        update_title=None,  # type: ignore[arg-type]
        suppress_callback_exceptions=True,
    )
    app.index_string = _INDEX
    app.layout = layout.page(name)
    register_callbacks(app, bridge)
    register_template_callbacks(app, bridge)
    register_draft_callbacks(app, bridge)
    app.qdo_bridge = bridge  # type: ignore[attr-defined]
    return app


def main(argv: list[str] | None = None) -> None:
    """``python -m qdesignoptimizer.sequencer.app``."""
    parser = argparse.ArgumentParser(prog="python -m qdesignoptimizer.sequencer.app")
    parser.add_argument(
        "project",
        nargs="+",
        help="a project's sequencer.py or folder; several share one queue",
    )
    parser.add_argument("--port", type=int, default=8051)
    args = parser.parse_args(argv)
    debug = os.environ.get("QDO_SEQUENCER_DEBUG", "") not in ("", "0", "false")
    if not debug:
        # The page polls every 2 s; one request line per poll buries
        # anything worth reading in the console.
        logging.getLogger("werkzeug").setLevel(logging.WARNING)
    app = build_app(list(args.project))
    try:
        app.run(debug=debug, port=args.port)
    finally:
        app.qdo_bridge.close()
