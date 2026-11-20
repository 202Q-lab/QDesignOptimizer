"""Dash entry point for the queue GUI.

Run with::

    python -m qdesignoptimizer.sequencer.app [<project> ...] [--port 8051]

A project is its ``sequencer.py`` or its folder (searched for one); a
parent folder opens every design below it. Several projects share one
queue (``engine.workspace``). The connected project folders are
remembered (``connected.json`` in the user's ``qdo_sequencer`` folder):
with no project named, the last ones open again; one named is added.

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
from ..engine.workspace import Workspace, user_home
from . import layout, theme
from .callbacks import (
    register_callbacks,
    register_editor_callbacks,
    register_pins_callbacks,
    register_preview_callbacks,
    register_template_callbacks,
)
from .queue_state import Bridge, read_connected

#: The connected project folders, kept for the next start.
CONNECTED_FILE = "connected.json"

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
    remember: Path | None = None,
) -> Dash:
    """Build the queue GUI for one project or several.

    Args:
        ref: A project's ``sequencer.py`` or folder, or a list of them.
        project: Injected project (tests); else opened from ``ref``.
        workspace: Injected workspace (tests).
        remember: The file the connected project folders are kept in.

    Returns:
        The Dash app, not started.
    """
    bridge = Bridge(ref, project=project, workspace=workspace, remember=remember)
    app = Dash(
        __name__,
        title="QDesignOptimizer orchestrator",
        update_title=None,  # type: ignore[arg-type]
        suppress_callback_exceptions=True,
    )
    app.index_string = _INDEX
    # A function: each page load gets its own tab id (layout.page).
    app.layout = layout.page
    register_callbacks(app, bridge)
    register_template_callbacks(app, bridge)
    register_preview_callbacks(app, bridge)
    register_pins_callbacks(app, bridge)
    register_editor_callbacks(app, bridge)
    app.qdo_bridge = bridge  # type: ignore[attr-defined]
    return app


def _gone(ref: str) -> bool:
    """Whether a remembered project is no longer there: a folder or file
    (``file.py:factory`` too) that does not exist. A dotted module name
    is not a path, and is left to load (or say why it does not)."""
    text = str(ref)
    if ".py:" in text:
        text = text.split(".py:", 1)[0] + ".py"
    elif not (text.endswith(".py") or "/" in text or "\\" in text):
        return False
    return not Path(text).exists()


def main(argv: list[str] | None = None) -> None:
    """``python -m qdesignoptimizer.sequencer.app``."""
    parser = argparse.ArgumentParser(prog="python -m qdesignoptimizer.sequencer.app")
    parser.add_argument(
        "project",
        nargs="*",
        help="a project's sequencer.py or folder, added to the connected "
        "ones (none: those connected last time); several share one queue",
    )
    parser.add_argument("--port", type=int, default=8051)
    args = parser.parse_args(argv)
    debug = os.environ.get("QDO_SEQUENCER_DEBUG", "") not in ("", "0", "false")
    if not debug:
        # The page polls every 2 s; one request line per poll buries
        # anything worth reading in the console.
        logging.getLogger("werkzeug").setLevel(logging.WARNING)
    remember = user_home() / CONNECTED_FILE
    refs = read_connected(remember)
    for given in args.project:
        given = str(Path(given).resolve()) if Path(given).exists() else given
        if given not in refs:
            refs.append(given)
    # Gone since last time (moved, deleted): not opened, not forgotten
    # silently either -- the console says so.
    missing = [r for r in refs if _gone(r)]
    for ref in missing:
        print(f"not connected (not there any more): {ref}")
    refs = [r for r in refs if r not in missing]
    app = build_app(refs, remember=remember)
    try:
        app.run(debug=debug, port=args.port)
    finally:
        app.qdo_bridge.close()
