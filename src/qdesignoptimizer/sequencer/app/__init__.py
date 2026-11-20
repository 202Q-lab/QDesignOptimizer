"""The Dash queue GUI. The ONLY package in qdesignoptimizer.sequencer that imports Dash.

``queue_state`` is the bridge to the queue and state files and imports
no Dash either, so every edit the GUI makes is testable without a
browser. ``server.build_app`` builds the app without starting it.
"""
