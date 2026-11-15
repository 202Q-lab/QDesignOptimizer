"""Colours and the page stylesheet. One home for every visual constant."""

from __future__ import annotations

#: Status -> (text colour, background). One colour per meaning.
STATUS_COLORS = {
    "queued": ("#37474f", "#eceff1"),
    "running": ("#0d47a1", "#bbdefb"),
    "done": ("#1b5e20", "#c8e6c9"),
    "failed": ("#b71c1c", "#ffcdd2"),
    "skipped": ("#546e7a", "#f5f5f5"),
    "blocked": ("#546e7a", "#e0e0e0"),
    "stale": ("#e65100", "#ffe0b2"),
    "interrupted": ("#e65100", "#ffe0b2"),
    "committing": ("#0d47a1", "#bbdefb"),
}

RUN_COLORS = {
    "idle": "#546e7a",
    "running": "#0d47a1",
    "finished": "#1b5e20",
    "stopped": "#546e7a",
    "aborted": "#b71c1c",
    "error": "#b71c1c",
    "held": "#e65100",
    "stopped unexpectedly": "#b71c1c",
}

ACCENT = "#1565c0"
MUTED = "#607d8b"
BORDER = "#cfd8dc"
SELECTED_BG = "#e3f2fd"

CSS = """
body { font-family: "Segoe UI", system-ui, sans-serif; margin: 0;
       background: #fafafa; color: #263238; font-size: 14px; }
.qs-page { padding: 8px 12px; height: 100vh; box-sizing: border-box;
           display: flex; flex-direction: column; }
.qs-header { display: flex; flex-wrap: wrap; align-items: center;
             gap: 10px; margin-bottom: 10px; }
.qs-logo { height: 26px; }
.qs-project { margin-bottom: 8px; font-weight: 600; }
.qs-navhead { display: flex; justify-content: space-between;
              align-items: center; margin: 10px 0 4px;
              border-bottom: 1px solid #eceff1; padding-bottom: 2px; }
.qs-btn-small { padding: 1px 8px; font-size: 12px; }
.qs-title { font-size: 20px; font-weight: 600; }
.qs-muted { color: #607d8b; }
.qs-bar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
          margin: 8px 0; }
.qs-btn { border: 1px solid #b0bec5; background: white; border-radius: 6px;
          padding: 5px 12px; cursor: pointer; font-size: 13px; }
.qs-btn:hover { background: #eceff1; }
.qs-btn:disabled { opacity: 0.45; cursor: default; }
.qs-btn-primary { border-color: #1565c0; color: #1565c0; }
.qs-btn-danger { border-color: #b71c1c; color: #b71c1c; }
.qs-icon { border: none; background: none; cursor: pointer; padding: 0 4px;
           font-size: 14px; color: #546e7a; }
.qs-icon:hover { color: #1565c0; }
/* Left (shared queue over project) | divider | study editor. The left
   width is --qs-left, set by dragging the divider (assets/splitter.js). */
.qs-main { display: grid; flex: 1; min-height: 0;
           grid-template-columns: var(--qs-left, 42%) 8px minmax(0, 1fr); }
.qs-left { display: flex; flex-direction: column; gap: 8px; min-height: 0;
           min-width: 0; }
.qs-left > .qs-area { flex: 1 1 50%; }
.qs-area { background: white; border: 1px solid #cfd8dc; border-radius: 8px;
           padding: 10px 12px; overflow: auto; min-height: 0; }
.qs-editor-area { height: 100%; box-sizing: border-box; }
.qs-area-title { font-size: 13px; text-transform: uppercase;
                 letter-spacing: 0.06em; color: #1565c0; margin: 0 0 6px; }
.qs-splitter { cursor: col-resize; position: relative; }
.qs-splitter::after { content: ""; position: absolute; top: 0; bottom: 0;
                      left: 3px; width: 2px; background: #cfd8dc; }
.qs-splitter:hover::after, .qs-splitter.qs-active::after {
  background: #1565c0; }
.qs-inset { border: 1px solid #eceff1; border-radius: 6px; padding: 8px;
            margin: 6px 0; }
.qs-mode { display: flex; flex-wrap: wrap; justify-content: space-between;
           align-items: center; gap: 8px; border-radius: 6px;
           padding: 6px 10px; margin-bottom: 8px; }
.qs-mode-inspect { background: #eceff1; border: 1px solid #cfd8dc; }
.qs-mode-edit { background: #e8f5e9; border: 1px solid #66bb6a; }
.qs-warning { background: #fff8e1; border: 1px solid #ffb300;
              border-radius: 6px; padding: 8px; margin-bottom: 8px; }
.qs-brow { padding: 2px 4px; border-radius: 4px; }
.qs-light { background: #eceff1; color: #263238; }
.qs-readme { font-size: 13px; }
/* Narrower than this, the areas stack and the page scrolls. */
@media (max-width: 1100px) {
  .qs-page { height: auto; }
  .qs-main { grid-template-columns: 1fr; gap: 8px; }
  .qs-splitter { display: none; }
  .qs-area { overflow: visible; }
}
.qs-card { background: white; border: 1px solid #cfd8dc;
           border-radius: 8px; padding: 12px; overflow-x: auto; }
.qs-seqrow { border-bottom: 1px solid #eceff1; padding: 4px 0; }
.qs-table { width: 100%; border-collapse: collapse; }
.qs-table th { text-align: left; font-weight: 500; color: #607d8b;
               border-bottom: 1px solid #b0bec5; padding: 6px; }
.qs-table td { border-bottom: 1px solid #eceff1; padding: 6px;
               vertical-align: top; }
.qs-row { cursor: pointer; }
.qs-row:hover { background: #f5f9ff; }
.qs-chip { display: inline-block; border-radius: 10px; padding: 1px 9px;
           font-size: 12px; white-space: nowrap; }
.qs-h { margin: 0 0 8px; font-size: 15px; }
.qs-pick { cursor: pointer; }
.qs-pick:hover { color: #1565c0; }
.qs-small { font-size: 12px; color: #607d8b; }
.qs-tag { display: inline-block; border-radius: 4px; padding: 0 6px;
          font-size: 11px; background: #ede7f6; color: #4527a0;
          white-space: nowrap; }
.qs-btn-small { padding: 2px 8px; font-size: 12px; }
.qs-editor { width: 100%; height: 360px; font-family: Consolas, monospace;
             font-size: 12px; }
.qs-file { max-height: 420px; }
.qs-tabs { margin-top: 10px; }
.qs-row[draggable="true"] { cursor: grab; }
.qs-dragging { opacity: 0.4; }
.qs-drop-before td { box-shadow: inset 0 2px 0 #1565c0; }
.qs-drop-after td { box-shadow: inset 0 -2px 0 #1565c0; }
.qs-pre { background: #263238; color: #eceff1; border-radius: 6px;
          padding: 8px; font-size: 12px; max-height: 260px;
          overflow: auto; white-space: pre-wrap; }
.qs-field { margin: 6px 0; }
.qs-field label { display: block; font-size: 12px; color: #607d8b; }
.qs-field textarea { width: 100%; font-family: Consolas, monospace;
                     font-size: 12px; }
.qs-problems { background: #fff3e0; border: 1px solid #ffb74d;
               border-radius: 6px; padding: 8px 12px; margin: 8px 0; }
.qs-message { min-height: 20px; color: #37474f; }
.qs-form { margin: 8px 0; }
.qs-sec { border-left: 2px solid #cfd8dc; margin: 6px 0 6px 4px;
          padding-left: 10px; }
.qs-sec > summary { cursor: pointer; padding: 2px 0; }
.qs-sec-body { padding: 2px 0 4px; }
.qs-off { opacity: 0.55; border-left-style: dashed; }
.qs-off-tag { color: #b71c1c; font-size: 12px; }
.qs-frow { display: grid; grid-template-columns: minmax(140px, 38%) 1fr;
           gap: 8px; align-items: center; margin: 2px 0; }
.qs-frow label { font-size: 12px; color: #455a64; overflow-wrap: anywhere; }
.qs-input { width: 100%; box-sizing: border-box; padding: 2px 6px;
            font-size: 13px; }
.qs-mono { font-family: Consolas, monospace; font-size: 12px; }
.qs-scroll { max-height: 380px; overflow: auto; }
"""
