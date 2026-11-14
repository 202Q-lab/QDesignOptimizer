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
.qs-page { max-width: 1400px; margin: 0 auto; padding: 16px; }
.qs-header { display: flex; flex-wrap: wrap; align-items: baseline;
             gap: 16px; margin-bottom: 8px; }
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
.qs-main { display: grid; grid-template-columns: minmax(0, 3fr)
           minmax(320px, 2fr); gap: 16px; }
@media (max-width: 900px) { .qs-main { grid-template-columns: 1fr; } }
.qs-card { background: white; border: 1px solid #cfd8dc;
           border-radius: 8px; padding: 12px; }
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
.qs-modal { position: fixed; inset: 0; background: rgba(38, 50, 56, 0.45);
            z-index: 1000; }
.qs-modal-panel { position: fixed; left: 2vw; top: 4vh; width: 43vw;
                  max-height: 92vh; overflow: auto; background: white;
                  border-radius: 8px; padding: 16px; box-sizing: border-box;
                  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25); }
/* While the popup is open the details pane edits the draft's studies:
   it moves beside the popup, above the shade. */
.qs-drafting #details-card { position: fixed; right: 2vw; top: 4vh;
                             width: 51vw; max-height: 92vh; overflow: auto;
                             z-index: 1001; box-sizing: border-box;
                             box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25); }
@media (max-width: 900px) {
  .qs-modal-panel { width: 96vw; max-height: 44vh; }
  .qs-drafting #details-card { width: 96vw; top: 50vh; max-height: 46vh; }
}
"""
