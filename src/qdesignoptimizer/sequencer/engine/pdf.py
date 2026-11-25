"""A written report as PDF: its Markdown as a page that a Chromium browser
(Edge, Chrome or Chromium, run headless) prints.

``QDO_PDF_BROWSER`` names the browser to use, or ``none`` for no PDFs;
without it the usual places are searched. Without a browser a report has
no PDF, and nothing else changes.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from html import escape
from pathlib import Path

BROWSER_ENV = "QDO_PDF_BROWSER"
TIMEOUT_SECONDS = 120
# The printable part of an A4 page (12 mm margins) is this much taller than wide.
PAGE_ASPECT = 273 / 186

_COMMANDS = ("msedge", "chrome", "google-chrome", "chromium", "chromium-browser")
_PLACES = {
    "win32": [
        r"Microsoft\Edge\Application\msedge.exe",
        r"Google\Chrome\Application\chrome.exe",
    ],
    "darwin": [
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
    ],
}

CSS = """
@page { size: A4; margin: 12mm; }
body { font-family: "Segoe UI", Helvetica, Arial, sans-serif; font-size: 10pt;
       color: #1a1a1a; line-height: 1.4; }
h1 { font-size: 18pt; border-bottom: 2px solid #333; padding-bottom: 4px; }
h2 { font-size: 14pt; border-bottom: 1px solid #aaa; margin-top: 1.6em;
     break-after: avoid; }
h3, h4 { font-size: 11.5pt; break-after: avoid; }
table { border-collapse: collapse; margin: 0.6em 0; font-size: 8pt; }
tr, img { break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 2px 5px; text-align: left;
         vertical-align: top; }
th { background: #eee; }
td em { color: #6b7380; font-size: 7pt; }
code { font-family: Consolas, Menlo, monospace; font-size: 8.5pt;
       background: #f3f3f3; padding: 0 2px; }
pre { background: #f3f3f3; padding: 6px; white-space: pre-wrap; font-size: 8pt; }
img { max-width: 100%; max-height: 260mm; }
"""

# The study flow's Mermaid source: the picture beside it is what it draws.
_MERMAID = re.compile(r"```mermaid\n.*?\n```\n?", re.S)
_IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")


def _html(markdown: str) -> str:
    """Markdown as HTML, any HTML in it as text; a picture may come from
    disk (``file:``), as the page is printed here, never served."""
    import mistune  # pylint: disable=import-outside-toplevel

    class Renderer(mistune.HTMLRenderer):
        def safe_url(self, url: str) -> str:
            if url.startswith("file:"):
                return escape(url)
            return super().safe_url(url)

    render = mistune.create_markdown(
        renderer=Renderer(escape=True), plugins=["table", "strikethrough"]
    )
    return str(render(markdown))


def browser() -> str | None:
    """The browser that prints, or None for no PDFs."""
    named = os.environ.get(BROWSER_ENV, "").strip()
    if named:
        return None if named.lower() == "none" else named
    for command in _COMMANDS:
        found = shutil.which(command)
        if found:
            return found
    roots = [
        os.environ.get(key, "")
        for key in ("PROGRAMFILES(X86)", "PROGRAMFILES", "LOCALAPPDATA")
    ]
    for place in _PLACES.get(sys.platform, []):
        for candidate in [Path(place)] + [Path(r) / place for r in roots if r]:
            if candidate.is_absolute() and candidate.is_file():
                return str(candidate)
    return None


def write(reports: list[Path]) -> list[Path]:
    """Each Markdown report as ``<name>.pdf`` beside it, printed at the
    same time; the PDFs written (none without a browser, or for a report
    the browser did not print)."""
    command = browser()
    if command is None or not reports:
        return []
    with tempfile.TemporaryDirectory(prefix="qdo_pdf_") as scratch:
        jobs = [
            (command, report, Path(scratch) / str(n))
            for n, report in enumerate(reports)
        ]
        with ThreadPoolExecutor(len(jobs)) as pool:
            made = list(pool.map(lambda job: _print(*job), jobs))
    return [pdf for pdf in made if pdf is not None]


def page(markdown: str, folder: Path, scratch: Path) -> str:
    """The HTML page of a report written in ``folder``; a picture taller
    than a page is cut into ``scratch``."""
    text = _MERMAID.sub("", markdown)

    def image(match: re.Match) -> str:
        parts = _page_parts(folder / match[2], scratch)
        return "\n\n".join(f"![{match[1]}]({p.as_uri()})" for p in parts) or match[0]

    body = _html(_IMAGE.sub(image, text))
    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        f'<base href="{folder.resolve().as_uri()}/">'
        f"<title>{escape(folder.name)}</title><style>{CSS}</style>"
        f"</head><body>{body}</body></html>"
    )


def _print(command: str, report: Path, scratch: Path) -> Path | None:
    """One report printed to PDF (None if the browser did not)."""
    scratch.mkdir()
    page_file = scratch / "report.html"
    page_file.write_text(
        page(report.read_text("utf-8"), report.parent, scratch), "utf-8"
    )
    pdf = report.with_suffix(".pdf")
    pdf.unlink(missing_ok=True)
    try:
        subprocess.run(
            [
                command,
                "--headless",
                "--disable-gpu",
                "--no-first-run",
                "--no-pdf-header-footer",
                # its own profile: a browser the user has open is not touched
                f"--user-data-dir={scratch / 'profile'}",
                f"--print-to-pdf={pdf}",
                page_file.as_uri(),
            ],
            capture_output=True,
            timeout=TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return pdf if pdf.is_file() and pdf.stat().st_size > 0 else None


def _page_parts(picture: Path, scratch: Path) -> list[Path]:
    """A picture taller than a page, cut at blank rows into parts that
    each fit one; [] for a picture that fits as it is (or does not read)."""
    try:
        from PIL import Image

        full = Image.open(picture)
        full.load()
    except Exception:  # noqa: BLE001 -- the picture stays as it is
        return []
    width, height = full.size
    step = int(width * PAGE_ASPECT * 0.95)
    if height <= step:
        return []
    gray = full.convert("L").tobytes()  # a byte a pixel, row by row

    def blank(y: int) -> bool:
        return min(gray[y * width : (y + 1) * width : 4]) > 245

    cuts, top = [], 0
    while height - top > step:
        cut = top + step
        while cut > top + step // 2 and not blank(cut):
            cut -= 1
        if cut <= top + step // 2:  # no blank row: cut at the page
            cut = top + step
        cuts.append((top, cut))
        top = cut
    cuts.append((top, height))
    parts = []
    for n, (upper, lower) in enumerate(cuts, 1):
        part = scratch / f"{picture.stem}_part{n}.png"
        full.crop((0, upper, width, lower)).save(part)
        parts.append(part)
    return parts
