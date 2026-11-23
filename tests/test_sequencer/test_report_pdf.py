"""A written report printed to PDF: which browser, the page it prints,
and a report that stands without one."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from qdesignoptimizer.sequencer.engine import pdf
from qdesignoptimizer.sequencer.engine import report as rp
from qdesignoptimizer.sequencer.engine.runner import Runner

from .helpers import make_project
from .test_report import ADAPTER

REPORT = """# Simulation report

| Qubit | f01 |
| --- | --- |
| qubit_1 | 4.05 <GHz> |

![Study flow](study_flow.png)

```mermaid
flowchart TD
    s1["<b>study</b>"]
```

![Model](pictures/run/hfss.png)
"""


def _picture(path, width, height, gaps=()):
    """A grey picture with white rows at ``gaps``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (width, height), (90, 90, 90))
    for y in gaps:
        image.paste((255, 255, 255), (0, y, width, y + 10))
    image.save(path)


def test_the_browser_is_the_one_named_or_none(monkeypatch):
    monkeypatch.setenv(pdf.BROWSER_ENV, "none")
    assert pdf.browser() is None
    assert pdf.write([]) == []
    monkeypatch.setenv(pdf.BROWSER_ENV, "/opt/edge/msedge")
    assert pdf.browser() == "/opt/edge/msedge"


def test_the_page_drops_the_mermaid_source_and_cuts_a_tall_picture(tmp_path):
    folder, scratch = tmp_path / "report", tmp_path / "scratch"
    scratch.mkdir(parents=True)
    # five pages tall, with blank rows between its cards
    _picture(folder / "study_flow.png", 200, 1400, gaps=range(100, 1400, 150))
    _picture(folder / "pictures" / "run" / "hfss.png", 200, 150)

    page = pdf.page(REPORT, folder, scratch)

    assert "mermaid" not in page and "flowchart" not in page
    assert "<table>" in page and "&lt;GHz&gt;" in page
    parts = sorted(scratch.glob("study_flow_part*.png"))
    assert len(parts) >= 5
    assert all(p.as_uri() in page for p in parts)
    # each part fits a page, and the parts make the whole picture
    heights = [Image.open(p).size[1] for p in parts]
    assert max(heights) <= 200 * pdf.PAGE_ASPECT and sum(heights) == 1400
    # cut at the blank rows, not through a card
    for part in parts[:-1]:
        last = (
            Image.open(part).convert("L").getpixel((100, Image.open(part).size[1] - 1))
        )
        assert last == 255
    # a picture that fits stays as it is, found from the report's folder
    assert "](pictures/run/hfss.png)" not in page  # rendered, not left as Markdown
    assert 'src="pictures/run/hfss.png"' in page
    assert folder.resolve().as_uri() in page


def test_a_browser_that_does_not_print_leaves_no_pdf(tmp_path, monkeypatch):
    monkeypatch.setenv(pdf.BROWSER_ENV, str(tmp_path / "no-such-browser"))
    report = tmp_path / "report.md"
    report.write_text(REPORT, "utf-8")
    assert pdf.write([report]) == []
    assert not report.with_suffix(".pdf").exists()


def test_a_written_report_comes_as_pdf_where_a_browser_prints(tmp_path, monkeypatch):
    monkeypatch.delenv(pdf.BROWSER_ENV)
    if pdf.browser() is None:
        pytest.skip("no Chromium browser here to print with")
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    project = make_project(tmp_path / "p")
    (project.root / "sequencer.py").write_text(ADAPTER, "utf-8")
    project.add(project.backend.catalog().tasks[0].key)
    Runner(project).run()
    from qdesignoptimizer.sequencer.engine.workspace import Workspace

    workspace = Workspace([project])
    runs = rp.collect(workspace, rp.default_selection(rp.completed_runs(workspace)))

    folder = rp.write(project, runs, None)

    for name in ("report.pdf", "report_brief.pdf"):
        assert (folder / name).read_bytes()[:5] == b"%PDF-"
