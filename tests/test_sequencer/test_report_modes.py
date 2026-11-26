"""What kind each mode is and the range it can be tuned over (the
project's modes.yaml), the qubit table grouped per element, and the
picture of where every mode's target sits."""

from __future__ import annotations

import base64
import math

import pytest

from qdesignoptimizer.estimation.junction import squid_frequency_range
from qdesignoptimizer.sequencer.backends import qdo_modes, qdo_report
from qdesignoptimizer.sequencer.engine import report as rp

from .test_report import _partition_run, _qdo_run


def test_a_squid_tunes_down_from_zero_flux_by_its_asymmetry():
    # symmetric: all the way down
    assert squid_frequency_range(5.0e9, 0.2e9, 0.0) == (0.0, 5.0e9)
    low, high = squid_frequency_range(5.0e9, 0.2e9, 0.25)
    assert high == 5.0e9 and math.isclose(low, 5.2e9 * 0.5 - 0.2e9)
    with pytest.raises(ValueError):
        squid_frequency_range(5.0e9, 0.2e9, 1.5)


def test_modes_file_gives_kinds_and_ranges(tmp_path):
    assert qdo_modes.load(tmp_path) == {}
    (tmp_path / "modes.yaml").write_text(
        "coupler: {kind: squid, asymmetry: 0.25}\n"
        "linc_1: {kind: LINC, range: [6.5, 5.0]}\n",
        "utf-8",
    )
    modes = qdo_modes.load(tmp_path)
    assert modes["coupler"]["kind"] == "squid"
    assert modes["linc_1"] == {"kind": "linc", "range": (5.0e9, 6.5e9)}
    assert qdo_modes.band(modes, "linc_1", None, None) == (5.0e9, 6.5e9)
    assert qdo_modes.band(modes, "qubit_1", 4e9, 0.2e9) is None
    for bad in ("q: {kind: fluxonium}\n", "q: {range: [5]}\n", "q: [\n"):
        (tmp_path / "modes.yaml").write_text(bad, "utf-8")
        with pytest.raises(ValueError, match="modes.yaml"):
            qdo_modes.load(tmp_path)


def test_the_table_groups_each_element(tmp_path):
    run = _partition_run(tmp_path, [])
    modes = {"coupler": {"kind": "squid", "asymmetry": 0}}
    made = qdo_report.table([run], {"coupler": "4 nH"}, modes)
    display = made["display"]
    rows = [dict(zip(display["columns"], r)) for r in display["rows"]]
    coupler = rows[: next(n for n, r in enumerate(rows[1:], 1) if r["Element"])]
    assert coupler[0]["Element"] == "**coupler** (squid)"
    by_quantity = {r["Quantity"]: r for r in coupler}
    assert list(by_quantity)[:3] == ["Junction Lj", "Ej", "f01 at zero flux"]
    assert by_quantity["Tuning range"]["Simulated"].startswith("0.000–5.300")
    # the side note on the element's first row only
    assert coupler[0]["From study"] and not coupler[1]["From study"]
    # the flat paste has the kind and the range too
    flat = dict(zip(made["columns"], made["rows"][0]))
    assert flat["Kind"] == "squid" and flat["Tuning range (GHz)"] == "0.000–5.300"


def test_a_given_range_is_drawn_as_given(tmp_path):
    run = _partition_run(tmp_path, [])
    made = qdo_report.table(
        [run], {"coupler": "4 nH"}, {"coupler": {"kind": "linc", "range": (4e9, 6e9)}}
    )
    flat = dict(zip(made["columns"], made["rows"][0]))
    assert flat["Tuning range (GHz)"] == "4.000–6.000"
    # a given range is not rescaled by the junction as made
    assert flat["Tuning range from fab (GHz)"] == ""


def test_the_picture_places_every_target_and_the_report_shows_it(tmp_path):
    made = qdo_report.table([_qdo_run(tmp_path)], {})
    png = base64.b64decode(made["picture"]["png"])
    assert png[:4] == b"\x89PNG"
    folder = tmp_path / "report"
    folder.mkdir()
    assert rp.save_table_picture(made, folder / rp.TABLE_PNG)
    made["picture_src"] = rp.TABLE_PNG
    lines, _ = rp._head([], [], made, None)
    text = "\n".join(lines)
    # the picture comes first, the table after the design variables
    assert text.index("![Frequencies](frequencies.png)") < text.index(
        "| Element | Quantity"
    )
    design = {"project": "p", "purpose": "Why.", "variables": [], "_backend": None}
    lines, _ = rp._head([], [design], made, None)
    text = "\n".join(lines)
    assert text.index("Why.") < text.index("## Frequencies") < text.index("| Element |")
    # the paste stays a row per element
    assert rp._table_md(made, flat=True).startswith("| Element | Kind |")


def test_a_lane_per_mode_with_a_target_even_one_not_simulated(tmp_path):
    lanes = []

    def keep(given, title=""):
        lanes.extend(given)
        return None

    run = _qdo_run(tmp_path)
    (tmp_path / run["folder"] / "parameter_targets.yaml").write_text(
        "qubit_1_freq: 4.0e9\nresonator_9_freq: 7.0e9\n", "utf-8"
    )
    original = qdo_modes.picture
    qdo_modes.picture = keep
    try:
        qdo_report.table([run], {})
    finally:
        qdo_modes.picture = original
    by_mode = {lane["mode"]: lane for lane in lanes}
    assert by_mode["resonator_9"]["target"] == 7.0e9
    assert by_mode["resonator_9"]["kind"] == "resonator"
    assert by_mode["qubit_1"]["simulated"] == 4.055e9


def test_the_editor_lists_given_and_found_modes_and_saves_them(tmp_path):
    (tmp_path / "modes.yaml").write_text(
        "# kept on top\ncoupler: {kind: squid, asymmetry: 0.1, note: hand}\n",
        "utf-8",
    )
    made = qdo_modes.editor(tmp_path, {"qubit_1": "transmon", "res_1": "resonator"})
    rows = {r["mode"]: r for r in made["rows"]}
    assert rows["coupler"]["kind"] == "squid" and rows["coupler"]["asymmetry"] == 0.1
    assert rows["res_1"]["kind"] == "resonator" and rows["qubit_1"]["low"] == ""
    assert "ats" in made["kinds"] and not made["problem"]
    rows["qubit_1"].update(kind="ats", low=5.5, high="4.5")
    qdo_modes.save(tmp_path, list(rows.values()))
    text = (tmp_path / "modes.yaml").read_text("utf-8")
    assert text.startswith("# kept on top\n")
    modes = qdo_modes.load(tmp_path)
    assert modes["qubit_1"] == {"kind": "ats", "range": (4.5e9, 5.5e9)}
    assert modes["coupler"]["note"] == "hand"
    for bad in (
        {"mode": "q", "kind": "fluxonium"},
        {"mode": "q", "kind": "squid", "asymmetry": 2},
        {"mode": "q", "kind": "linc", "low": 4},
        {"mode": "q", "kind": "linc", "low": "four", "high": 5},
    ):
        with pytest.raises(ValueError, match="q:"):
            qdo_modes.save(tmp_path, [bad])


def test_the_modes_found_in_runs(tmp_path):
    run = _qdo_run(tmp_path)
    (tmp_path / run["folder"] / "parameter_targets.yaml").write_text(
        "resonator_9_freq: 7.0e9\n", "utf-8"
    )
    run["outputs"]["resonator_1_freq"] = 7.2e9  # has no Lj
    found = qdo_report.modes_in([run])
    assert found["qubit_1"] == "transmon"
    assert found["resonator_9"] == found["resonator_1"] == "resonator"
