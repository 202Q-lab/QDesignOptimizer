"""The full report's study flow: built from what the runs recorded only."""

from __future__ import annotations

from types import SimpleNamespace

from qdesignoptimizer.sequencer.engine import flow as fl
from qdesignoptimizer.sequencer.engine import report as rp


def _run(
    rid,
    started,
    *,
    task="Q1.study",
    status="done",
    targets=(),
    ran_with=None,
    committed=None,
    runs=("HFSS eigenmode",),
    passes=15,
    folder=None,
):
    return {
        "key": rid,
        "project": "p",
        "folder": folder or f"/x/{rid}",
        "id": rid,
        "task": task,
        "subject": "Q1",
        "status": status,
        "message": "",
        "started": started,
        "finished": started,
        "campaign": "",
        "sequence": {},
        "settings": {},
        "notes": "",
        "why": "",
        "added_by": "",
        "outputs": {},
        "updates": {},
        "committed": committed or {},
        "checks": {},
        "targets": [
            {"parameter": p, "design_var": v, "target": 1.0, "values": vals}
            for p, v, vals in targets
        ],
        "pictures": [],
        "simulation": {"runs": list(runs), "passes": passes},
        "ran_with": ran_with or {},
        "_backend": None,
    }


def _three_studies():
    kappa = _run(
        "kappa",
        "2026-11-20 10:00:00",
        task="Q1.resonator_kappa",
        runs=("Q3D capacitance: ResonatorDecayIntoWaveguideStudy",),
        passes=8,
        targets=[("resonator_1_kappa", "design_var_coupl_tee", [0.5, 0.98])],
        ran_with={"design_var_coupl_tee": "400um", "design_var_lj": "10nH"},
        committed={"design_var_coupl_tee": "799.2 um"},
    )
    eigen = _run(
        "eigen",
        "2026-11-20 11:00:00",
        targets=[
            ("qubit_1_freq", "design_var_lj", [1.2, 1.01]),
            ("resonator_1_kappa", "design_var_coupl_tee", [0.9, 1.0]),
        ],
        ran_with={"design_var_coupl_tee": "799.2um", "design_var_lj": "10nH"},
        committed={"design_var_lj": "12 nH", "design_var_coupl_tee": "810 um"},
    )
    # Started from eigen's lj, and from a tee length changed by hand.
    fine = _run(
        "fine",
        "2026-11-20 12:00:00",
        status="failed",
        targets=[("qubit_1_freq", "design_var_lj", [1.002])],
        ran_with={"design_var_coupl_tee": "900 um", "design_var_lj": "12.0 nH"},
    )
    return kappa, eigen, fine


def test_an_arrow_carries_what_a_study_started_from_as_another_wrote_it():
    kappa, eigen, fine = _three_studies()
    # The order the runs started in, whatever the order asked for.
    graph = fl.graph([fine, kappa, eigen])
    assert [n["title"] for n in graph["nodes"]] == [
        "Store before these studies",
        "kappa",
        "eigen",
        "fine",
    ]
    edges = {(e["source"], e["target"]): e["variables"] for e in graph["edges"]}
    assert edges == {
        # What a study tunes and no study here wrote: from the store.
        ("start", "s1"): ["design_var_coupl_tee"],
        ("start", "s2"): ["design_var_lj"],
        ("s1", "s2"): ["design_var_coupl_tee"],  # "799.2 um" == "799.2um"
        ("s2", "s3"): ["design_var_lj"],  # 12 nH == 12.0 nH
        # fine's tee length is not eigen's (810 um): changed by hand.
    }
    assert graph == fl.graph([kappa, eigen, fine])  # the same, every time


def test_a_box_says_what_ran_and_what_each_parameter_tuned():
    kappa, eigen, fine = _three_studies()
    nodes = fl.graph([kappa, eigen, fine])["nodes"]
    assert nodes[1]["lines"][:2] == [
        "Q1.resonator_kappa · done",
        "Q3D capacitance: ResonatorDecayIntoWaveguideStudy · 8 passes · 2 iterations",
    ]
    assert "resonator_1_kappa → coupl_tee ✓  (-2.0 % off)" in nodes[1]["lines"]
    assert nodes[3]["lines"][-1] == "wrote nothing back"


def test_the_mermaid_flowchart():
    text = fl.mermaid(fl.graph(list(_three_studies())))
    assert text.startswith("flowchart TD\n")
    assert '    s1 -->|"coupl_tee"| s2' in text
    assert "class s3 failed;" in text


def test_a_long_hand_over_names_the_first_variables():
    names = [f"design_var_{i:02d}" for i in range(9)]
    assert fl._arrow_label(names, ", ").endswith("05, +3 more")


def test_the_picture(tmp_path):
    path = fl.draw(fl.graph(list(_three_studies())), tmp_path / "flow.png")
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_the_written_report_shows_the_flow(tmp_path):
    runs = [{**r, "folder": tmp_path / "completed" / r["id"]} for r in _three_studies()]
    folder = rp.write(SimpleNamespace(out=tmp_path / "out"), runs, None)
    text = (folder / "report.md").read_text("utf-8")
    assert "## Study flow" in text and "![Study flow](study_flow.png)" in text
    assert "```mermaid\nflowchart TD" in text
    assert (folder / rp.FLOW_PNG).is_file()
    # The GUI shows the picture itself.
    assert "## Study flow" not in rp.full(runs, flow=False)


def test_a_replaced_run_no_study_built_on_is_left_out():
    failed = _run("q_r1", "2026-11-20 09:00:00", task="Q1.q", status="failed")
    # Wrote lj back, then was run again before any study used it.
    unused = _run(
        "q_r2",
        "2026-11-20 10:00:00",
        task="Q1.q",
        committed={"design_var_lj": "11 nH"},
    )
    used = _run(
        "q_r3",
        "2026-11-20 11:00:00",
        task="Q1.q",
        committed={"design_var_lj": "12 nH"},
    )
    # Another study started from q_r3's value: q_r3 stays.
    other = _run(
        "kappa",
        "2026-11-20 12:00:00",
        task="Q1.kappa",
        ran_with={"design_var_lj": "12 nH"},
    )
    last = _run(
        "q_r4",
        "2026-11-20 13:00:00",
        task="Q1.q",
        status="failed",
    )
    graph = fl.graph([failed, unused, used, other, last])
    assert [n["title"] for n in graph["nodes"][1:]] == ["q_r3", "kappa", "q_r4"]
    assert graph["left_out"] == ["q_r1", "q_r2"]
    assert {(e["source"], e["target"]) for e in graph["edges"]} == {("s1", "s2")}
    assert "q_r1, q_r2" in fl.left_out_text(graph)
    # The last run of each study stays, used or not.
    assert fl.graph([failed])["left_out"] == []


def test_the_report_names_what_the_flow_leaves_out():
    runs = [
        _run("q", "2026-11-20 09:00:00", status="failed"),
        _run("q_r2", "2026-11-20 10:00:00"),
    ]
    text = rp.full(runs)
    assert "*Not shown (replaced by a later run" in text and ": q*" in text


def test_a_card_shows_the_model_as_ansys_drew_it(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # A white picture with the model in the middle and the logo in the corner.
    picture = np.ones((100, 160, 3))
    picture[40:60, 50:110] = 0.2
    picture[2:6, 150:158] = 0.0
    cropped = fl._crop(picture)
    # The model, 3 px (2 % of the long side) around it; the logo left out.
    assert cropped.shape[:2] == (26, 66)

    shot = tmp_path / "ansys" / "hfss_3.png"
    shot.parent.mkdir()
    plt.imsave(shot, picture)
    kappa, eigen, fine = _three_studies()
    eigen["pictures"] = [
        {"title": "Junction", "path": tmp_path / "layout" / "jj_Q1.png"},
        {"title": "HFSS", "path": shot},
    ]
    node = fl.graph([kappa, eigen, fine])["nodes"][2]
    assert node["image"] == str(shot)
    assert node["kind"] == "HFSS eigenmode"
    assert node["tuned"][0] == {
        "variable": "lj",
        "change": "",
        "written": True,
        "parameter": "qubit_1_freq",
        "off": "+1.0 %",
    }
    path = fl.draw(fl.graph([kappa, eigen, fine]), tmp_path / "flow.png")
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# -- stress cases ------------------------------------------------------------


def _chain(count, subjects=1):
    """``count`` studies taking turns over ``subjects`` subjects, each
    starting from what the last study of its subject wrote."""
    runs, store = [], {}
    for i in range(count):
        subject = f"Q{i % subjects}"
        name = f"design_var_width_{subject}"
        start = store.get(name, "1 um")
        written = f"{i + 2} um"
        runs.append(
            _run(
                f"r{i:02d}",
                f"2026-11-20 10:{i:02d}:00",
                task=f"{subject}.study_{i}",
                targets=[(f"{subject}_freq", name, [1.0, 1.01])],
                ran_with={name: start},
                committed={name: written},
            )
        )
        store[name] = written
    return runs


def test_the_same_runs_in_any_order_give_the_same_diagram(tmp_path):
    import hashlib
    import random

    runs = _chain(9, subjects=3)
    shuffled = runs[:]
    random.Random(1).shuffle(shuffled)
    assert fl.mermaid(fl.graph(runs)) == fl.mermaid(fl.graph(shuffled))
    pictures = [
        fl.draw(fl.graph(r), tmp_path / f"flow{i}.png").read_bytes()
        for i, r in enumerate([runs, shuffled])
    ]
    assert hashlib.sha256(pictures[0]).digest() == hashlib.sha256(pictures[1]).digest()


def test_arrows_passing_the_same_card_take_their_own_lanes(tmp_path):
    # Four subjects taking turns: each study starts from the one four back.
    flow = fl.graph(_chain(12, subjects=4))
    keys = [n["key"] for n in flow["nodes"]]
    spans = [(keys.index(e["source"]), keys.index(e["target"])) for e in flow["edges"]]
    lanes = fl._lanes(spans)
    assert set(lanes) == {i for i, (a, b) in enumerate(spans) if b - a > 1}
    for i in lanes:
        for j in lanes:
            (a, b), (c, d) = spans[i], spans[j]
            if i != j and a < d and c < b:  # both pass some card
                assert lanes[i] != lanes[j]
    assert max(lanes.values()) == 3  # four arrows overlap at most
    fl.draw(flow, tmp_path / "flow.png")


def test_a_card_without_a_usable_picture_gets_the_placeholder(tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.axes
    import matplotlib.pyplot as plt
    import numpy as np

    model = np.ones((100, 160, 3))
    model[40:60, 50:110] = 0.2
    good = tmp_path / "good.png"
    plt.imsave(good, model)
    blank = tmp_path / "blank.png"
    plt.imsave(blank, np.ones((100, 160, 3)))
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not a png")
    assert fl._crop(np.ones((100, 160, 3))) is None

    runs = _chain(4)
    runs.append(_run("no_task", "2026-11-20 11:00:00", task="", targets=[]))
    flow = fl.graph(runs)
    pictures = [good, blank, broken, tmp_path / "missing.png"]
    for node, picture in zip(flow["nodes"][1:], pictures):
        node["image"] = str(picture)

    shown, placeholders = [], []
    imshow, text = matplotlib.axes.Axes.imshow, matplotlib.axes.Axes.text

    def spy_imshow(self, data, *args, **kwargs):
        shown.append(data.shape)
        return imshow(self, data, *args, **kwargs)

    def spy_text(self, x, y, s, *args, **kwargs):
        if str(s).startswith("no model picture"):
            placeholders.append(s)
        return text(self, x, y, s, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "imshow", spy_imshow)
    monkeypatch.setattr(matplotlib.axes.Axes, "text", spy_text)
    assert fl.draw(flow, tmp_path / "flow.png").is_file()
    # Only the good picture is drawn, cropped to the model; the blank,
    # broken and missing ones and the run with none get the placeholder.
    assert shown == [(26, 66, 4)]
    assert len(placeholders) == 4
    assert flow["nodes"][-1]["subtitle"].startswith("HFSS eigenmode")


def test_a_run_without_a_start_time_comes_last():
    runs = _chain(3)
    runs[0]["started"] = runs[0]["finished"] = None
    flow = fl.graph(runs)
    assert [n["title"] for n in flow["nodes"][1:]] == ["r01", "r02", "r00"]
    # It ran last as far as the flow knows: no arrow leaves it.
    assert all(e["source"] != "s3" for e in flow["edges"])
