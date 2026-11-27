"""The full report's pictures: the model as Ansys drew it in the last
iteration, and a close-up of each junction in the final GDS."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from qdesignoptimizer.design_analysis import DesignAnalysis
from qdesignoptimizer.sequencer.backends import qdo_record
from qdesignoptimizer.sequencer.engine import report as rp
from qdesignoptimizer.sim_capacitance_matrix import save_ansys_picture

PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 16


def _transmon_design():
    from qiskit_metal import designs
    from qiskit_metal.qlibrary.qubits.transmon_pocket import TransmonPocket

    design = designs.DesignPlanar()
    TransmonPocket(design, "Q1", options={"pos_x": "1mm", "pos_y": "-0.5mm"})
    return design


class _Modeler:
    """What save_ansys_picture asks of an Ansys 3D modeler."""

    def __init__(self, fail=False):
        self.fail = fail
        self.fitted = 0
        self.sizes = []
        self.selections = []
        # As Q3D draws one qubit: its substrate, ground plane and metal.
        self.groups = {
            "Solids": ["main"],
            "Sheets": ["cross_Q1", "connector_Q1", "ground_main_plane"],
        }
        self.colours = {"ground_main_plane": (143, 175, 143)}
        self.ground_at_export = []

    def GetPropertyValue(self, tab, name, prop):  # noqa: N802 -- Ansys's name
        # AEDT 2022 R2 gives one number, red in the lowest byte.
        r, g, b = self.colours[name]
        return str(r | g << 8 | b << 16)

    def ChangeProperty(self, args):  # noqa: N802 -- Ansys's name
        tab = args[1]
        name = tab[1][1]
        colour = tab[2][1]
        self.colours[name] = (colour[2], colour[4], colour[6])

    def GetObjectsInGroup(self, group):  # noqa: N802 -- Ansys's name
        if group not in self.groups:
            raise RuntimeError("no such group")
        return self.groups[group]

    def FitAll(self):  # noqa: N802 -- Ansys's name
        self.fitted += 1

    def ExportModelImageToFile(self, path, width, height, params):  # noqa: N802
        self.sizes.append((width, height))
        self.selections.append(params[params.index("Selections:=") + 1])
        self.ground_at_export.append(self.colours.get("ground_main_plane"))
        if self.fail:
            raise RuntimeError("The remote procedure call failed.")
        Path(path).write_bytes(PNG)


class _Renderer:
    """What save_ansys_picture asks of an Ansys renderer."""

    def __init__(self, fail=False):
        self.modeler = SimpleNamespace(_modeler=_Modeler(fail))
        self.design = SimpleNamespace(chips={"main": {}})


def test_an_ansys_picture_shows_the_drawn_metal_and_the_ground(tmp_path):
    renderer = _Renderer()
    path = str(tmp_path / "hfss_1.png")
    assert save_ansys_picture(renderer, path) == path
    modeler = renderer.modeler._modeler
    assert modeler.fitted == 1 and Path(path).read_bytes() == PNG
    # Never 0 x 0 (the window's size): that crashes AEDT run from a script.
    assert modeler.sizes and all(w > 0 and h > 0 for w, h in modeler.sizes)
    # Ansys draws only what is selected: the components and the ground
    # plane with its gaps, not the substrate or the box around the chip.
    assert modeler.selections == ["cross_Q1,connector_Q1,ground_main_plane"]
    assert save_ansys_picture(renderer, None) is None


def test_the_ground_plane_is_darker_in_the_picture_and_put_back(tmp_path):
    from qdesignoptimizer.sim_capacitance_matrix import GROUND_PICTURE_COLOUR

    for fail in (False, True):
        renderer = _Renderer(fail=fail)
        save_ansys_picture(renderer, str(tmp_path / "x.png"))
        modeler = renderer.modeler._modeler
        assert modeler.ground_at_export == [GROUND_PICTURE_COLOUR]
        assert modeler.colours["ground_main_plane"] == (143, 175, 143)


def test_a_model_of_only_the_chip_is_pictured_whole(tmp_path):
    renderer = _Renderer()
    renderer.modeler._modeler.groups = {"Solids": ["main"]}
    assert save_ansys_picture(renderer, str(tmp_path / "x.png"))
    assert renderer.modeler._modeler.selections == [""]


def test_a_colour_reads_in_either_form_ansys_gives():
    from qdesignoptimizer.sim_capacitance_matrix import _colour_of

    def says(text):
        return SimpleNamespace(GetPropertyValue=lambda *a: text)

    assert _colour_of(says("9416591"), "g") == (143, 175, 143)
    assert _colour_of(says("13482682"), "s") == (186, 186, 205)
    assert _colour_of(says("(10 20 30)"), "x") == (10, 20, 30)
    assert _colour_of(says("blue"), "x") is None


def test_a_failing_ansys_export_does_not_fail_the_simulation(tmp_path):
    assert save_ansys_picture(_Renderer(fail=True), str(tmp_path / "x.png")) is None


def test_the_sequencer_names_the_folder_while_a_study_runs(tmp_path):
    analysis = object.__new__(DesignAnalysis)
    analysis.optimization_results = [{}, {}]
    with qdo_record.ansys_pictures(tmp_path / "ansys"):
        assert DesignAnalysis.ansys_pictures_default == str(tmp_path / "ansys")
        analysis.ansys_pictures = DesignAnalysis.ansys_pictures_default
        path = analysis.ansys_picture_path(f"hfss_{analysis._iteration()}")
    assert DesignAnalysis.ansys_pictures_default is None
    assert Path(path) == tmp_path / "ansys" / "hfss_3.png"
    assert (tmp_path / "ansys").is_dir()
    analysis.ansys_pictures = None
    assert analysis.ansys_picture_path("hfss_1") is None


def test_the_report_shows_the_last_iterations_pictures(tmp_path):
    shots = tmp_path / qdo_record.ANSYS
    shots.mkdir()
    for name in ("hfss_1", "hfss_2", "hfss_10", "q3d_1_2", "q3d_1_1", "q3d_2_2"):
        (shots / f"{name}.png").write_bytes(PNG)
    layout = tmp_path / qdo_record.LAYOUT
    layout.mkdir()
    (layout / "jj_Q1.png").write_bytes(PNG)
    (layout / "captions.txt").write_text("jj_Q1.png: Q1, drawn from final.gds\n")
    pictures = qdo_record.final_pictures(tmp_path)
    assert [Path(p["path"]).name for p in pictures] == [
        "hfss_10.png",
        "q3d_1_2.png",
        "q3d_2_2.png",
        "jj_Q1.png",
    ]
    assert pictures[0]["title"] == "HFSS eigenmode model (iteration 10)"
    assert pictures[-1]["caption"] == "Q1, drawn from final.gds"
    assert qdo_record.final_pictures(tmp_path / "nothing") == []


def test_the_final_layout_and_a_close_up_of_each_junction(tmp_path):
    design = _transmon_design()
    (junction,) = qdo_record.junctions(design)
    assert junction["component"] == "Q1"
    assert junction["x"] == pytest.approx(1000) and junction["y"] == pytest.approx(-500)
    written = qdo_record.save_layout(tmp_path, design)
    layout = tmp_path / qdo_record.LAYOUT
    assert [Path(p).name for p in written] == ["final.gds", "jj_Q1.png"]
    assert (layout / "jj_Q1.png").read_bytes()[:8] == PNG[:8]
    # No junction cell to put in (the renderer's path_filename): said so.
    caption = (layout / "captions.txt").read_text()
    assert "no junction cell" in caption


def test_a_junction_cell_in_the_gds_is_found(tmp_path):
    import gdstk

    library = gdstk.Library(unit=1e-3)
    cell = library.new_cell("TOP")
    cell.add(gdstk.rectangle((0.99, -0.51), (1.01, -0.49)))
    gds = tmp_path / "chip.gds"
    library.write_gds(str(gds))
    junction = {
        "component": "Q1",
        "x": 1000.0,
        "y": -500.0,
        "ends": [(1000.0, -515.0), (1000.0, -485.0)],
        "length_um": 30.0,
        "width_um": 0.2,
    }
    assert qdo_record.plot_junction(gds, junction, tmp_path / "jj.png")
    assert not qdo_record.plot_junction(gds, junction, tmp_path / "jj2.png", skip={0})


def test_the_last_iterations_design_variables(tmp_path):
    (tmp_path / "design_variables.json").write_text(
        json.dumps({"a": "1 um", "b": "2 um"})
    )
    assert qdo_record.final_variables(tmp_path) is None  # no results file
    results = {
        "optimization_results": [
            {"design_variables": {"a": "3 um"}},
            {"design_variables": {"a": "4 um"}},
        ]
    }
    np.save(tmp_path / "chip_20260101-000000", np.array([results]), allow_pickle=True)
    assert qdo_record.final_variables(tmp_path) == {"a": "4 um", "b": "2 um"}


def test_a_layout_that_cannot_be_built_is_tried_once(tmp_path):
    from qdesignoptimizer.sequencer.backends.qdo import QDOBackend

    (tmp_path / "design_variables.json").write_text(json.dumps({"a": "1 um"}))
    results = {"optimization_results": [{"design_variables": {"a": "3 um"}}]}
    np.save(tmp_path / "chip_20260101-000000", np.array([results]), allow_pickle=True)
    built = []

    def design(variables):
        built.append(variables)
        raise RuntimeError("no such component")

    backend = SimpleNamespace(project=SimpleNamespace(design=design))
    assert QDOBackend.report_pictures(backend, tmp_path) == []
    assert QDOBackend.report_pictures(backend, tmp_path) == []
    assert built == [{"a": "3 um"}]  # the second report did not build it
    marker = tmp_path / qdo_record.LAYOUT / qdo_record.NO_LAYOUT
    assert "no such component" in marker.read_text()
    marker.unlink()
    QDOBackend.report_pictures(backend, tmp_path)
    assert len(built) == 2  # deleted: tried again


def _run(folder: Path, pictures: list[dict]) -> dict:
    return {
        "key": "x-1",
        "project": "p",
        "folder": folder,
        "id": "x-1",
        "task": "Q1.study",
        "subject": "Q1",
        "status": "done",
        "message": "",
        "started": "2026-11-20 10:00:00",
        "finished": "2026-11-20 11:00:00",
        "campaign": "",
        "sequence": {},
        "settings": {},
        "notes": "",
        "why": "",
        "added_by": "",
        "outputs": {},
        "updates": {},
        "committed": {},
        "checks": {},
        "targets": [],
        "pictures": pictures,
        "_backend": None,
    }


def test_the_written_report_carries_its_pictures(tmp_path):
    run_folder = tmp_path / "completed" / "x-1_2026-11-20T10-00-00"
    (run_folder / "ansys").mkdir(parents=True)
    shot = run_folder / "ansys" / "hfss_2.png"
    shot.write_bytes(PNG)
    run = _run(
        run_folder,
        [
            {"title": "HFSS", "path": shot, "caption": "last"},
            {"title": "gone", "path": run_folder / "missing.png"},
        ],
    )
    project = SimpleNamespace(out=tmp_path / "out")
    folder = rp.write(project, [run], None)
    copied = folder / "pictures" / run_folder.name / "hfss_2.png"
    assert copied.read_bytes() == PNG
    text = (folder / "report.md").read_text("utf-8")
    assert f"![HFSS](pictures/{run_folder.name}/hfss_2.png)" in text
    assert "*last*" in text and "gone" not in text
    # The GUI shows the pictures itself, below the text.
    assert "### Pictures" not in rp.full([run], pictures=False)
    assert "### Pictures" in rp.full([run])


def test_same_named_pictures_of_a_run_are_copied_apart(tmp_path):
    """Review 2026-11-22: two pictures with one file name were copied to
    one file, and both showed the second."""
    run_folder = tmp_path / "completed" / "x-1_2026-11-20T10-00-00"
    for sub, data in (("a", PNG), ("b", PNG + b"b")):
        (run_folder / sub).mkdir(parents=True)
        (run_folder / sub / "model.png").write_bytes(data)
    pictures = [
        {"title": "A", "path": run_folder / "a" / "model.png"},
        {"title": "B", "path": run_folder / "b" / "model.png"},
    ]
    folder = rp.write(
        SimpleNamespace(out=tmp_path / "out"), [_run(run_folder, pictures)], None
    )
    copies = folder / "pictures" / run_folder.name
    assert (copies / "1_model.png").read_bytes() == PNG
    assert (copies / "2_model.png").read_bytes() == PNG + b"b"


def test_a_layout_marker_that_cannot_be_written_keeps_the_pictures(tmp_path):
    """Review 2026-11-22: a failed marker write lost all the run's pictures."""
    from qdesignoptimizer.sequencer.backends.qdo import QDOBackend

    (tmp_path / "design_variables.json").write_text(json.dumps({"a": "1 um"}))
    results = {"optimization_results": [{"design_variables": {"a": "3 um"}}]}
    np.save(tmp_path / "chip_20260101-000000", np.array([results]), allow_pickle=True)
    (tmp_path / qdo_record.ANSYS).mkdir()
    (tmp_path / qdo_record.ANSYS / "hfss_1.png").write_bytes(PNG)
    (tmp_path / qdo_record.LAYOUT).write_text("a file, so no folder fits here")

    def design(variables):
        raise RuntimeError("no such component")

    backend = SimpleNamespace(project=SimpleNamespace(design=design))
    pictures = QDOBackend.report_pictures(backend, tmp_path)
    assert [Path(p["path"]).name for p in pictures] == ["hfss_1.png"]


def test_the_report_says_why_a_study_has_no_final_layout(tmp_path, monkeypatch):
    from qdesignoptimizer.sequencer.backends import qdo, qdo_preview

    monkeypatch.setattr(qdo_preview, "runs", lambda study: [])
    monkeypatch.setattr(qdo, "_passes", lambda study: {})
    backend = SimpleNamespace(_study_in=lambda folder: None)
    assert "layout" not in qdo.QDOBackend.report_simulation(backend, tmp_path)
    (tmp_path / qdo_record.LAYOUT).mkdir()
    (tmp_path / qdo_record.LAYOUT / qdo_record.NO_LAYOUT).write_text(
        "Building the final layout failed: boom\nDelete this file.\n"
    )
    simulation = qdo.QDOBackend.report_simulation(backend, tmp_path)
    assert simulation["layout"].startswith("Building the final layout failed: boom")
    assert qdo_record.NO_LAYOUT in simulation["layout"]
    run = {**_run(tmp_path, []), "simulation": simulation}
    assert "**No final layout:** Building" in rp.full([run])
