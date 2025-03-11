"""Tools for creating and configuring basic chips in Qiskit Metal designs."""

from dataclasses import dataclass

from qiskit_metal import MetalGUI
from qiskit_metal.designs.design_planar import DesignPlanar


@dataclass
class ChipType:
    """Define sizes for designed chip."""

    material: str
    size_x: str
    size_y: str
    size_z: str


def create_chip_base(
    chip_name: str, chip_type: ChipType, open_gui: bool = True
) -> tuple[DesignPlanar, MetalGUI]:
    """Return basic qiskit-metal chip design."""
    design = DesignPlanar({}, True)
    design.chip_name = chip_name
    design.chips.main.material = chip_type.material
    design.chips.main.size.size_x = chip_type.size_x
    design.chips.main.size.size_y = chip_type.size_y
    design.chips.main.size.size_z = chip_type.size_z
    design.overwrite_enabled = True
    design.render_mode = "simulate"

    gui = None
    if open_gui:
        gui = MetalGUI(design)
        gui.toggle_docks()

    return design, gui
