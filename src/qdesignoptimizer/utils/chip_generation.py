"""Tools for creating and configuring basic chips in Qiskit Metal designs."""

from dataclasses import dataclass

from qiskit_metal import MetalGUI
from qiskit_metal.designs.design_planar import DesignPlanar


@dataclass
class ChipType:
    """Define physical dimensions and material for a designed chip."""

    material: str
    size_x: str
    size_y: str
    size_z: str


def create_chip_base(
    chip_name: str, chip_type: ChipType, open_gui: bool = True
) -> tuple[DesignPlanar, MetalGUI]:
    """
    Create and return a basic Qiskit Metal planar chip design.

    Args:
        chip_name (str): The name to assign to the chip design.
        chip_type (ChipType): The physical dimensions and material of the chip.
        open_gui (bool, optional): Whether to open the Qiskit Metal GUI. Defaults to True.

    Returns:
        tuple[DesignPlanar, MetalGUI]: A tuple containing:
            - The initialized planar design object.
            - The Metal GUI instance if open_gui is True, otherwise None.

    Example:
        >>> chip_dimensions = ChipType(size_x="10mm", size_y="10mm", size_z="0.5mm")
        >>> design, gui = create_chip_base("my_quantum_chip", chip_dimensions)
    """
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
