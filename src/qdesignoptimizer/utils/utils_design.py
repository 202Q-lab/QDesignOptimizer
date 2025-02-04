from qiskit_metal import MetalGUI
from qiskit_metal.designs.design_planar import DesignPlanar


def create_chip_base(
    chip_name: str, chip_type: dict, open_gui: bool = True
) -> tuple[DesignPlanar, MetalGUI]:
    design = DesignPlanar({}, True)
    design.chip_name = chip_name
    design.chips.main.material = "silicon"
    design.chips.main.size.size_x = chip_type["size_x"]
    design.chips.main.size.size_y = chip_type["size_y"]
    design.chips.main.size.size_z = chip_type["size_z"]
    design.overwrite_enabled = True
    design.render_mode = "simulate"

    gui = MetalGUI(design) if open_gui else None

    return design, gui
