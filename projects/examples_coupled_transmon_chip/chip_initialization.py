import json
# Importing files for design assembly
import design as d
import names as n

from qdesignoptimizer.utils.chip_generation import create_chip_base, ChipType

OPEN_GUI = True

import os
os.makedirs(os.path.dirname("out/"), exist_ok=True)

def create_chip_and_gui():
    chip_type = ChipType(size_x="10mm", size_y="10mm", size_z="-300um", material="silicon")

    # Creating design and gui
    design, gui = create_chip_base(
        chip_name=n.CHIP_NAME, chip_type=chip_type, open_gui=OPEN_GUI
        )

    # Introducing variables for the design
    with open("design_variables.json") as in_file:
        initial_design_variables = json.load(in_file)
    n.add_design_variables_to_design(design, initial_design_variables)
    
    return design, gui