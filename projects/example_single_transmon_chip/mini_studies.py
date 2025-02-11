import json
with open('design_variables.json') as in_file:
    dv = json.load(in_file)

import qdesignoptimizer.utils.constants as dc
import qdesignoptimizer.utils.utils_design_variables as u
from qdesignoptimizer.design_analysis_types import MiniStudy
from qdesignoptimizer.utils.utils_design_variables import junction_setup

CONVERGENCE = dict(nbr_passes=7, delta_f=0.03)


def get_mini_study_qb_res(branch: int):
    return MiniStudy(
        component_names=[u.name_qb(branch), u.name_res(branch), u.name_tee(branch)],
        port_list=[
            (u.name_tee(branch), "prime_end", 50),
            (u.name_tee(branch), "prime_start", 50),
        ],
        open_pins=[],
        mode_freqs=[
            (str(branch), dc.QUBIT_FREQ),
            (str(branch), dc.RES_FREQ),
        ],
        jj_var=dv,
        jj_setup={**junction_setup(u.name_qb(branch))},
        hfss_wire_bond_size = 2, 
        hfss_wire_bond_offset = '0um', 
        hfss_wire_bond_threshold = '300um', 
        design_name="get_mini_study_qb_res",
        adjustment_rate=0.8,
        **CONVERGENCE
    )
