from src.qdesignoptimizer.design_analysis_types import MiniStudy
from src.utils.utils_design_variables import junction_setup
import src.utils.utils_design_variables as u
import design_variables as dv 
import src.utils.constants as dc

CONVERGENCE = dict(nbr_passes = 7, delta_f = 0.03)

def get_mini_study_qb_res(branch: int):
    return MiniStudy(
        component_names = [
            u.name_qb(branch),
            u.name_res(branch),
            u.name_tee(branch)
        ],
        port_list =  [(u.name_tee(branch),'prime_end', 50),
                    (u.name_tee(branch),'prime_start', 50),],
        open_pins = [],
        mode_freqs = [
            (str(branch), dc.QUBIT_FREQ),
            (str(branch), dc.RES_FREQ),
        ],
        jj_var = dv.JUNCTION_VARS,
        jj_setup = {**junction_setup(u.name_qb(branch))},
        design_name = "get_mini_study_qb_res",
        adjustment_rate = 0.8,
        **CONVERGENCE
    )