import design_variables as dv

import qdesignoptimizer.utils.constants as dc
import qdesignoptimizer.utils.utils_design_variables as u
from qdesignoptimizer.design_analysis_types import MiniStudy, ScatteringStudy
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
        jj_var=dv.JUNCTION_VARS,
        jj_setup={**junction_setup(u.name_qb(branch))},
        design_name="get_mini_study_qb_res",
        adjustment_rate=0.8,
        **CONVERGENCE
    )

def get_mini_study_scattering_analysis(branch :int):
    return MiniStudy(
        component_names=[ u.name_res(branch), u.name_tee(branch)],
        port_list=[
            (u.name_tee(branch), "prime_end", 50),
            (u.name_tee(branch), "prime_start", 50),
        ],
        open_pins=[],
        mode_freqs=[

            (str(branch), dc.RES_FREQ),
        ],
        design_name="get_mini_study_scattering_analysis",
        adjustment_rate=0.8,
        scattering_studies = ScatteringStudy(nbr_passes = 20,
                                             max_delta_s =0.005,
                                             basis_order = -1),

        **CONVERGENCE
    )