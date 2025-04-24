import names as n

from qdesignoptimizer.sim_plot_progress import OptPltSet
from qdesignoptimizer.utils.names_parameters import param


def get_plot_settings_resonator(group: int):
    resonator = [n.RESONATOR_1, n.RESONATOR_2][group - 1]
    return {
        "FREQUENCIES": [
            OptPltSet(
                n.ITERATION,
                param(resonator, n.FREQ),
                y_label="Resonator Frequency",
                unit="GHz",
            )
        ]
    }
