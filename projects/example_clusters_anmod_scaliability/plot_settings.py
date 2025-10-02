from qdesignoptimizer.sim_plot_progress import OptPltSet
from qdesignoptimizer.utils.names_parameters import (
    ITERATION,
    mode,
    param,
)


def get_plot_settings(n_clusters, m_params_per_cluster):
    settings = {
        "SCALED": [
            OptPltSet(
                ITERATION,
                param(mode(f"{i},{j}"), ""),
                y_label=f"Mode {i, j}",
                unit="arb.",
            )
            for i in range(2)
            for j in range(2)
        ]
    }
    # "SCALED": [
    #     OptPltSet(
    #         ITERATION, param(mode(f"{i},{j}"), ""), y_label=f"Mode {i, j}", unit="arb."
    #     )
    # for i in range(n_clusters) for j in range(m_params_per_cluster)]
    # }
    return settings
