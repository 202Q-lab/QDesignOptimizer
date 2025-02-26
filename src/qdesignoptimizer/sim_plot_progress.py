import os
from itertools import cycle

import numpy as np
from matplotlib import pyplot as plt

import qdesignoptimizer.utils.constants as c
from qdesignoptimizer.utils.names_design_variables import name_mode
from qdesignoptimizer.utils.names_parameters import param
from qdesignoptimizer.utils.utils import get_value_and_unit

DEFAULT_PLT_SET = {
    #     dc.RES_FREQ: {"label": "Res freq (Hz)"},
    #     dc.RES_KERR: {"label": "Res Kerr (Hz)"},
    #     dc.RES_KAPPA: {"label": "Res kappa (Hz)"},
    #     dc.QUBIT_FREQ: {"label": "Qubit freq (Hz)"},
    #     dc.QUBIT_ANHARMONICITY: {"label": "Qubit anharm (Hz)"},
    #     dc.RES_QUBIT_CHI: {"label": "Res-qubit chi (Hz)"},
    #     dc.CAVITY_FREQ: {"label": "Cavity freq (Hz)"},
    #     dc.CAVITY_QUBIT_CHI: {"label": "Cavity-qubit chi (Hz)"},
    #     dc.CAVITY_RES_CROSS_KERR: {"label": "Cavity-res cross-Kerr (Hz)"},
    #     dc.ITERATION: {"label": "Iteration"},
    #     # Any other will default to the key name
}


class OptPltSet:
    def __init__(self, x: str, y: str, x_label: str = None, y_label: str = None):
        """Set the plot settings for a progress plots of the optimization framework

        Args:
            x (str): The x-axis parameter
            y (str): The y-axis parameter
            x_label (str, optional): The x-axis label. None will use the default label from DEFAULT_PLT_SET.
            y_label (str, optional): The y-axis label. None will use the default label from DEFAULT_PLT_SET.
        """
        self.x = x
        self.y = y
        self.x_label = self._get_label(x, x_label)
        self.y_label = self._get_label(y, y_label)

    def _get_label(self, variable: str, x_label: str):
        if x_label is not None:
            return x_label
        elif variable in DEFAULT_PLT_SET:
            return DEFAULT_PLT_SET[variable]["label"]
        else:
            return variable


def plot_progress(
    opt_results: dict,
    system_target_params: dict,
    plot_settings: dict,
    plot_branches_separately: bool = False,
    plot_option: str = "linear",
    block_plots: bool = False,
):
    """Plot the progress of the optimization framework

    Args:
        opt_results (dict): The optimization results
        system_target_params (dict): The target system parameters
        plot_settings (dict): The plot settings, example
            {"plt_name": [OptPltSet("panel1_x", "panel1_y"), OptPltSet("panel2_x", "panel2_y")], "plt_name2": ...}
        plot_option (str, optional): 'linear', 'log', 'loglog'.
    """

    def get_data_from_parameter(axes_parameter: str, result: dict, ii: int):
        if axes_parameter == c.ITERATION:
            data_opt = ii + 1
        elif axes_parameter in result["system_optimized_params"]:
            data_opt = result["system_optimized_params"][axes_parameter]
        elif axes_parameter in result["system_optimized_params"]:
            data_opt = result["system_optimized_params"][axes_parameter]
        elif axes_parameter in result["design_variables"]:
            data_opt = get_value_and_unit(result["design_variables"][axes_parameter])[0]
        else:
            data_opt = None
        return data_opt

    def plot_figure(
        opt_results: dict,
        system_target_params: dict,
        panels: list,
        axs: list,
        colors: cycle,
        plot_option: str,
    ) -> bool:
        """Plot all panels in the figure

        Args:
            opt_results (dict): The optimization results
            system_target_params (dict): The target system parameters
            panels (list): The list of OptPltSet objects
            axs (list): The list of axes, one for each panel in panels
            colors (cycle): The cycle of colors
            plot_option (str): 'linear', 'log', 'loglog'

        Returns:
            bool: True if data was plotted, used to delete empty figures
        """
        data_plotted = False
        for idx, panel in enumerate(panels):

            if len(panels) == 1:
                axes = axs
            else:
                axes = axs[idx]
            if axes.get_legend() is not None:
                axes.get_legend().remove()
            color = next(colors)
            x_data_opt = [
                get_data_from_parameter(panel.x, result, ii)
                for ii, result in enumerate(opt_results)
            ]
            y_data_opt = [
                get_data_from_parameter(panel.y, result, ii)
                for ii, result in enumerate(opt_results)
            ]
            if all(element is not None for element in y_data_opt):
                data_plotted = True
            print(idx, panel, x_data_opt, y_data_opt)
            axes.plot(x_data_opt, y_data_opt, "o-", label=f"optimized", color=color)

            if (
                panel.y in system_target_params
                and (not None in x_data_opt)
                and (not None in y_data_opt)
            ):
                y_data_target = system_target_params[panel.y]
                axes.plot(
                    [min(x_data_opt), max(x_data_opt)],
                    [y_data_target, y_data_target],
                    "--" if len(x_data_opt) and len(x_data_opt) > 1 else "*",
                    color=color,
                    label=f"target",
                )

            axes.legend(["optimized", "target"])
            axes.set_xlabel(panel.x_label)
            axes.set_ylabel(panel.y_label)
            if plot_option == "log":
                axes.set_yscale("log")
            if plot_option == "loglog":
                axes.set_xscale("log")
                axes.set_yscale("log")
        return data_plotted

    plt.close("all")
    colors = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    for plot_name, panels in plot_settings.items():
        if not plot_branches_separately:
            fig, axs = plt.subplots(len(panels))
        data_plotted_in_any_branch = False
        if plot_branches_separately:
            fig, axs = plt.subplots(len(panels))
        data_plotted = plot_figure(
            opt_results,
            system_target_params,
            panels,
            axs,
            colors,
            plot_option,
        )
        data_plotted_in_any_branch = data_plotted_in_any_branch or data_plotted
        fig.suptitle(plot_name)
        fig.subplots_adjust(hspace=0.5)

        if plot_branches_separately:
            if not data_plotted:
                "plot_branches_separately: No data to plot for this figure"
                plt.close(fig)

        if not plot_branches_separately:
            if not data_plotted_in_any_branch:
                "NOT plot_branches_separately: No data to plot for this figure"
                plt.close(fig)
    plt.show(block=block_plots)


PLOT_SETTINGS_EXAMPLE = {
    "RES": [
        OptPltSet(
            c.ITERATION,
            param(name_mode("resonator"), "freq"),
            "Edited label Y",
            "Edited label X",
        ),
    ],
    "QUBIT": [
        OptPltSet(
            c.ITERATION,
            param(name_mode("qubit"), "freq"),
            "Edited label Y",
            "Edited label X",
        ),
    ],
}

if __name__ == "__main__":
    from pprint import pprint

    SAVE_PATH = r"C:\qiskit-metal\projects\continuous_variables\pbc_planar_boson_coupler\design_v5"
    FILE = r"pbc_v5_20240829-153113.npy"

    loaded_data = np.load(os.path.join(SAVE_PATH, FILE), allow_pickle=True)
    simulation = loaded_data[0]
    pprint(simulation["optimization_results"][-1]["design_variables"])
    plot_progress(
        simulation["optimization_results"],
        simulation["system_target_params"],
        simulation["plot_settings"],
        plot_option="linear",
        block_plots=True,
    )
