# type: ignore

"""Visualization utilities for tracking optimization progress of quantum circuit designs."""

import time
from itertools import cycle
from typing import List, Literal, Optional, Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from qdesignoptimizer.design_analysis_types import OptTarget
from qdesignoptimizer.utils.names_parameters import (
    CAPACITANCE,
    ITERATION,
    NONLIN,
    param,
    param_capacitance,
    param_nonlin,
)
from qdesignoptimizer.utils.utils import get_value_and_unit


class OptPltSet:
    """Manages the configuration of plots showing the optimization progress."""

    def __init__(
        self,
        x: str,
        y: Union[str, List[str]],
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        x_scale: str = "linear",
        y_scale: str = "linear",
        unit: Literal["Hz", "kHz", "MHz", "GHz", "fF"] = "Hz",
    ):
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
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.unit = unit
        self.normalization = {"Hz": 1, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9, "fF": 1}[
            unit
        ]

    def _get_label(
        self, variable: Union[str, List[str]], x_label: Optional[str] = None
    ) -> Union[str, List[str]]:
        if x_label is not None:
            return x_label
        return variable


def plot_progress(
    opt_results: list[list[dict]],
    system_target_params: dict,
    plot_settings: dict,
    block_plots: bool = False,
    save_figures: bool = False,
    plot_variance: bool = False,
    plot_design_variables: Optional[Literal["chronological", "sorted"]] = None,
    opt_target_list: Union[None, List[OptTarget]] = None,
):
    """Plot the progress of the optimization framework

    Args:
        opt_results List[(dict)]: Takes a list of optimization results
        system_target_params (dict): The target system parameters
        plot_settings (dict): The plot settings, example
            {"plt_name": [OptPltSet("panel1_x", "panel1_y"), OptPltSet("panel2_x", "panel2_y")], "plt_name2": ...}
        save_figures: bool = False: Whether to save the plots.
        plot_variance: bool = False: When there are multiple optimization results, whether to add individual lines for each or plot mean and std. deviation.
        plot_design_variables: Optional[Literal["chronological", "sorted"]]  = None: Whether to plot design variables vs iteration and target parameters vs design variables (None if not to be plotted).
                                And whether to sort the design variables (x-axis) when plotting target parameters vs design variables ("sorted" for sorting otherwise "chronological").
                                (Be mindful that some target parameters may depend on multiple design variables, so plotting target parameters vs design variables may not represent the complete physics)
        opt_target_list: Union[None,List[OptTarget]] = None: List of optimization targets to be used when plotting design variables automatically (when plot_design_variables is set True)
                        for mapping target parameters to the respective design variables.
    """

    assert (
        plot_design_variables in ["chronological", "sorted"]
        or plot_design_variables is None
    ), "plot_design_variables can only be None, 'chronological', or 'sorted'"

    def get_data_from_parameter(axes_parameter: str, result: dict, ii: int):
        if axes_parameter == ITERATION:
            data_opt = ii + 1
        elif axes_parameter in result["system_optimized_params"]:
            data_opt = result["system_optimized_params"][axes_parameter]
        elif axes_parameter in result["design_variables"]:
            data_opt = get_value_and_unit(result["design_variables"][axes_parameter])[0]
        else:
            data_opt = None
        return data_opt

    def plot_figure(
        opt_results: list[dict],
        system_target_params: dict,
        panels: list,
        axs: list,
        colors: cycle,
    ) -> bool:
        """Plot all panels in the figure

        Args:
            opt_results (list[dict]): The optimization results
            system_target_params (dict): The target system parameters
            panels (list): The list of OptPltSet objects
            axs (list): The list of axes, one for each panel in panels
            colors (cycle): The cycle of colors

        Returns:
            bool: True if data was plotted, used to delete empty figures
        """
        data_plotted = False
        for idx, panel in enumerate(panels):

            if len(panels) == 1:
                axes: plt.Axes = axs
            else:
                axes = axs[idx]
            if axes.get_legend() is not None:
                axes.get_legend().remove()
            color = next(colors)
            no_opt_results = len(opt_results)
            x_data_opt = [
                get_data_from_parameter(panel.x, result, ii)
                for ii, result in enumerate(
                    opt_results[0]
                )  # Using the 0th index because all different instances of optimization are supposed to have same number of passes
            ]
            if isinstance(panel.y, str):
                y_data_opt_list = []
                for i, opt_result in enumerate(
                    opt_results
                ):  # Looping for different instances of optimization used for analysis
                    # Handle single y parameter (string)
                    y_data_opt = [
                        get_data_from_parameter(panel.y, result, ii)
                        for ii, result in enumerate(opt_result)
                    ]
                    y_data_opt_list.append(y_data_opt)
                y_data_opt_list = np.array(y_data_opt_list)
                if all(
                    all(element is not None for element in y_data_opt)
                    for y_data_opt in y_data_opt_list
                ):
                    data_plotted = True
                if plot_variance is False:
                    for i, y_data_opt in enumerate(y_data_opt_list):
                        axes.plot(
                            x_data_opt,
                            np.array(y_data_opt) / panel.normalization,
                            "o-",
                            label=f"optimized {'' if no_opt_results==1 else i+1}",
                            color=color,
                        )
                else:
                    y_data_mean = np.mean(np.transpose(y_data_opt_list), axis=1)
                    y_data_std = np.std(np.transpose(y_data_opt_list), axis=1)
                    axes.plot(
                        x_data_opt,
                        np.array(y_data_mean) / panel.normalization,
                        "o-",
                        label="optimized mean",
                        color=color,
                    )
                    axes.fill_between(
                        x_data_opt,
                        np.array(y_data_mean) / panel.normalization
                        - np.array(y_data_std) / panel.normalization,
                        np.array(y_data_mean) / panel.normalization
                        + np.array(y_data_std) / panel.normalization,
                        alpha=0.3,
                        facecolor=color,
                    )
                if (
                    x_data_opt
                    and panel.y in system_target_params
                    and (not None in x_data_opt)
                    and (not None in y_data_opt)
                ):
                    y_data_target = system_target_params[panel.y]
                    axes.plot(
                        [min(x_data_opt), max(x_data_opt)],
                        [
                            y_data_target / panel.normalization,
                            y_data_target / panel.normalization,
                        ],
                        "--" if len(x_data_opt) and len(x_data_opt) > 1 else "*",
                        color=color,
                        label="target",
                    )
            else:
                # Handle multiple y parameters (list of strings)
                for y_idx, y_param in enumerate(panel.y):
                    y_data_opt_list: list = []
                    for i, opt_result in enumerate(
                        opt_results
                    ):  # Looping for different instances of optimization used for analysis
                        y_data_opt = [
                            get_data_from_parameter(y_param, result, ii)
                            for ii, result in enumerate(opt_result)
                        ]
                        y_data_opt_list.append(y_data_opt)
                    if all(
                        all(element is not None for element in y_data_opt)
                        for y_data_opt in y_data_opt_list
                    ):
                        data_plotted = True
                    curr_color = color if y_idx == 0 else f"C{y_idx}"
                    if plot_variance is False:
                        for i, y_data_opt in enumerate(y_data_opt_list):
                            axes.plot(
                                x_data_opt,
                                np.array(y_data_opt) / panel.normalization,
                                "o-",
                                label=f"optimized {y_param} {'' if no_opt_results==1 else i+1}",
                                color=curr_color,
                            )
                    else:
                        y_data_mean = np.mean(np.transpose(y_data_opt_list), axis=1)
                        y_data_std = np.std(np.transpose(y_data_opt_list), axis=1)
                        axes.plot(
                            x_data_opt,
                            np.array(y_data_mean) / panel.normalization,
                            "o-",
                            label="optimized mean",
                            color=color,
                        )
                        axes.fill_between(
                            x_data_opt,
                            np.array(y_data_mean) / panel.normalization
                            - np.array(y_data_std) / panel.normalization,
                            np.array(y_data_mean) / panel.normalization
                            + np.array(y_data_std) / panel.normalization,
                            alpha=0.3,
                            facecolor=color,
                        )
                    if (
                        x_data_opt
                        and y_param in system_target_params
                        and (not None in x_data_opt)
                        and (not None in y_data_opt)
                    ):
                        y_data_target = system_target_params[y_param]
                        axes.plot(
                            [min(x_data_opt), max(x_data_opt)],
                            [
                                np.array(y_data_target) / panel.normalization,
                                np.array(y_data_target) / panel.normalization,
                            ],
                            "--" if len(x_data_opt) and len(x_data_opt) > 1 else "*",
                            color=curr_color,
                            label=f"target {y_param}",
                        )

            axes.legend()
            axes.set_xlabel(panel.x_label)
            axes.set_ylabel(panel.y_label + f" ({panel.unit})")
            axes.set_xscale(panel.x_scale)
            axes.set_yscale(panel.y_scale)
            axes.xaxis.set_major_locator(MaxNLocator(integer=True))
        return data_plotted

    def get_design_variable_name_from_target_parameter(
        target_parameter: str, opt_target_list: List[OptTarget]
    ):
        found = False  # Check for whether the target_parameter is found
        for opt_target in opt_target_list:
            opt_target_variable = None

            if opt_target.target_param_type == NONLIN:
                opt_target_variable = param_nonlin(*opt_target.involved_modes)
            elif opt_target.target_param_type == CAPACITANCE:
                opt_target_variable = param_capacitance(*opt_target.involved_modes)
            else:
                opt_target_variable = param(
                    opt_target.involved_modes[0], opt_target.target_param_type
                )
            if target_parameter == opt_target_variable:
                found = True
                return opt_target.design_var

        assert (
            found is True
        ), f"The target parameter {target_parameter} is not found in the optimization targets "

    def get_design_variable_value_from_target_parameter(
        target_parameter: str, result, _, opt_target_list: List
    ):
        design_variable = get_design_variable_name_from_target_parameter(
            target_parameter, opt_target_list
        )
        design_variable_value = result["design_variables"][design_variable]
        assert (
            design_variable_value != None
        ), "design variable {design_variable} does not exist in the file containing the optimization results!"
        value, unit = get_value_and_unit(design_variable_value)
        return value, unit

    def plot_target_parameters_vs_design_variables(
        opt_results: List[dict],
        system_target_params: dict,
        panels: list,
        opt_target_list: List[OptTarget],
        axs: list,
        colors: cycle,
        plot_design_variables_sorted: bool = True,
    ) -> bool:
        """Plot all panels in the figure

        Args:
            opt_results (dict): The optimization results
            system_target_params (dict): The target system parameters
            panels (list): The list of OptPltSet objects
            opt_target_list: Union[None,List[OptTarget]] = None: List of optimization targets for mapping target parameters to the respective design variables
            axs (list): The list of axes, one for each panel in panels
            colors (cycle): The cycle of colors
            plot_design_variables_sorted: bool= True: Whether to sort the design variables (x-axis)

        Returns:
            bool: True if data was plotted, used to delete empty figures
        """
        data_plotted = False
        for idx, panel in enumerate(panels):

            if len(panels) == 1:
                axes: plt.Axes = axs
            else:
                axes = axs[idx]
            if axes.get_legend() is not None:
                axes.get_legend().remove()
            color = next(colors)
            no_opt_results = len(opt_results)
            if isinstance(panel.y, str):
                y_data_opt_list = []
                x_data_opt_list = []
                for i, opt_result in enumerate(
                    opt_results
                ):  # Looping for different instances of optimization used for analysis
                    # Handle single y parameter (string)
                    x_data_opt = [
                        get_design_variable_value_from_target_parameter(
                            panel.y, result, ii, opt_target_list
                        )[0]
                        for ii, result in enumerate(
                            opt_result
                        )  # Using the 0th index because all different instances of optimization are supposed to have same number of passes
                    ]
                    y_data_opt = [
                        get_data_from_parameter(panel.y, result, ii)
                        for ii, result in enumerate(opt_result)
                    ]
                    x_data_opt_list.append(x_data_opt)
                    y_data_opt_list.append(y_data_opt)
                y_data_opt_list = np.array(y_data_opt_list)
                x_data_opt_list = np.array(x_data_opt_list)
                if all(
                    all(element is not None for element in y_data_opt)
                    for y_data_opt in y_data_opt_list
                ):
                    data_plotted = True
                for i, y_data_opt in enumerate(y_data_opt_list):
                    x_data = x_data_opt_list[i]
                    y_data = y_data_opt
                    if plot_design_variables_sorted is True:
                        x_data, y_data = zip(*sorted(zip(x_data, y_data)))  # type: ignore
                    axes.plot(
                        x_data,
                        np.array(y_data) / panel.normalization,
                        "o-",
                        label=f"optimized {'' if no_opt_results==1 else i+1}",
                        color=color,
                    )

                if (
                    panel.y in system_target_params
                    and (not None in x_data_opt)
                    and (not None in y_data_opt)
                ):
                    y_data_target = system_target_params[panel.y]
                    axes.plot(
                        [min(x_data_opt), max(x_data_opt)],
                        [
                            y_data_target / panel.normalization,
                            y_data_target / panel.normalization,
                        ],
                        "--" if len(x_data_opt) and len(x_data_opt) > 1 else "*",
                        color=color,
                        label="target",
                    )
            else:
                # Handle multiple y parameters (list of strings)
                for y_idx, y_param in enumerate(panel.y):
                    x_data_opt_list: list = []
                    y_data_opt_list: list = []
                    for i, opt_result in enumerate(
                        opt_results
                    ):  # Looping for different instances of optimization used for analysis
                        x_data_opt = [
                            get_design_variable_value_from_target_parameter(
                                panel.y, result, ii, opt_target_list
                            )[0]
                            for ii, result in enumerate(opt_result)
                        ]
                        y_data_opt = [
                            get_data_from_parameter(y_param, result, ii)
                            for ii, result in enumerate(opt_result)
                        ]
                        x_data_opt_list.append(x_data_opt)
                        y_data_opt_list.append(y_data_opt)
                    if all(
                        all(element is not None for element in y_data_opt)
                        for y_data_opt in y_data_opt_list
                    ):
                        data_plotted = True
                    curr_color = color if y_idx == 0 else f"C{y_idx}"

                    for i, y_data_opt in enumerate(y_data_opt_list):
                        x_data = x_data_opt_list[i]
                        y_data = y_data_opt
                        if plot_design_variables_sorted is True:
                            x_data, y_data = zip(*sorted(zip(x_data, y_data)))
                        axes.plot(
                            x_data,
                            np.array(y_data) / panel.normalization,
                            "o-",
                            label=f"optimized {y_param} {'' if no_opt_results==1 else i+1}",
                            color=curr_color,
                        )
                    if (
                        y_param in system_target_params
                        and (not None in x_data_opt)
                        and (not None in y_data_opt)
                    ):
                        y_data_target = system_target_params[y_param]
                        axes.plot(
                            [min(x_data_opt), max(x_data_opt)],
                            [
                                np.array(y_data_target) / panel.normalization,
                                np.array(y_data_target) / panel.normalization,
                            ],
                            "--" if len(x_data_opt) and len(x_data_opt) > 1 else "*",
                            color=curr_color,
                            label=f"target {y_param}",
                        )

            axes.legend()
            axes.set_xlabel(
                f"{get_design_variable_name_from_target_parameter(panel.y, opt_target_list)} ({get_design_variable_value_from_target_parameter(panel.y, opt_results[0][0], 0, opt_target_list)[1]})"
            )
            axes.set_ylabel(panel.y_label + f" ({panel.unit})")
            axes.set_xscale(panel.x_scale)
            axes.set_yscale(panel.y_scale)
            axes.xaxis.set_major_locator(MaxNLocator(integer=True))
        return data_plotted

    def plot_design_variables_vs_iterations(
        opt_results: List[dict],
        panels: list,
        axs: list,
        colors: cycle,
    ) -> bool:
        """Plot all panels in the figure

        Args:
            opt_results (dict): The optimization results
            system_target_params (dict): The target system parameters
            panels (list): The list of OptPltSet objects
            axs (list): The list of axes, one for each panel in panels
            colors (cycle): The cycle of colors

        Returns:
            bool: True if data was plotted, used to delete empty figures
        """
        data_plotted = False
        for idx, panel in enumerate(panels):

            if len(panels) == 1:
                axes: plt.Axes = axs
            else:
                axes = axs[idx]
            if axes.get_legend() is not None:
                axes.get_legend().remove()
            color = next(colors)
            no_opt_results = len(opt_results)
            if isinstance(panel.y, str):
                y_data_opt_list = []
                x_data_opt_list = []
                for i, opt_result in enumerate(
                    opt_results
                ):  # Looping for different instances of optimization used for analysis
                    # Handle single y parameter (string)
                    x_data_opt = [
                        get_data_from_parameter(panel.x, result, ii)
                        for ii, result in enumerate(opt_result)
                    ]
                    y_data_opt = [
                        get_design_variable_value_from_target_parameter(
                            panel.y, result, ii, opt_target_list
                        )[0]
                        for ii, result in enumerate(opt_result)
                    ]
                    x_data_opt_list.append(x_data_opt)
                    y_data_opt_list.append(y_data_opt)
                y_data_opt_list = np.array(y_data_opt_list)
                x_data_opt_list = np.array(x_data_opt_list)
                if all(
                    all(element is not None for element in y_data_opt)
                    for y_data_opt in y_data_opt_list
                ):
                    data_plotted = True
                for i, y_data_opt in enumerate(y_data_opt_list):
                    x_data = x_data_opt_list[i]
                    y_data = y_data_opt
                    axes.plot(
                        x_data,
                        y_data,
                        "o-",
                        label=f"optimized {'' if no_opt_results==1 else i+1}",
                        color=color,
                    )  # it may give division error where you will have to make np.array(y_data)
            else:
                # Handle multiple y parameters (list of strings)
                for y_idx, y_param in enumerate(panel.y):
                    x_data_opt_list = []
                    y_data_opt_list = []
                    for i, opt_result in enumerate(
                        opt_results
                    ):  # Looping for different instances of optimization used for analysis
                        x_data_opt = [
                            get_data_from_parameter(panel.x, result, ii)
                            for ii, result in enumerate(opt_result)
                        ]
                        y_data_opt = [
                            get_design_variable_value_from_target_parameter(
                                y_param, result, ii, opt_target_list
                            )[0]
                            for ii, result in enumerate(opt_result)
                        ]
                        x_data_opt_list.append(x_data_opt)
                        y_data_opt_list.append(y_data_opt)
                    if all(
                        all(element is not None for element in y_data_opt)
                        for y_data_opt in y_data_opt_list
                    ):
                        data_plotted = True
                    curr_color = color if y_idx == 0 else f"C{y_idx}"

                    for i, y_data_opt in enumerate(y_data_opt_list):
                        x_data = x_data_opt_list[i]
                        y_data = y_data_opt
                        axes.plot(
                            x_data,
                            np.array(y_data),
                            "o-",
                            label=f"optimized {y_param} {'' if no_opt_results==1 else i+1}",
                            color=curr_color,
                        )
            axes.legend()
            axes.set_xlabel(panel.x_label)
            axes.set_ylabel(
                f"{get_design_variable_name_from_target_parameter(panel.y, opt_target_list)} ({get_design_variable_value_from_target_parameter(panel.y, opt_results[0][0], 0, opt_target_list)[1]})"
            )
            axes.set_xscale(panel.x_scale)
            axes.set_yscale(panel.y_scale)
            axes.xaxis.set_major_locator(MaxNLocator(integer=True))
        return data_plotted

    plt.close("all")
    colors = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    for plot_name, panels in plot_settings.items():
        fig, axs = plt.subplots(len(panels))
        plot_figure([opt_results], system_target_params, panels, axs, colors)
        fig.suptitle(plot_name)
        fig.subplots_adjust(hspace=0.5)
        if save_figures is True:
            fig.savefig(
                f"optimization_plot_{time.strftime('%Y%m%d-%H%M%S')}_{plot_name}.png"
            )

        if plot_design_variables is not None:
            assert (
                opt_target_list is not None
            ), "To plot design variables as a function of target parameters, optimization target list cannot be None"

            fig, axs = plt.subplots(len(panels))
            plot_target_parameters_vs_design_variables(
                opt_results,
                system_target_params,
                panels,
                opt_target_list,
                axs,
                colors,
                plot_design_variables_sorted=(plot_design_variables == "chronological"),
            )
            fig.suptitle(plot_name)
            fig.subplots_adjust(hspace=0.5)
            if save_figures is True:
                fig.savefig(
                    f"optimization_plot_{time.strftime('%Y%m%d-%H%M%S')}_{plot_name}.png"
                )

            fig, axs = plt.subplots(len(panels))
            plot_design_variables_vs_iterations(
                opt_results,
                panels,
                axs,
                colors,
            )
            fig.suptitle(plot_name)
            fig.subplots_adjust(hspace=0.5)
            if save_figures is True:
                fig.savefig(
                    f"optimization_plot_{time.strftime('%Y%m%d-%H%M%S')}_{plot_name}.png"
                )

    plt.show(block=block_plots)
