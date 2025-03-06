import os
import time
from itertools import cycle
from typing import Union, List, Literal

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from qdesignoptimizer.design_analysis_types import OptTarget
from qdesignoptimizer.utils.names_design_variables import name_mode
from qdesignoptimizer.utils.names_parameters import NONLIN, get_modes_from_param_nonlin, param, ITERATION, param_nonlin
from qdesignoptimizer.utils.utils import get_value_and_unit

class OptPltSet:
    def __init__(self, x: str, 
                 y: Union[str, List[str]], 
                 x_label: str = None, 
                 y_label: str = None, 
                 x_scale: str = 'linear', 
                 y_scale: str = 'linear',
                 unit: Literal['Hz', 'kHz', 'MHz','GHz']= 'Hz'):
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
        self.normalization =  {'Hz': 1,'kHz': 1e3, 'MHz': 1e6,'GHz': 1e9}[unit]

    def _get_label(self, variable: str, x_label: str):
        if x_label is not None:
            return x_label
        else:
            return variable


def plot_progress(
    opt_results: List[dict],
    system_target_params: dict,
    plot_settings: dict,
    block_plots: bool = False,
    save_figures: bool = False,
    plot_variance: bool = False,
    plot_type: Literal['target_vs_iterations', 'target_vs_variable', 'variable_vs_iteration']= 'target_vs_iterations',
    plot_design_variables_sorted: bool= True,
    opt_target_list: Union[None,List[OptTarget]] = None
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
        opt_results: List[dict],
        system_target_params: dict,
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
            no_opt_results = len(opt_results)
            x_data_opt = [
                get_data_from_parameter(panel.x, result, ii)
                for ii, result in enumerate(opt_results[0]) # Using the 0th index because all different instances of optimization are supposed to have same number of passes
            ]
            if isinstance(panel.y, str):
                y_data_opt_list = []
                for i, opt_result in enumerate(opt_results): # Looping for different instances of optimization used for analysis
                    # Handle single y parameter (string)
                    y_data_opt = [
                        get_data_from_parameter(panel.y, result, ii)
                        for ii, result in enumerate(opt_result)
                    ]
                    y_data_opt_list.append(y_data_opt)
                y_data_opt_list = np.array(y_data_opt_list)
                if all(all(element is not None for element in y_data_opt) for y_data_opt in y_data_opt_list):
                    data_plotted = True
                if plot_variance is False:
                    for i, y_data_opt in enumerate(y_data_opt_list):
                        axes.plot(x_data_opt, np.array(y_data_opt)/panel.normalization, "o-", label=f"optimized {'' if no_opt_results==1 else i+1}", color=color)
                else:
                    y_data_mean = np.mean(np.transpose(y_data_opt_list),axis=1)
                    y_data_std = np.std(np.transpose(y_data_opt_list),axis=1)
                    axes.plot(x_data_opt, np.array(y_data_mean)/panel.normalization, "o-", label=f"optimized mean", color=color)
                    axes.fill_between(x_data_opt, np.array(y_data_mean)/panel.normalization - np.array(y_data_std)/panel.normalization, np.array(y_data_mean)/panel.normalization +  np.array(y_data_std)/panel.normalization, alpha=0.3, facecolor= color)
                if (
                    panel.y in system_target_params
                    and (not None in x_data_opt)
                    and (not None in y_data_opt)
                ):
                    y_data_target = system_target_params[panel.y]
                    axes.plot( 
                        [min(x_data_opt), max(x_data_opt)],
                        [y_data_target/panel.normalization, y_data_target/panel.normalization],
                        "--" if len(x_data_opt) and len(x_data_opt) > 1 else "*",
                        color=color,
                        label=f"target",
                    )
            else:
                # Handle multiple y parameters (list of strings)
                for y_idx, y_param in enumerate(panel.y):
                    y_data_opt_list = []
                    for i, opt_result in enumerate(opt_results): # Looping for different instances of optimization used for analysis
                        y_data_opt = [
                            get_data_from_parameter(y_param, result, ii)
                            for ii, result in enumerate(opt_result)
                        ]
                        y_data_opt_list.append(y_data_opt)
                    if all(all(element is not None for element in y_data_opt) for y_data_opt in y_data_opt_list):
                        data_plotted = True
                    curr_color = color if y_idx == 0 else f"C{y_idx}"
                    if plot_variance is False:
                        for i, y_data_opt in enumerate(y_data_opt_list):
                            axes.plot(x_data_opt, np.array(y_data_opt)/panel.normalization, "o-", label=f"optimized {y_param} {'' if no_opt_results==1 else i+1}", color=curr_color)
                    else:
                        y_data_mean = np.mean(np.transpose(y_data_opt_list),axis=1)
                        y_data_std = np.std(np.transpose(y_data_opt_list),axis=1)
                        axes.plot(x_data_opt, np.array(y_data_mean)/panel.normalization, "o-", label=f"optimized mean", color=color)
                        axes.fill_between(x_data_opt, np.array(y_data_mean)/panel.normalization - np.array(y_data_std)/panel.normalization, np.array(y_data_mean)/panel.normalization +  np.array(y_data_std)/panel.normalization, alpha=0.3, facecolor= color)
                    if (
                        y_param in system_target_params
                        and (not None in x_data_opt)
                        and (not None in y_data_opt)
                    ):
                        y_data_target = system_target_params[y_param]
                        axes.plot(
                            [min(x_data_opt), max(x_data_opt)],
                            [np.array(y_data_target)/panel.normalization, np.array(y_data_target)/panel.normalization],
                            "--" if len(x_data_opt) and len(x_data_opt) > 1 else "*",
                            color=curr_color,
                            label=f"target {y_param}",
                        )

            axes.legend()
            axes.set_xlabel(panel.x_label)
            axes.set_ylabel(panel.y_label+f" ({panel.unit})")
            axes.set_xscale(panel.x_scale)
            axes.set_yscale(panel.y_scale)
            axes.xaxis.set_major_locator(MaxNLocator(integer=True))
        return data_plotted
    
    def get_design_variable_name_from_target_parameter(target_parameter: str, opt_target_list: List[OptTarget]):
        found = False # Check for whether the target_parameter is found
        for opt_target in opt_target_list:
            opt_target_variable = None

            if opt_target.system_target_param != NONLIN:
                opt_target_variable = param(opt_target.involved_modes[0],opt_target.system_target_param)
            else:
                opt_target_variable = param_nonlin(*opt_target.involved_modes)
            if target_parameter == opt_target_variable:
                found = True
                return opt_target.design_var
            
        assert found==True, f"The target parameter {target_parameter} is not found in the optimization targets "
            
    def get_design_variable_value_from_target_parameter(target_parameter: str, result, ii, opt_target_list: List):
        design_variable = get_design_variable_name_from_target_parameter(target_parameter, opt_target_list)
        design_variable_value = result["design_variables"][design_variable]
        assert design_variable_value!= None, "design variable {design_variable} does not exist in the file containing the optimization results!"
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
            no_opt_results = len(opt_results)
            if isinstance(panel.y, str):
                y_data_opt_list = []
                x_data_opt_list = []
                for i, opt_result in enumerate(opt_results): # Looping for different instances of optimization used for analysis
                    # Handle single y parameter (string)
                    x_data_opt = [
                        get_design_variable_value_from_target_parameter(panel.y, result, ii, opt_target_list)[0]
                        for ii, result in enumerate(opt_result) # Using the 0th index because all different instances of optimization are supposed to have same number of passes
                    ]
                    y_data_opt = [
                        get_data_from_parameter(panel.y, result, ii)
                        for ii, result in enumerate(opt_result)
                    ]
                    x_data_opt_list.append(x_data_opt)
                    y_data_opt_list.append(y_data_opt)
                y_data_opt_list = np.array(y_data_opt_list)
                x_data_opt_list = np.array(x_data_opt_list)
                if all(all(element is not None for element in y_data_opt) for y_data_opt in y_data_opt_list):
                    data_plotted = True
                for i, y_data_opt in enumerate(y_data_opt_list):
                    x_data = x_data_opt_list[i]
                    y_data = y_data_opt
                    if plot_design_variables_sorted == True:
                        x_data, y_data = zip(*sorted(zip(x_data,y_data)))
                    axes.plot(x_data, np.array(y_data)/panel.normalization, "o-", label=f"optimized {'' if no_opt_results==1 else i+1}", color=color)
                
                if (
                    panel.y in system_target_params
                    and (not None in x_data_opt)
                    and (not None in y_data_opt)
                ):
                    y_data_target = system_target_params[panel.y]
                    axes.plot( 
                        [min(x_data_opt), max(x_data_opt)],
                        [y_data_target/panel.normalization, y_data_target/panel.normalization],
                        "--" if len(x_data_opt) and len(x_data_opt) > 1 else "*",
                        color=color,
                        label=f"target",
                    )
            else:
                # Handle multiple y parameters (list of strings)
                for y_idx, y_param in enumerate(panel.y):
                    x_data_opt_list = []
                    y_data_opt_list = []
                    for i, opt_result in enumerate(opt_results): # Looping for different instances of optimization used for analysis
                        x_data_opt = [
                        get_design_variable_value_from_target_parameter(panel.y, result, ii, opt_target_list)[0]
                        for ii, result in enumerate(opt_result) 
                        ]
                        y_data_opt = [
                            get_data_from_parameter(y_param, result, ii)
                            for ii, result in enumerate(opt_result)
                        ]
                        x_data_opt_list.append(x_data_opt)
                        y_data_opt_list.append(y_data_opt)
                    if all(all(element is not None for element in y_data_opt) for y_data_opt in y_data_opt_list):
                        data_plotted = True
                    curr_color = color if y_idx == 0 else f"C{y_idx}"

                    for i, y_data_opt in enumerate(y_data_opt_list):
                        x_data = x_data_opt_list[i]
                        y_data = y_data_opt
                        if plot_design_variables_sorted == True:
                            x_data, y_data = zip(*sorted(zip(x_data,y_data)))
                        axes.plot(x_data, np.array(y_data)/panel.normalization, "o-", label=f"optimized {y_param} {'' if no_opt_results==1 else i+1}", color=curr_color)
                    if (
                        y_param in system_target_params
                        and (not None in x_data_opt)
                        and (not None in y_data_opt)
                    ):
                        y_data_target = system_target_params[y_param]
                        axes.plot(
                            [min(x_data_opt), max(x_data_opt)],
                            [np.array(y_data_target)/panel.normalization, np.array(y_data_target)/panel.normalization],
                            "--" if len(x_data_opt) and len(x_data_opt) > 1 else "*",
                            color=curr_color,
                            label=f"target {y_param}",
                        )

            axes.legend()
            axes.set_xlabel(f"{get_design_variable_name_from_target_parameter(panel.y, opt_target_list)} ({get_design_variable_value_from_target_parameter(panel.y, opt_results[0][0], 0, opt_target_list)[1]})")
            axes.set_ylabel(panel.y_label+f" ({panel.unit})")
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
            no_opt_results = len(opt_results)
            if isinstance(panel.y, str):
                y_data_opt_list = []
                x_data_opt_list = []
                for i, opt_result in enumerate(opt_results): # Looping for different instances of optimization used for analysis
                    # Handle single y parameter (string)
                    x_data_opt = [
                        get_data_from_parameter(panel.x, result, ii)
                        for ii, result in enumerate(opt_result)
                    ]
                    y_data_opt = [
                        get_design_variable_value_from_target_parameter(panel.y, result, ii, opt_target_list)[0]
                        for ii, result in enumerate(opt_result) 
                    ]
                    x_data_opt_list.append(x_data_opt)
                    y_data_opt_list.append(y_data_opt)
                y_data_opt_list = np.array(y_data_opt_list)
                x_data_opt_list = np.array(x_data_opt_list)
                if all(all(element is not None for element in y_data_opt) for y_data_opt in y_data_opt_list):
                    data_plotted = True
                for i, y_data_opt in enumerate(y_data_opt_list):
                    x_data = x_data_opt_list[i]
                    y_data = y_data_opt
                    axes.plot(x_data, y_data, "o-", label=f"optimized {'' if no_opt_results==1 else i+1}", color=color) # it may give division error where you will have to make np.array(y_data)
            else:
                # Handle multiple y parameters (list of strings)
                for y_idx, y_param in enumerate(panel.y):
                    x_data_opt_list = []
                    y_data_opt_list = []
                    for i, opt_result in enumerate(opt_results): # Looping for different instances of optimization used for analysis
                        x_data_opt = [
                            get_data_from_parameter(panel.x, result, ii)
                            for ii, result in enumerate(opt_result)
                        ]
                        y_data_opt = [
                        get_design_variable_value_from_target_parameter(y_param, result, ii, opt_target_list)[0]
                        for ii, result in enumerate(opt_result)
                        ]
                        x_data_opt_list.append(x_data_opt)
                        y_data_opt_list.append(y_data_opt)
                    if all(all(element is not None for element in y_data_opt) for y_data_opt in y_data_opt_list):
                        data_plotted = True
                    curr_color = color if y_idx == 0 else f"C{y_idx}"

                    for i, y_data_opt in enumerate(y_data_opt_list):
                        x_data = x_data_opt_list[i]
                        y_data = y_data_opt
                        axes.plot(x_data, np.array(y_data), "o-", label=f"optimized {y_param} {'' if no_opt_results==1 else i+1}", color=curr_color)
            axes.legend()
            axes.set_xlabel(panel.x_label)
            axes.set_ylabel(f"{get_design_variable_name_from_target_parameter(panel.y, opt_target_list)} ({get_design_variable_value_from_target_parameter(panel.y, opt_results[0][0], 0, opt_target_list)[1]})")
            axes.set_xscale(panel.x_scale)
            axes.set_yscale(panel.y_scale)
            axes.xaxis.set_major_locator(MaxNLocator(integer=True))
        return data_plotted

    plt.close("all")
    colors = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    for plot_name, panels in plot_settings.items():
        fig, axs = plt.subplots(len(panels))
        if plot_type == 'target_vs_iterations':
            plot_figure(
                opt_results,
                system_target_params,
                panels,
                axs,
                colors
            )
        elif plot_type=='target_vs_variable':
            assert opt_target_list is not None, "To plot design variables as a function of target parameters, optimization target list cannot be None"
            plot_target_parameters_vs_design_variables( opt_results,
                                  system_target_params,
                                  panels,
                                  opt_target_list,
                                  axs,
                                  colors,
                                  plot_design_variables_sorted)
        else:
            plot_design_variables_vs_iterations(
                opt_results,
                panels,
                axs,
                colors,
            )
        fig.suptitle(plot_name)
        fig.subplots_adjust(hspace=0.5)
        if save_figures == True:
            fig.savefig(f"optimization_plot_{time.strftime('%Y%m%d-%H%M%S')}_{plot_name}.png")
    plt.show(block=block_plots)



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
        block_plots=True,
    )
