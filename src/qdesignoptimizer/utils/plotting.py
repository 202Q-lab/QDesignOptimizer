"""Additional plotting utilities for data loaded from files."""

import os
from datetime import datetime
from typing import List, Literal, Optional, Union

import numpy as np

# plotting function
from qdesignoptimizer.design_analysis_types import OptTarget
from qdesignoptimizer.sim_plot_progress import plot_progress


def load_data_by_date(
    folder_parent: str,
    name_experiment: str,
    experiment_beginning: str,
    experiment_end: str,
):
    """Loads file names with location for plotting with plot_optimization_results

    Args:
    folder_parent: str: Location of the parent folder from where to retrieve the files.
    name_experiment: str: Name of the experiment (which is a string literal which will be used to sort out the files). eg: "multi_transmon_chip".
    experiment_beginning: str: Name or datetime of the first result (YYYYMMDD-HHMMSS format). eg: "multi_transmon_chip_20250306-165308", "20250306-165308"
    experiment_end: str: Name or datetime of the last result (YYYYMMDD-HHMMSS format)
    """

    files_experiment = []

    all_files_in_folder = os.listdir(folder_parent)
    # get all .npy
    all_files_in_folder = [os.path.join(folder_parent, f) for f in all_files_in_folder]
    npy_ind = []
    for i, folder in enumerate(all_files_in_folder):
        if folder.endswith("npy"):
            npy_ind.append(i)

    npy_ind = np.array(npy_ind)
    all_files_in_folder = np.array(all_files_in_folder)

    files_npy = all_files_in_folder[npy_ind]

    # filter for experiment name
    experiment_ind = []
    for i, file_npy in enumerate(files_npy):
        if file_npy.find(name_experiment) > 0:
            experiment_ind.append(i)

    experiment_ind = np.array(experiment_ind)
    files_experiment.append(files_npy[experiment_ind])

    date_start_dt = datetime.strptime(
        experiment_beginning.split("_")[-1], "%Y%m%d-%H%M%S"
    )
    date_stop_dt = datetime.strptime(experiment_end.split("_")[-1], "%Y%m%d-%H%M%S")

    filtered_files = []
    for i, file in enumerate(np.concatenate(files_experiment)):
        if (
            date_start_dt
            <= datetime.strptime(file.split("_")[-1].rstrip(".npy"), "%Y%m%d-%H%M%S")
            <= date_stop_dt
        ):
            filtered_files.append(file)

    return filtered_files


def plot_optimization_results(
    files: List[str],
    plot_variance: bool = True,
    plot_design_variables: Optional[Literal["chronological", "sorted"]] = None,
    opt_target_list: Union[None, List[OptTarget]] = None,
    save_figures: bool = True,
):
    """Wrapper for plotting optimization results.

    Args:
        files: List[str]: List of file locations to be analysed.
        plot_variance: bool = False: When there are multiple optimization results, whether to add individual lines for each or plot mean and std. deviation.
        plot_design_variables: Optional[Literal['chronological', 'sorted']]  = None: Whether to plot design variables vs iteration and target parameters vs design variables (None if not to be plotted).
                                And whether to sort the design variables (x-axis) when plotting target parameters vs design variables ('sorted' for sorting otherwise 'chronological').
                                (Be mindful that some target parameters may depend on multiple design variables, so plotting target parameters vs design variables may not represent the complete physics)
        opt_target_list: Union[None,List[OptTarget]] = None: List of optimization targets to be used when plotting design variables automatically (when plot_design_variables is set True)
                        for mapping target parameters to the respective design variables.
        opt_results List[(dict)]: Takes a list of optimization results.
        save_figures: bool = False: Whether to save the plots.

    """
    results = []
    for file in files:
        results.append(np.load(file, allow_pickle=True)[0])
    results = np.array(results)

    for result in results:
        assert (
            result["system_target_params"] == results[0]["system_target_params"]
        ), "All optimization results must have the same target parameters"
        assert len(result["optimization_results"]) == len(
            results[0]["optimization_results"]
        ), "All optimization results must have the same number of passes"

    plot_progress(
        [result["optimization_results"] for result in results],
        results[0]["system_target_params"],
        results[0]["plot_settings"],
        save_figures=save_figures,
        plot_variance=plot_variance,
        plot_design_variables=plot_design_variables,
        opt_target_list=opt_target_list,
    )
