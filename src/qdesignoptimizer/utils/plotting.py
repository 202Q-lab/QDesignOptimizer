"""Additional plotting utilities for loading and visualizing optimization results from files."""

import os
from datetime import datetime
from typing import List, Literal, Optional

import numpy as np

from qdesignoptimizer.design_analysis_types import OptTarget
from qdesignoptimizer.sim_plot_progress import plot_progress


def load_data_by_date(
    folder_parent: str,
    name_experiment: str,
    experiment_beginning: str,
    experiment_end: str,
) -> List[str]:
    """
    Load optimization data files within a specified date range and experiment name.

    Searches a directory for .npy files matching an experiment name and falling within
    a specified date range. The function extracts dates from filenames following the
    convention 'experiment_name_YYYYMMDD-HHMMSS.npy'.

    Args:
        folder_parent: Directory path containing the optimization result files.
        name_experiment: Experiment name to filter files (e.g., "multi_transmon_chip").
        experiment_beginning: Start date in 'YYYYMMDD-HHMMSS' format, either as full filename
                             or just the date portion.
        experiment_end: End date in 'YYYYMMDD-HHMMSS' format, either as full filename
                       or just the date portion.

    Returns:
        List of full file paths for matching optimization result files.

    Example:
        >>> files = load_data_by_date(
        ...     "results/",
        ...     "multi_transmon_chip",
        ...     "20250306-165308",
        ...     "20250308-120000"
        ... )
    """
    files_experiment = []

    # Get all files in the parent folder
    all_files_in_folder = os.listdir(folder_parent)
    all_files_in_folder = [os.path.join(folder_parent, f) for f in all_files_in_folder]

    # Filter for .npy files
    npy_ind = []
    for i, folder in enumerate(all_files_in_folder):
        if folder.endswith("npy"):
            npy_ind.append(i)

    npy_ind = np.array(npy_ind)
    all_files_in_folder = np.array(all_files_in_folder)
    files_npy = all_files_in_folder[npy_ind]

    # Filter for experiment name
    experiment_ind = []
    for i, file_npy in enumerate(files_npy):
        if file_npy.find(name_experiment) > 0:
            experiment_ind.append(i)

    experiment_ind = np.array(experiment_ind)
    files_experiment.append(files_npy[experiment_ind])

    # Extract dates from filenames and filter by date range
    date_start_dt = datetime.strptime(
        experiment_beginning.split("_")[-1], "%Y%m%d-%H%M%S"
    )
    date_stop_dt = datetime.strptime(experiment_end.split("_")[-1], "%Y%m%d-%H%M%S")

    filtered_files = []
    for file in np.concatenate(files_experiment):
        file_date_str = file.split("_")[-1].rstrip(".npy")
        file_date = datetime.strptime(file_date_str, "%Y%m%d-%H%M%S")
        if date_start_dt <= file_date <= date_stop_dt:
            filtered_files.append(file)

    return filtered_files


def plot_optimization_results(
    files: List[str],
    plot_variance: bool = True,
    plot_design_variables: Optional[Literal["chronological", "sorted"]] = None,
    opt_target_list: Optional[List[OptTarget]] = None,
    plot_settings: Optional[dict] = None,
    save_figures: bool = True,
) -> None:
    """
    Plot optimization results from multiple data files.

    Loads optimization data from provided file paths and creates visualizations
    using the plot_progress function. Handles multiple optimization runs and
    configures plotting options for parameter visualization.

    Args:
        files: List of file paths to optimization result files (.npy format).
        plot_variance: Whether to show mean and standard deviation across multiple
                      optimization runs (True) or individual lines for each run (False).
                      Defaults to True.
        plot_design_variables: How design variables should be plotted against parameters:

                              - "chronological": Plot in order of iterations
                              - "sorted": Sort by design variable values
                              - None: Don't plot design variables

                              Note that some target parameters may depend on multiple design
                              variables, so these plots may not capture the full physics.
        opt_target_list: List of optimization targets defining relationships between
                        physical parameters and design variables. Required when
                        plot_design_variables is set.
        save_figures: Whether to save generated plots to disk. Defaults to True.

    Raises:
        AssertionError: If loaded optimization results have inconsistent target
                       parameters or different numbers of iterations.

    Example:
        >>> files = load_data_by_date("results/", "transmon", "20250306-165308", "20250308-120000")
        >>> targets = get_opt_targets_qb_res_transmission("qubit_1", "resonator_1", "feedline")
        >>> plot_optimization_results(
        ...     files,
        ...     plot_variance=True,
        ...     plot_design_variables="sorted",
        ...     opt_target_list=targets
        ... )
    """
    # Load results from files
    results = []
    for file in files:
        results.append(np.load(file, allow_pickle=True)[0])
    results = np.array(results)

    if plot_settings is None:
        plt_set = results[0]["plot_settings"]
    else:
        plt_set = plot_settings

    # Verify consistency of loaded results
    for result in results:
        assert (
            result["system_target_params"] == results[0]["system_target_params"]
        ), "All optimization results must have the same target parameters"
        assert len(result["optimization_results"]) == len(
            results[0]["optimization_results"]
        ), "All optimization results must have the same number of passes"

    # Generate plots using the plot_progress function
    plot_progress(
        [result["optimization_results"] for result in results],
        results[0]["system_target_params"],
        plt_set,
        save_figures=save_figures,
        plot_variance=plot_variance,
        plot_design_variables=plot_design_variables,
        opt_target_list=opt_target_list,
    )
