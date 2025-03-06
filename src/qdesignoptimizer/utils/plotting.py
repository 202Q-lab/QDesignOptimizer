import numpy as np
import os

# plotting function
from qdesignoptimizer.design_analysis_types import OptTarget
from qdesignoptimizer.sim_plot_progress import plot_progress
from typing import List, Literal, Union

from pprint import pprint
import json 
from datetime import datetime



def load_data_by_date(folder_parent: str, name_experiment: str, experiment_beginning: str, experiment_end: str):

    files_experiment = []

    all_files_in_folder = os.listdir(folder_parent)
    # get all .npy
    all_files_in_folder = [os.path.join(folder_parent, f) for f in all_files_in_folder]
    npy_ind = []
    for i in range(len(all_files_in_folder)):
        if all_files_in_folder[i].endswith('npy'):
            npy_ind.append(i)

    npy_ind = np.array(npy_ind)
    all_files_in_folder = np.array(all_files_in_folder)

    files_npy = all_files_in_folder[npy_ind]

    # filter for experiment name
    experiment_ind = []
    for i in range(len(files_npy)):
        if files_npy[i].find(name_experiment) >0:
            experiment_ind.append(i)

    experiment_ind = np.array(experiment_ind)
    files_experiment.append(files_npy[experiment_ind])

    date_start_dt = datetime.strptime(experiment_beginning.split('_')[-1], '%Y%m%d-%H%M%S')
    date_stop_dt = datetime.strptime(experiment_end.split('_')[-1], '%Y%m%d-%H%M%S')

    filtered_files = []
    for i, file in enumerate(np.concatenate(files_experiment)):
        if date_start_dt <= datetime.strptime(file.split('_')[-1].rstrip('.npy'), '%Y%m%d-%H%M%S') <= date_stop_dt:
            filtered_files.append(file)


    return filtered_files


def plot_optimization_results(files: List[str], plot_variance: bool = True,plot_type: Literal['target_vs_iterations', 'target_vs_variable', 'variable_vs_iteration']= 'target_vs_iterations', plot_design_variables_sorted: bool = True, opt_target_list: Union[None, List[OptTarget]] = None ):
    results = []
    for file in files:
        results.append(np.load(file, allow_pickle= True)[0])
    results = np.array(results)

    for result in results:
        assert result["system_target_params"] == results[0]["system_target_params"], "All optimization results must have the same target parameters"
        assert len(result["optimization_results"]) == len(results[0]["optimization_results"]), "All optimization results must have the same number of passes"

    plot_progress(
                [result["optimization_results"]for result in results],
                results[0]["system_target_params"],
                results[0]["plot_settings"],
                save_figures=True,
                plot_variance=plot_variance,
                plot_type= plot_type,
                plot_design_variables_sorted=plot_design_variables_sorted,
                opt_target_list=opt_target_list
    )