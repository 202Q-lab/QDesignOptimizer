"""Visualization utilities for tracking optimization progress of quantum circuit designs."""

import time
from dataclasses import dataclass
from enum import Enum
from itertools import cycle
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
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


class UnitEnum(str, Enum):
    """Supported frequency units for plotting."""

    HZ = "Hz"
    KHZ = "kHz"
    MHZ = "MHz"
    GHZ = "GHz"
    FF = "fF"
    S = "s"
    MS = "ms"
    US = "us"
    NS = "ns"
    ARB = "arb."


@dataclass
class OptPltSet:
    """Configuration for optimization progress plots."""

    x: str
    """X-axis parameter name."""

    y: Union[str, List[str]]
    """Y-axis parameter name(s)."""

    x_label: Optional[str] = None
    """Custom X-axis label (uses x if None)."""

    y_label: Optional[str] = None
    """Custom Y-axis label (uses y if None)."""

    x_scale: Literal["linear", "log"] = "linear"
    """X-axis scale (linear or logarithmic)."""

    y_scale: Literal["linear", "log"] = "linear"
    """Y-axis scale (linear or logarithmic)."""

    unit: UnitEnum = UnitEnum.HZ
    """Display unit for y-axis values."""

    @property
    def normalization(self) -> float:
        """Get the normalization factor for the selected unit."""
        factors = {
            UnitEnum.HZ: 1,
            UnitEnum.KHZ: 1e3,
            UnitEnum.MHZ: 1e6,
            UnitEnum.GHZ: 1e9,
            UnitEnum.FF: 1,
            UnitEnum.S: 1,
            UnitEnum.MS: 1e-3,
            UnitEnum.US: 1e-6,
            UnitEnum.NS: 1e-9,
            UnitEnum.ARB: 1,
        }
        return factors[self.unit]

    def get_x_label(self) -> str:
        """Get the x-axis label, using the default if none specified."""
        return self.x_label if self.x_label is not None else self.x

    def get_y_label(self) -> str:
        """Get the y-axis label, using the default if none specified."""
        return (
            self.y_label
            if self.y_label is not None
            else (self.y if isinstance(self.y, str) else ", ".join(self.y))
        )


OptimizationResult = Dict[str, Any]
SystemParams = Dict[str, Any]
PlotSettingsType = Dict[str, List[OptPltSet]]
T = TypeVar("T")


class DataExtractor:
    """Extracts data from optimization results for plotting."""

    def __init__(
        self,
        opt_results: List[List[OptimizationResult]],
        system_target_params: SystemParams,
        opt_target_list: Optional[List[OptTarget]] = None,
    ):
        """Initialize the data extractor.

        Args:
            opt_results: List of optimization result sequences (one per optimization run)
            system_target_params: Target system parameters
            opt_target_list: Optional list of optimization targets for mapping parameters
        """
        self.opt_results = opt_results
        self.system_target_params = system_target_params
        self.opt_target_list = opt_target_list

    def get_parameter_value(
        self, param_name: str, result: OptimizationResult, iteration: int
    ) -> Optional[float]:
        """Extract parameter value from a result entry.

        Args:
            param_name: Name of the parameter to extract
            result: Optimization result entry
            iteration: Iteration number (for ITERATION parameter)

        Returns:
            Parameter value or None if not found
        """
        if param_name == ITERATION:
            return iteration + 1
        elif param_name in result["system_optimized_params"]:
            return result["system_optimized_params"][param_name]
        elif param_name in result["design_variables"]:
            value, _ = get_value_and_unit(result["design_variables"][param_name])
            return value
        return None

    def get_design_var_name_for_param(self, target_parameter: str) -> str:
        """Find the design variable name associated with a target parameter.

        Args:
            target_parameter: Target parameter name

        Returns:
            Name of the associated design variable

        Raises:
            AssertionError: If the parameter is not found in optimization targets
        """
        if not self.opt_target_list:
            raise ValueError("No optimization targets provided")

        for opt_target in self.opt_target_list:
            opt_target_variable = None

            if opt_target.target_param_type == NONLIN:
                opt_target_variable = param_nonlin(*opt_target.involved_modes)
            elif opt_target.target_param_type == CAPACITANCE:
                opt_target_variable = param_capacitance(*opt_target.involved_modes)
            else:
                opt_target_variable = param(
                    opt_target.involved_modes[0], opt_target.target_param_type  # type: ignore
                )

            if target_parameter == opt_target_variable:
                return opt_target.design_var

        raise AssertionError(
            f"Target parameter {target_parameter} not found in optimization targets"
        )

    def get_design_var_for_param(
        self, target_parameter: str, result: OptimizationResult
    ) -> Tuple[float, str]:
        """Get design variable value and unit for a target parameter.

        Args:
            target_parameter: Target parameter name
            result: Optimization result entry

        Returns:
            Tuple of (value, unit)
        """
        design_variable = self.get_design_var_name_for_param(target_parameter)
        design_var_value = result["design_variables"][design_variable]
        assert (
            design_var_value is not None
        ), f"Design variable {design_variable} not found in results"

        value, unit = get_value_and_unit(design_var_value)
        return value, unit

    def extract_xy_data(
        self,
        x_param: str,
        y_param: str,
        run_index: int,
        use_design_var_as_x: bool = False,
        sort_by_x: bool = False,
    ) -> Tuple[List[float], List[float]]:
        """Extract x and y data series for plotting.

        Args:
            x_param: X-axis parameter name
            y_param: Y-axis parameter name
            run_index: Index of the optimization run to use
            use_design_var_as_x: If True, use design variable for x instead of parameter
            sort_by_x: If True, sort the data by x values

        Returns:
            Tuple of (x_values, y_values)
        """
        opt_result = self.opt_results[run_index]

        if use_design_var_as_x:
            # Use design variable associated with y_param as x
            x_values = [
                self.get_design_var_for_param(y_param, result)[0]
                for _, result in enumerate(opt_result)
            ]
        else:
            # Use parameter directly
            x_values = [
                self.get_parameter_value(x_param, result, i)  # type: ignore
                for i, result in enumerate(opt_result)
            ]

        y_values = [
            self.get_parameter_value(y_param, result, i)
            for i, result in enumerate(opt_result)
        ]

        # Filter out None values
        x_values_filtered, y_values_filtered = zip(
            *[
                (x, y)
                for x, y in zip(x_values, y_values)
                if x is not None and y is not None
            ]
        )

        x_values_filtered = list(x_values_filtered)
        y_values_filtered = list(y_values_filtered)

        if sort_by_x and x_values_filtered:
            # Sort by x values if requested
            sorted_pairs = sorted(zip(x_values_filtered, y_values_filtered))
            x_values_filtered, y_values_filtered = zip(*sorted_pairs)  # type: ignore

        return list(x_values_filtered), list(y_values_filtered)

    def get_y_data_with_statistics(
        self,
        x_param: str,
        y_param: str,
        use_design_var_as_x: bool = False,
        sort_by_x: bool = False,
    ) -> Tuple[List[float], np.ndarray, np.ndarray]:
        """Extract y data with mean and standard deviation across runs.

        Args:
            x_param: X-axis parameter name
            y_param: Y-axis parameter name
            use_design_var_as_x: If True, use design variable for x instead of parameter
            sort_by_x: If True, sort the data by x values

        Returns:
            Tuple of (x_values, y_mean, y_std)
        """
        all_x_values = []
        all_y_values = []

        for run_index in range(len(self.opt_results)):
            x_values, y_values = self.extract_xy_data(
                x_param, y_param, run_index, use_design_var_as_x, sort_by_x
            )

            if not x_values or not y_values:
                continue

            all_x_values.append(x_values)
            all_y_values.append(y_values)

        if not all_x_values:
            return [], np.array([]), np.array([])

        # Use first run's x values as reference
        x_ref = all_x_values[0]

        # Ensure all series have the same x values for statistics
        y_matrix = np.array(
            [y_values for y_values in all_y_values if len(y_values) == len(x_ref)]
        )

        if y_matrix.size == 0:
            return x_ref, np.array([]), np.array([])

        y_mean = np.mean(y_matrix, axis=0)
        y_std = np.std(y_matrix, axis=0)

        return x_ref, y_mean, y_std


class OptimizationPlotter:
    """Handles plotting of optimization progress data."""

    def __init__(
        self,
        data_extractor: DataExtractor,
        plot_variance: bool = False,
        save_figures: bool = False,
        save_path: Optional[str] = None
    ):
        """Initialize the plotter.

        Args:
            data_extractor: Data extractor instance
            plot_variance: If True, plot mean and variance across runs
            save_figures: If True, save generated figures to disk
        """
        self.extractor = data_extractor
        self.plot_variance = plot_variance
        self.save_figures = save_figures
        self.num_runs = len(data_extractor.opt_results)
        self.save_path = save_path
        

    def _setup_ax(
        self,
        ax: Axes,
        config: OptPltSet,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
    ) -> None:
        """Set up an axis with proper labels, scales, and formatting.

        Args:
            ax: Matplotlib axis
            config: Plot configuration
            x_label: Custom x-axis label (overrides config)
            y_label: Custom y-axis label (overrides config)
        """
        if ax.get_legend() is not None:
            ax.get_legend().remove()

        ax.set_xlabel(x_label if x_label is not None else config.get_x_label())

        if y_label is not None:
            ax.set_ylabel(f"{y_label} ({config.unit})")
        else:
            ax.set_ylabel(f"{config.get_y_label()} ({config.unit})")

        ax.set_xscale(config.x_scale)
        ax.set_yscale(config.y_scale)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    def _plot_target(
        self,
        ax: Axes,
        param_name: str,
        x_min: float,
        x_max: float,
        color: str,
        normalization: float,
    ) -> bool:
        """Plot horizontal target line if target exists for the parameter.

        Args:
            ax: Matplotlib axis
            param_name: Parameter name
            x_min: Minimum x value
            x_max: Maximum x value
            color: Line color
            normalization: Value normalization factor

        Returns:
            True if target was plotted, False otherwise
        """
        if param_name not in self.extractor.system_target_params:
            return False

        target_value = self.extractor.system_target_params[param_name]
        normalized_target = target_value / normalization
        if x_min == x_max:
            ax.scatter(
                x_min, normalized_target, color=color, label="target", marker="*", s=50
            )
        else:
            ax.axhline(normalized_target, ls="--", color=color, label="target", lw=2)
        return True

    def plot_standard(
        self,
        fig: Figure,
        axes: Union[Axes, List[Axes]],
        plot_settings: List[OptPltSet],
        plot_name: str,
    ) -> None:
        """Create standard parameter vs. iteration plots.

        Args:
            fig: Matplotlib figure
            axes: Single axis or list of axes
            plot_settings: List of plot configurations
            plot_name: Name for the plot (used for saving)
        """
        if len(plot_settings) == 1:
            axes = [cast(Axes, axes)]
        else:
            axes = cast(List[Axes], axes)

        colors = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

        for _, (ax, config) in enumerate(zip(axes, plot_settings)):
            color = next(colors)

            if isinstance(config.y, str):
                self._plot_single_param(ax, config, config.y, color)
            else:
                for y_idx, y_param in enumerate(config.y):
                    curr_color = color if y_idx == 0 else f"C{y_idx}"
                    self._plot_single_param(ax, config, y_param, curr_color)

            ax.legend()

        fig.suptitle(plot_name)
        fig.subplots_adjust(hspace=0.5)

        if self.save_figures:
            if self.save_path is None:
                fig.savefig(f"{self.save_path}_optimization_plot_{plot_name}.png")
            else:
                fig.savefig(
                    f"optimization_plot_{time.strftime('%Y%m%d-%H%M%S')}_{plot_name}.png"
                )


    def _plot_single_param(
        self, ax: Axes, config: OptPltSet, y_param: str, color: str, **kwargs: Any
    ) -> None:
        """Plot a single parameter series.

        Args:
            ax: Matplotlib axis
            config: Plot configuration
            y_param: Y-axis parameter name
            color: Line color
        """
        self._setup_ax(ax, config)

        if self.plot_variance and self.num_runs > 1:
            # Plot mean and variance
            x_values, y_mean, y_std = self.extractor.get_y_data_with_statistics(
                config.x, y_param
            )

            if len(x_values) > 0:
                normalized_mean = y_mean / config.normalization
                normalized_std = y_std / config.normalization

                ax.plot(
                    x_values, normalized_mean, "o-", label="optimized mean", color=color, **kwargs
                )

                ax.fill_between(
                    x_values,
                    normalized_mean - normalized_std,
                    normalized_mean + normalized_std,
                    alpha=0.3,
                    facecolor=color,
                )

                if x_values:
                    self._plot_target(
                        ax,
                        y_param,
                        min(x_values),
                        max(x_values),
                        color,
                        config.normalization,
                    )
        else:
            # Plot individual runs
            for run_idx in range(self.num_runs):
                x_values, y_values = self.extractor.extract_xy_data(
                    config.x, y_param, run_idx
                )

                if x_values and y_values:
                    normalized_y = np.array(y_values) / config.normalization

                    run_label = "optimized"
                    if self.num_runs > 1:
                        run_label += f" {run_idx+1}"

                    ax.plot(x_values, normalized_y, "o-", label=run_label, color=color, **kwargs)

                    self._plot_target(
                        ax,
                        y_param,
                        min(x_values),
                        max(x_values),
                        color,
                        config.normalization,
                    )

    def plot_params_vs_design_vars(
        self,
        fig: Figure,
        axes: Union[Axes, List[Axes]],
        plot_settings: List[OptPltSet],
        plot_name: str,
        sort_by_x: bool = True,
    ) -> None:
        """Create plots of parameters vs. their associated design variables.

        Args:
            fig: Matplotlib figure
            axes: Single axis or list of axes
            plot_settings: List of plot configurations
            plot_name: Name for the plot (used for saving)
            sort_by_x: If True, sort data points by x value
        """
        if len(plot_settings) == 1:
            axes = [cast(Axes, axes)]
        else:
            axes = cast(List[Axes], axes)

        if not self.extractor.opt_target_list:
            raise ValueError(
                "Cannot plot against design variables without optimization targets"
            )

        colors = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

        for ax, config in zip(axes, plot_settings):
            color = next(colors)

            if isinstance(config.y, str):
                self._plot_param_vs_design_var(ax, config, config.y, color, sort_by_x)
            else:
                for y_idx, y_param in enumerate(config.y):
                    curr_color = color if y_idx == 0 else f"C{y_idx}"
                    self._plot_param_vs_design_var(
                        ax, config, y_param, curr_color, sort_by_x
                    )

            ax.legend()

        fig.suptitle(f"{plot_name} vs Design Variables")
        fig.subplots_adjust(hspace=0.5)

        if self.save_figures:
            sorting = "sorted" if sort_by_x else "chronological"
            fig.savefig(
                f"optimization_plot_{time.strftime('%Y%m%d-%H%M%S')}_{plot_name}_vs_design_vars_{sorting}.png"
            )

    def _plot_param_vs_design_var(
        self, ax: Axes, config: OptPltSet, y_param: str, color: str, sort_by_x: bool, **kwargs: Any
    ) -> None:
        """Plot a parameter vs. its associated design variable.

        Args:
            ax: Matplotlib axis
            config: Plot configuration
            y_param: Y-axis parameter name
            color: Line color
            sort_by_x: If True, sort data points by x value
        """
        # Get design variable information for labeling
        first_result = self.extractor.opt_results[0][0]
        design_var_name = self.extractor.get_design_var_name_for_param(y_param)
        _, design_var_unit = self.extractor.get_design_var_for_param(
            y_param, first_result
        )

        # Set up the axis with proper labels
        self._setup_ax(ax, config, x_label=f"{design_var_name} ({design_var_unit})")

        for run_idx in range(self.num_runs):
            x_values, y_values = self.extractor.extract_xy_data(
                "", y_param, run_idx, use_design_var_as_x=True, sort_by_x=sort_by_x
            )

            if not x_values or not y_values:
                continue

            normalized_y = np.array(y_values) / config.normalization

            run_label = "optimized"
            if self.num_runs > 1:
                run_label += f" {run_idx+1}"

            ax.plot(x_values, normalized_y, "o-", label=run_label, color=color, **kwargs)

            if x_values:
                self._plot_target(
                    ax,
                    y_param,
                    min(x_values),
                    max(x_values),
                    color,
                    config.normalization,
                )

    def plot_design_vars_vs_iteration(
        self,
        fig: Figure,
        axes: Union[Axes, List[Axes]],
        plot_settings: List[OptPltSet],
        plot_name: str,
    ) -> None:
        """Create plots of design variables vs. iteration.

        Args:
            fig: Matplotlib figure
            axes: Single axis or list of axes
            plot_settings: List of plot configurations
            plot_name: Name for the plot (used for saving)
        """
        if len(plot_settings) == 1:
            axes = [cast(Axes, axes)]
        else:
            axes = cast(List[Axes], axes)

        if not self.extractor.opt_target_list:
            raise ValueError(
                "Cannot plot design variables without optimization targets"
            )

        colors = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

        for ax, config in zip(axes, plot_settings):
            color = next(colors)

            if isinstance(config.y, str):
                self._plot_design_var_vs_iteration(ax, config, config.y, color)
            else:
                for y_idx, y_param in enumerate(config.y):
                    curr_color = color if y_idx == 0 else f"C{y_idx}"
                    self._plot_design_var_vs_iteration(ax, config, y_param, curr_color)

            ax.legend()

        fig.suptitle(f"Design Variables for {plot_name}")
        fig.subplots_adjust(hspace=0.5)

        if self.save_figures:
            fig.savefig(
                f"optimization_plot_{time.strftime('%Y%m%d-%H%M%S')}_{plot_name}_design_vars.png"
            )

    def _plot_design_var_vs_iteration(
        self, ax: Axes, config: OptPltSet, y_param: str, color: str
    ) -> None:
        """Plot a design variable vs. iteration.

        Args:
            ax: Matplotlib axis
            config: Plot configuration
            y_param: Parameter name (used to find associated design variable)
            color: Line color
        """
        # Get design variable information for labeling
        first_result = self.extractor.opt_results[0][0]
        design_var_name = self.extractor.get_design_var_name_for_param(y_param)
        _, design_var_unit = self.extractor.get_design_var_for_param(
            y_param, first_result
        )

        # Set up the axis with proper labels
        self._setup_ax(ax, config, y_label=f"{design_var_name} ({design_var_unit})")

        for run_idx in range(self.num_runs):
            opt_result = self.extractor.opt_results[run_idx]

            x_values = [
                self.extractor.get_parameter_value(config.x, result, i)
                for i, result in enumerate(opt_result)
            ]

            # Get design variable values associated with the target parameter
            y_values = [
                self.extractor.get_design_var_for_param(y_param, result)[0]
                for result in opt_result
            ]

            # Filter out None values
            valid_pairs = [
                (x, y)
                for x, y in zip(x_values, y_values)
                if x is not None and y is not None
            ]

            if not valid_pairs:
                continue

            x_filtered, y_filtered = zip(*valid_pairs)

            run_label = f"{design_var_name} for {y_param}"
            if self.num_runs > 1:
                run_label += f" (run {run_idx+1})"

            ax.plot(x_filtered, y_filtered, "o-", label=run_label, color=color)


def plot_progress(
    opt_results: List[List[OptimizationResult]],
    system_target_params: SystemParams,
    plot_settings: PlotSettingsType,
    block_plots: bool = False,
    save_figures: bool = False,
    plot_variance: bool = False,
    plot_design_variables: Optional[Literal["chronological", "sorted"]] = None,
    opt_target_list: Optional[List[OptTarget]] = None,
    save_path: Optional[str] = None
) -> None:
    """Plot the progress of optimization iterations.

    Args:
        opt_results: List of optimization result sequences (one per optimization run)
        system_target_params: Target system parameters
        plot_settings: Plot configurations by plot name
        block_plots: If True, block execution until plots are closed
        save_figures: If True, save figures to disk
        plot_variance: If True, plot mean and variance across runs
        plot_design_variables: How to plot design variables ("chronological", "sorted", or None to disable)
        opt_target_list: Optional list of optimization targets
    """
    # Validate input arguments
    if plot_design_variables is not None and plot_design_variables not in [
        "chronological",
        "sorted",
    ]:
        raise ValueError(
            "plot_design_variables must be None, 'chronological', or 'sorted'"
        )

    # Close existing figures
    plt.close("all")

    # Create data extractor and plotter
    data_extractor = DataExtractor(opt_results, system_target_params, opt_target_list)
    plotter = OptimizationPlotter(data_extractor, plot_variance, save_figures)
    plotter.save_path = save_path

    # Create standard parameter plots
    for plot_name, plot_setting in plot_settings.items():
        fig, axs = plt.subplots(
            len(plot_setting), figsize=(6.4, 2.4 * len(plot_setting))
        )
        plotter.plot_standard(fig, axs, plot_setting, plot_name)

        # Create additional plot types if requested
        if plot_design_variables is not None:
            if opt_target_list is None:
                raise ValueError(
                    "opt_target_list is required when plot_design_variables is set"
                )

            # Plot parameters vs design variables
            fig, axs = plt.subplots(len(plot_setting))
            plotter.plot_params_vs_design_vars(
                fig,
                axs,
                plot_setting,
                plot_name,
                sort_by_x=(plot_design_variables == "sorted"),
            )

            # Plot design variables vs iteration
            fig, axs = plt.subplots(len(plot_setting))
            plotter.plot_design_vars_vs_iteration(fig, axs, plot_setting, plot_name)

    # Show plots
    plt.show(block=block_plots)
