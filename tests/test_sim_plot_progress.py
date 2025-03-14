from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from qdesignoptimizer.design_analysis_types import OptTarget
from qdesignoptimizer.sim_plot_progress import (
    DataExtractor,
    OptimizationPlotter,
    OptPltSet,
    UnitEnum,
    plot_progress,
)
from qdesignoptimizer.utils.names_parameters import ITERATION, param, param_nonlin
from qdesignoptimizer.utils.utils import get_value_and_unit


class TestOptPltSet:
    """Tests for the OptPltSet class."""

    def test_init_single_y(self):
        """Test OptPltSet initialization with a single y value."""
        plt_set = OptPltSet("x_param", "y_param", "X Label", "Y Label", "log", "log")

        assert plt_set.x == "x_param"
        assert plt_set.y == "y_param"
        assert plt_set.x_label == "X Label"
        assert plt_set.y_label == "Y Label"
        assert plt_set.x_scale == "log"
        assert plt_set.y_scale == "log"
        assert plt_set.unit == UnitEnum.HZ

    def test_init_multiple_y(self):
        """Test OptPltSet initialization with multiple y values."""
        y_values = ["y_param1", "y_param2", "y_param3"]
        plt_set = OptPltSet("x_param", y_values)

        assert plt_set.x == "x_param"
        assert plt_set.y == y_values
        assert plt_set.x_label is None  # No custom label
        assert plt_set.y_label is None  # No custom label
        assert plt_set.x_scale == "linear"  # Default scale
        assert plt_set.y_scale == "linear"  # Default scale

    def test_get_labels(self):
        """Test getting labels with and without custom labels."""
        # With custom labels
        plt_set = OptPltSet("x_param", "y_param", "Custom X", "Custom Y")
        assert plt_set.get_x_label() == "Custom X"
        assert plt_set.get_y_label() == "Custom Y"

        # Without custom labels
        plt_set = OptPltSet("x_param", "y_param")
        assert plt_set.get_x_label() == "x_param"
        assert plt_set.get_y_label() == "y_param"

        # With multiple y values and no custom label
        plt_set = OptPltSet("x_param", ["y1", "y2", "y3"])
        assert plt_set.get_y_label() == "y1, y2, y3"

    def test_normalization(self):
        """Test normalization factors for different units."""
        # Test Hz
        plt_set = OptPltSet("x", "y", unit=UnitEnum.HZ)
        assert plt_set.normalization == 1

        # Test kHz
        plt_set = OptPltSet("x", "y", unit=UnitEnum.KHZ)
        assert plt_set.normalization == 1e3

        # Test MHz
        plt_set = OptPltSet("x", "y", unit=UnitEnum.MHZ)
        assert plt_set.normalization == 1e6

        # Test GHz
        plt_set = OptPltSet("x", "y", unit=UnitEnum.GHZ)
        assert plt_set.normalization == 1e9


@pytest.fixture
def mock_optimization_results():
    """Create mock optimization results."""
    qubit_freq = param("qubit_1", "freq")
    resonator_freq = param("resonator_1", "freq")
    cross_kerr = param_nonlin("qubit_1", "resonator_1")

    # Create three iterations of results
    return [
        {
            "system_optimized_params": {
                qubit_freq: 5.0e9,
                resonator_freq: 7.0e9,
                cross_kerr: 1.0e6,
            },
            "design_variables": {
                "design_var_lj_qubit_1": "10.0nH",
                "design_var_width_qubit_1": "20.0um",
                "design_var_length_resonator_1": "5000.0um",
            },
        },
        {
            "system_optimized_params": {
                qubit_freq: 5.1e9,
                resonator_freq: 6.9e9,
                cross_kerr: 1.1e6,
            },
            "design_variables": {
                "design_var_lj_qubit_1": "9.8nH",
                "design_var_width_qubit_1": "20.5um",
                "design_var_length_resonator_1": "5100.0um",
            },
        },
        {
            "system_optimized_params": {
                qubit_freq: 5.2e9,
                resonator_freq: 6.8e9,
                cross_kerr: 1.2e6,
            },
            "design_variables": {
                "design_var_lj_qubit_1": "9.5nH",
                "design_var_width_qubit_1": "21.0um",
                "design_var_length_resonator_1": "5200.0um",
            },
        },
    ]


@pytest.fixture
def mock_system_target_params():
    """Create mock system target parameters."""
    qubit_freq = param("qubit_1", "freq")
    resonator_freq = param("resonator_1", "freq")
    cross_kerr = param_nonlin("qubit_1", "resonator_1")

    return {
        qubit_freq: 5.2e9,
        resonator_freq: 6.8e9,
        cross_kerr: 1.2e6,
    }


@pytest.fixture
def mock_opt_target_list():
    """Create mock optimization target list."""
    from qdesignoptimizer.utils.names_parameters import FREQ, NONLIN

    qubit = "qubit_1"
    resonator = "resonator_1"

    return [
        OptTarget(
            target_param_type=FREQ,
            involved_modes=[qubit],
            design_var="design_var_lj_qubit_1",
            design_var_constraint={"larger_than": "5nH", "smaller_than": "20nH"},
            prop_to=lambda p, v: 1.0,
            independent_target=True,
        ),
        OptTarget(
            target_param_type=FREQ,
            involved_modes=[resonator],
            design_var="design_var_length_resonator_1",
            design_var_constraint={"larger_than": "4000um", "smaller_than": "6000um"},
            prop_to=lambda p, v: 1.0,
            independent_target=True,
        ),
        OptTarget(
            target_param_type=NONLIN,
            involved_modes=[qubit, resonator],
            design_var="design_var_width_qubit_1",
            design_var_constraint={"larger_than": "10um", "smaller_than": "30um"},
            prop_to=lambda p, v: 1.0,
            independent_target=False,
        ),
    ]


@pytest.fixture
def mock_plot_settings():
    """Create mock plot settings."""
    qubit_freq = param("qubit_1", "freq")
    resonator_freq = param("resonator_1", "freq")
    cross_kerr = param_nonlin("qubit_1", "resonator_1")
    design_var_lj = "design_var_lj_qubit_1"

    return {
        "Frequencies": [
            OptPltSet(
                ITERATION, [qubit_freq, resonator_freq], "Iteration", "Frequency (Hz)"
            ),
        ],
        "Cross-Kerr vs LJ": [
            OptPltSet(design_var_lj, cross_kerr, "Inductance (nH)", "Cross-Kerr (Hz)"),
        ],
    }


class TestDataExtractor:
    """Tests for the DataExtractor class."""

    def test_get_parameter_value(self, mock_optimization_results):
        """Test extracting parameter values from results."""
        result = mock_optimization_results[0]
        extractor = DataExtractor([mock_optimization_results], {})

        # Test iteration parameter
        assert extractor.get_parameter_value(ITERATION, result, 0) == 1

        # Test system parameter
        qubit_freq = param("qubit_1", "freq")
        assert extractor.get_parameter_value(qubit_freq, result, 0) == 5.0e9

        # Test design variable
        design_var = "design_var_lj_qubit_1"
        assert extractor.get_parameter_value(design_var, result, 0) == 10.0

        # Test parameter not found
        assert extractor.get_parameter_value("nonexistent", result, 0) is None

    def test_get_design_var_name_for_param(self, mock_opt_target_list):
        """Test finding design variable name associated with a parameter."""
        extractor = DataExtractor([], {}, mock_opt_target_list)

        # Test frequency parameter
        freq_param = param("qubit_1", "freq")
        assert (
            extractor.get_design_var_name_for_param(freq_param)
            == "design_var_lj_qubit_1"
        )

        # Test nonlinearity parameter
        nonlin_param = param_nonlin("qubit_1", "resonator_1")
        assert (
            extractor.get_design_var_name_for_param(nonlin_param)
            == "design_var_width_qubit_1"
        )

        # Test parameter not found
        with pytest.raises(AssertionError):
            extractor.get_design_var_name_for_param("nonexistent_param")

    def test_extract_xy_data(
        self, mock_optimization_results, mock_system_target_params
    ):
        """Test extracting x and y data series."""
        extractor = DataExtractor(
            [mock_optimization_results], mock_system_target_params
        )

        # Test extracting iteration vs frequency
        qubit_freq = param("qubit_1", "freq")
        x_values, y_values = extractor.extract_xy_data(ITERATION, qubit_freq, 0)

        assert x_values == [1, 2, 3]
        assert y_values == [5.0e9, 5.1e9, 5.2e9]

        # Test extracting design variable vs frequency
        design_var = "design_var_lj_qubit_1"
        x_values, y_values = extractor.extract_xy_data(design_var, qubit_freq, 0)

        assert x_values == [10.0, 9.8, 9.5]
        assert y_values == [5.0e9, 5.1e9, 5.2e9]

    def test_get_y_data_with_statistics(
        self, mock_optimization_results, mock_system_target_params
    ):
        """Test extracting y data with statistics across runs."""
        # Create multiple runs with identical data for testing
        multiple_runs = [mock_optimization_results, mock_optimization_results]
        extractor = DataExtractor(multiple_runs, mock_system_target_params)

        # Test statistics for iteration vs frequency
        qubit_freq = param("qubit_1", "freq")
        x_values, y_mean, y_std = extractor.get_y_data_with_statistics(
            ITERATION, qubit_freq
        )

        assert x_values == [1, 2, 3]
        np.testing.assert_array_equal(y_mean, np.array([5.0e9, 5.1e9, 5.2e9]))
        np.testing.assert_array_equal(
            y_std, np.array([0.0, 0.0, 0.0])
        )  # Same data, so std = 0

        # Test with varying data
        varying_run = [
            {
                "system_optimized_params": {
                    qubit_freq: 5.1e9,
                },
                "design_variables": {},
            },
            {
                "system_optimized_params": {
                    qubit_freq: 5.2e9,
                },
                "design_variables": {},
            },
            {
                "system_optimized_params": {
                    qubit_freq: 5.3e9,
                },
                "design_variables": {},
            },
        ]

        varying_runs = [mock_optimization_results, varying_run]
        extractor = DataExtractor(varying_runs, mock_system_target_params)

        x_values, y_mean, y_std = extractor.get_y_data_with_statistics(
            ITERATION, qubit_freq
        )

        assert x_values == [1, 2, 3]
        np.testing.assert_array_almost_equal(y_mean, np.array([5.05e9, 5.15e9, 5.25e9]))
        np.testing.assert_array_almost_equal(y_std, np.array([0.05e9, 0.05e9, 0.05e9]))


class TestOptimizationPlotter:
    """Tests for the OptimizationPlotter class."""

    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.show")
    def test_plot_standard(
        self,
        mock_show,
        mock_subplots,
        mock_figure,
        mock_optimization_results,
        mock_system_target_params,
    ):
        """Test plotting standard parameter vs. iteration plots."""
        # Mock the subplot and axes objects
        mock_fig = MagicMock()
        mock_axes = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_axes)

        extractor = DataExtractor(
            [mock_optimization_results], mock_system_target_params
        )
        plotter = OptimizationPlotter(extractor)

        # Test single plot
        qubit_freq = param("qubit_1", "freq")
        config = OptPltSet(ITERATION, qubit_freq, "Iteration", "Frequency")

        plotter.plot_standard(mock_fig, mock_axes, [config], "Test Plot")

        # Verify setup was called
        mock_axes.set_xlabel.assert_called_with("Iteration")
        mock_axes.set_ylabel.assert_called_with("Frequency (Hz)")
        mock_fig.suptitle.assert_called_with("Test Plot")

    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.show")
    def test_plot_params_vs_design_vars(
        self,
        mock_show,
        mock_subplots,
        mock_figure,
        mock_optimization_results,
        mock_system_target_params,
        mock_opt_target_list,
    ):
        """Test plotting parameters vs. design variables."""
        # Mock the subplot and axes objects
        mock_fig = MagicMock()
        mock_axes = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_axes)

        extractor = DataExtractor(
            [mock_optimization_results], mock_system_target_params, mock_opt_target_list
        )
        plotter = OptimizationPlotter(extractor)

        # Test plotting parameter vs design variable
        qubit_freq = param("qubit_1", "freq")
        config = OptPltSet(ITERATION, qubit_freq, "Iteration", "Frequency")

        plotter.plot_params_vs_design_vars(mock_fig, mock_axes, [config], "Test Plot")

        # Verify setup was called with design variable name in label
        # The _setup_ax method will be called but we can't directly test the label
        # since it's determined dynamically based on the design variable
        mock_fig.suptitle.assert_called_with("Test Plot vs Design Variables")

    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.show")
    def test_plot_design_vars_vs_iteration(
        self,
        mock_show,
        mock_subplots,
        mock_figure,
        mock_optimization_results,
        mock_system_target_params,
        mock_opt_target_list,
    ):
        """Test plotting design variables vs. iteration."""
        # Mock the subplot and axes objects
        mock_fig = MagicMock()
        mock_axes = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_axes)

        extractor = DataExtractor(
            [mock_optimization_results], mock_system_target_params, mock_opt_target_list
        )
        plotter = OptimizationPlotter(extractor)

        # Test plotting design variables vs iteration
        qubit_freq = param("qubit_1", "freq")
        config = OptPltSet(ITERATION, qubit_freq, "Iteration", "Frequency")

        plotter.plot_design_vars_vs_iteration(
            mock_fig, mock_axes, [config], "Test Plot"
        )

        # Verify figure title is set correctly
        mock_fig.suptitle.assert_called_with("Design Variables for Test Plot")


class TestPlotProgress:
    """Tests for the plot_progress function."""

    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.show")
    def test_plot_progress(
        self,
        mock_show,
        mock_subplots,
        mock_figure,
        mock_optimization_results,
        mock_system_target_params,
        mock_plot_settings,
    ):
        """Test that plot_progress creates the expected plots."""
        # Mock the subplot and axes objects
        mock_fig = MagicMock()
        mock_axes = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_axes)

        # Run the function
        plot_progress(
            [mock_optimization_results],
            mock_system_target_params,
            mock_plot_settings,
            block_plots=False,
        )

        # Check the right number of plots were created
        assert mock_subplots.call_count == len(
            mock_plot_settings
        ), "Wrong number of subplots created"
        assert mock_fig.suptitle.call_count == len(
            mock_plot_settings
        ), "Wrong number of titles set"
        assert mock_show.call_count == 1, "plt.show() should be called once"

    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.close")
    def test_plot_progress_with_design_variables(
        self,
        mock_close,
        mock_show,
        mock_subplots,
        mock_figure,
        mock_optimization_results,
        mock_system_target_params,
        mock_plot_settings,
        mock_opt_target_list,
    ):
        """Test plotting with design variables option."""
        # Mock the subplot and axes objects
        mock_fig = MagicMock()
        mock_axes = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_axes)

        # Run the function with design variables
        plot_progress(
            [mock_optimization_results],
            mock_system_target_params,
            mock_plot_settings,
            block_plots=False,
            plot_design_variables="sorted",
            opt_target_list=mock_opt_target_list,
        )

        # Should create 3x the number of figures (standard + params vs design + design vs iteration)
        assert mock_subplots.call_count == len(mock_plot_settings) * 3
        assert mock_show.call_count == 1, "plt.show() should be called once"

    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.show")
    def test_plot_progress_with_variance(
        self,
        mock_show,
        mock_subplots,
        mock_figure,
        mock_optimization_results,
        mock_system_target_params,
        mock_plot_settings,
    ):
        """Test plotting with variance option."""
        # Mock the subplot and axes objects
        mock_fig = MagicMock()
        mock_axes = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_axes)

        # Create multiple runs for variance plotting
        multiple_runs = [mock_optimization_results, mock_optimization_results]

        # Run the function with variance
        plot_progress(
            multiple_runs,
            mock_system_target_params,
            mock_plot_settings,
            block_plots=False,
            plot_variance=True,
        )

        # Should create the standard number of figures
        assert mock_subplots.call_count == len(mock_plot_settings)
        assert mock_show.call_count == 1, "plt.show() should be called once"

    def test_plot_progress_invalid_args(self):
        """Test that invalid arguments raise appropriate errors."""
        # Test invalid plot_design_variables value
        with pytest.raises(ValueError):
            plot_progress(
                [[]],  # Empty list of optimization results
                {},  # Empty system target params
                {},  # Empty plot settings
                plot_design_variables="invalid",
            )

        # Test missing opt_target_list when plot_design_variables is set
        with pytest.raises(ValueError):
            plot_progress(
                [[]],  # Empty list of optimization results
                {},  # Empty system target params
                {"Test": [OptPltSet("x", "y")]},  # Non-empty plot settings
                plot_design_variables="sorted",
                opt_target_list=None,
            )
