from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from qdesignoptimizer.sim_plot_progress import OptPltSet, plot_progress
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

    def test_init_multiple_y(self):
        """Test OptPltSet initialization with multiple y values."""
        y_values = ["y_param1", "y_param2", "y_param3"]
        plt_set = OptPltSet("x_param", y_values)

        assert plt_set.x == "x_param"
        assert plt_set.y == y_values
        assert plt_set.x_label == "x_param"  # Default label
        assert plt_set.y_label == y_values  # Default label
        assert plt_set.x_scale == "linear"  # Default scale
        assert plt_set.y_scale == "linear"  # Default scale

    def test_default_labels(self):
        """Test that default labels are set correctly when not provided."""
        plt_set = OptPltSet("x_param", "y_param")

        assert plt_set.x_label == "x_param"
        assert plt_set.y_label == "y_param"

    def test_get_label_custom(self):
        """Test that _get_label returns custom label when provided."""
        plt_set = OptPltSet("x_param", "y_param", "Custom X", "Custom Y")

        assert plt_set._get_label("x_param", "Custom X") == "Custom X"
        assert plt_set._get_label("y_param", "Custom Y") == "Custom Y"

    def test_get_label_default(self):
        """Test that _get_label returns variable name when no custom label is provided."""
        plt_set = OptPltSet("x_param", "y_param")

        assert plt_set._get_label("x_param", None) == "x_param"
        assert plt_set._get_label("y_param", None) == "y_param"

    def test_init_with_empty_y_list(self):
        """Test initialization with an empty y list."""
        plt_set = OptPltSet("x_param", [])

        assert plt_set.x == "x_param"
        assert plt_set.y == []
        assert plt_set.x_label == "x_param"
        assert plt_set.y_label == []


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
            mock_optimization_results,
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

    @patch("matplotlib.pyplot.close")
    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.show")
    def test_plot_progress_with_iteration(
        self,
        mock_show,
        mock_subplots,
        mock_close,
        mock_optimization_results,
        mock_system_target_params,
    ):
        """Test plotting with iteration as x-axis."""
        # Mock the subplot and axes objects
        mock_fig = MagicMock()
        mock_axes = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_axes)

        qubit_freq = param("qubit_1", "freq")

        # Create plot settings with iteration as x-axis
        plot_settings = {
            "Qubit Frequency Progress": [
                OptPltSet(ITERATION, qubit_freq, "Iteration", "Frequency (Hz)"),
            ]
        }

        # Run the function
        plot_progress(
            mock_optimization_results,
            mock_system_target_params,
            plot_settings,
            block_plots=False,
        )

        # Verify close was called to clear previous plots
        assert mock_close.call_count == 1

    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.show")
    def test_plot_progress_with_design_variable(
        self,
        mock_show,
        mock_subplots,
        mock_optimization_results,
        mock_system_target_params,
    ):
        """Test plotting with design variable as x-axis."""
        # Mock the subplot and axes objects
        mock_fig = MagicMock()
        mock_axes = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_axes)

        qubit_freq = param("qubit_1", "freq")
        design_var_lj = "design_var_lj_qubit_1"

        # Create plot settings with design variable as x-axis
        plot_settings = {
            "Frequency vs LJ": [
                OptPltSet(design_var_lj, qubit_freq, "LJ (nH)", "Frequency (Hz)"),
            ]
        }

        # Run the function
        plot_progress(
            mock_optimization_results,
            mock_system_target_params,
            plot_settings,
            block_plots=False,
        )

        # Verify subplots was called with the right number of panels
        mock_subplots.assert_called_once()

    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.show")
    def test_plot_progress_multiple_panels(
        self,
        mock_show,
        mock_subplots,
        mock_optimization_results,
        mock_system_target_params,
    ):
        """Test plotting with multiple panels in a single figure."""
        # Mock the subplot and axes objects
        mock_fig = MagicMock()
        mock_axes = MagicMock()
        mock_subplots.return_value = (
            mock_fig,
            [mock_axes, mock_axes],
        )  # Return list of axes

        qubit_freq = param("qubit_1", "freq")
        resonator_freq = param("resonator_1", "freq")

        # Create plot settings with multiple panels
        plot_settings = {
            "Frequency Progress": [
                OptPltSet(ITERATION, qubit_freq, "Iteration", "Qubit Frequency (Hz)"),
                OptPltSet(
                    ITERATION, resonator_freq, "Iteration", "Resonator Frequency (Hz)"
                ),
            ]
        }

        # Run the function
        plot_progress(
            mock_optimization_results,
            mock_system_target_params,
            plot_settings,
            block_plots=False,
        )

        # Verify subplots was called with 2 panels
        mock_subplots.assert_called_once_with(2)

    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.show")
    def test_plot_progress_with_missing_data(
        self,
        mock_show,
        mock_subplots,
        mock_optimization_results,
        mock_system_target_params,
    ):
        """Test plotting with missing data points."""
        # Mock the subplot and axes objects
        mock_fig = MagicMock()
        mock_axes = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_axes)

        # Create a param that doesn't exist in the results
        missing_param = param("missing_mode", "freq")

        # Create plot settings with the missing parameter
        plot_settings = {
            "Missing Data Test": [
                OptPltSet(ITERATION, missing_param, "Iteration", "Frequency (Hz)"),
            ]
        }

        # Run the function - should not crash on missing data
        plot_progress(
            mock_optimization_results,
            mock_system_target_params,
            plot_settings,
            block_plots=False,
        )

        # Verify the function completed without error
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.show")
    def test_plot_progress_log_scale(
        self,
        mock_show,
        mock_subplots,
        mock_optimization_results,
        mock_system_target_params,
    ):
        """Test plotting with log scales."""
        # Mock the subplot and axes objects
        mock_fig = MagicMock()
        mock_axes = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_axes)

        qubit_freq = param("qubit_1", "freq")

        # Create plot settings with log scales
        plot_settings = {
            "Log Scale Test": [
                OptPltSet(
                    ITERATION,
                    qubit_freq,
                    "Iteration",
                    "Frequency (Hz)",
                    x_scale="log",
                    y_scale="log",
                ),
            ]
        }

        # Run the function
        plot_progress(
            mock_optimization_results,
            mock_system_target_params,
            plot_settings,
            block_plots=False,
        )

        # Verify that set_xscale and set_yscale were called with "log"
        mock_axes.set_xscale.assert_called_with("log")
        mock_axes.set_yscale.assert_called_with("log")


@pytest.fixture
def realistic_optimization_data():
    """Create a more realistic set of optimization data."""
    # Define parameter names
    qubit_freq = param("qubit_1", "freq")
    resonator_freq = param("resonator_1", "freq")
    qubit_kappa = param("qubit_1", "kappa")
    resonator_kappa = param("resonator_1", "kappa")
    anharmonicity = param_nonlin("qubit_1", "qubit_1")
    cross_kerr = param_nonlin("qubit_1", "resonator_1")

    # Create iterations with improving results
    iterations = []
    for i in range(5):
        # Gradually improve parameters toward target
        progress = i / 4.0  # 0 to 1

        # System parameters get closer to target
        iterations.append(
            {
                "system_optimized_params": {
                    qubit_freq: 5.0e9 + progress * 0.2e9,  # 5.0 to 5.2 GHz
                    resonator_freq: 7.0e9 - progress * 0.2e9,  # 7.0 to 6.8 GHz
                    qubit_kappa: 1e4 + progress * 1e4,  # 10 to 20 kHz
                    resonator_kappa: 5e5 + progress * 1e5,  # 500 to 600 kHz
                    anharmonicity: -200e6 - progress * 20e6,  # -200 to -220 MHz
                    cross_kerr: 0.8e6 + progress * 0.4e6,  # 0.8 to 1.2 MHz
                },
                "design_variables": {
                    "design_var_lj_qubit_1": f"{10.0 - progress * 0.5}nH",  # 10.0 to 9.5 nH
                    "design_var_width_qubit_1": f"{20.0 + progress * 1.0}um",  # 20.0 to 21.0 um
                    "design_var_length_resonator_1": f"{5000.0 + progress * 200.0}um",  # 5000 to 5200 um
                    "design_var_coupl_length_resonator_1_qubit_1": f"{100.0 + progress * 10.0}um",  # 100 to 110 um
                },
            }
        )

    # System target parameters
    system_target_params = {
        qubit_freq: 5.2e9,
        resonator_freq: 6.8e9,
        qubit_kappa: 2e4,
        resonator_kappa: 6e5,
        anharmonicity: -220e6,
        cross_kerr: 1.2e6,
    }

    # Plot settings with multiple figures and panels
    plot_settings = {
        "Frequencies": [
            OptPltSet(
                ITERATION, [qubit_freq, resonator_freq], "Iteration", "Frequency (Hz)"
            ),
        ],
        "Linewidths": [
            OptPltSet(
                ITERATION,
                [qubit_kappa, resonator_kappa],
                "Iteration",
                "Linewidth (Hz)",
                y_scale="log",
            ),
        ],
        "Nonlinearities": [
            OptPltSet(
                ITERATION, [anharmonicity, cross_kerr], "Iteration", "Nonlinearity (Hz)"
            ),
        ],
        "Design Variables": [
            OptPltSet(
                ITERATION, "design_var_lj_qubit_1", "Iteration", "Inductance (nH)"
            ),
            OptPltSet(ITERATION, "design_var_width_qubit_1", "Iteration", "Width (um)"),
        ],
        "Performance vs Design": [
            OptPltSet(
                "design_var_lj_qubit_1", qubit_freq, "Inductance (nH)", "Frequency (Hz)"
            ),
            OptPltSet(
                "design_var_coupl_length_resonator_1_qubit_1",
                cross_kerr,
                "Coupling Length (um)",
                "Cross-Kerr (Hz)",
            ),
        ],
    }

    return {
        "optimization_results": iterations,
        "system_target_params": system_target_params,
        "plot_settings": plot_settings,
    }


class TestPlotProgressIntegration:
    """Integration tests for the plot_progress function with realistic data."""

    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.show")
    def test_realistic_data(
        self, mock_show, mock_subplots, mock_figure, realistic_optimization_data
    ):
        """Test with a more realistic dataset."""
        # Mock the subplot and axes objects
        mock_fig = MagicMock()
        mock_axes = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_axes)

        # Run the function with realistic data
        plot_progress(
            realistic_optimization_data["optimization_results"],
            realistic_optimization_data["system_target_params"],
            realistic_optimization_data["plot_settings"],
            block_plots=False,
        )

        # Check that all 5 figures were created (one for each key in plot_settings)
        assert mock_subplots.call_count == 5, "Wrong number of figures created"

        # Verify titles were set
        assert mock_fig.suptitle.call_count == 5, "Wrong number of titles set"

        # Verify show was called once
        assert mock_show.call_count == 1, "plt.show() should be called once"

    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.show")
    def test_empty_results(
        self, mock_show, mock_subplots, mock_figure, realistic_optimization_data
    ):
        """Test behavior with empty optimization results."""
        # Mock the subplot and axes objects
        mock_fig = MagicMock()
        mock_axes = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_axes)

        # Use empty results
        empty_results = []

        # Run the function
        plot_progress(
            empty_results,
            realistic_optimization_data["system_target_params"],
            realistic_optimization_data["plot_settings"],
            block_plots=False,
        )

        # Should still create the figures but won't have data to plot
        assert (
            mock_subplots.call_count == 5
        ), "Should create figures even with empty results"
        assert mock_show.call_count == 1, "plt.show() should be called once"

    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.show")
    def test_inconsistent_keys(
        self, mock_show, mock_subplots, mock_figure, realistic_optimization_data
    ):
        """Test with inconsistent keys in optimization results."""
        # Mock the subplot and axes objects
        mock_fig = MagicMock()
        mock_axes = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_axes)

        # Create inconsistent data
        inconsistent_results = realistic_optimization_data[
            "optimization_results"
        ].copy()

        # Remove a key from the first result
        qubit_freq = param("qubit_1", "freq")
        del inconsistent_results[0]["system_optimized_params"][qubit_freq]

        # Add a new key to the last result
        new_param = param("new_mode", "freq")
        inconsistent_results[-1]["system_optimized_params"][new_param] = 3.0e9

        # Run the function
        plot_progress(
            inconsistent_results,
            realistic_optimization_data["system_target_params"],
            realistic_optimization_data["plot_settings"],
            block_plots=False,
        )

        # Should still function without errors
        assert mock_show.call_count == 1, "plt.show() should be called once"


if __name__ == "__main__":
    pytest.main(["-v", "test_sim_plot_progress.py"])
