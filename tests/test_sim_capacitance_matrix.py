from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from qiskit_metal import Dict, config

from qdesignoptimizer.sim_capacitance_matrix import (
    CapacitanceMatrixStudy,
    ModeDecayIntoChargeLineStudy,
)


@pytest.fixture
def mock_design():
    """Create a mock QDesign object."""
    design = MagicMock()
    design.name = "mock_design"
    return design


@pytest.fixture
def sample_capacitance_matrix():
    """Create a sample capacitance matrix."""
    # Create a DataFrame with representative values for island capacitance tests
    df = pd.DataFrame(
        {
            "island_1": [10.0, -5.0, -2.0, -3.0],
            "island_2": [-5.0, 12.0, -3.0, -4.0],
            "charge_line": [-2.0, -3.0, 8.0, -3.0],
            "ground": [-3.0, -4.0, -3.0, 10.0],
        },
        index=["island_1", "island_2", "charge_line", "ground"],
    )
    return df


@pytest.fixture
def mock_lom_analysis(sample_capacitance_matrix):
    """Create a mock LOManalysis object with necessary attributes."""
    mock = MagicMock()
    # Create nested attributes
    mock.name = "mock_lom"
    mock.sim = MagicMock()
    mock.sim.name = "mock_sim"
    mock.sim.setup = Dict()
    mock.sim.renderer = MagicMock()
    mock.sim.renderer.name = "mock_renderer"
    mock.sim.renderer.options = {}

    # Set renderer_initialized to True
    mock.sim.renderer_initialized = True

    # Mock the get_capacitance_matrix method to return matrix and units
    mock.sim.renderer.get_capacitance_matrix.return_value = (
        sample_capacitance_matrix,
        "fF",
    )

    # Mock the get_capacitance_all_passes method
    mock.sim.renderer.get_capacitance_all_passes.return_value = (
        [sample_capacitance_matrix],
        "fF",
    )

    # Mock the get_convergence method
    mock.sim.renderer.get_convergence.return_value = True

    # Configure run method to set the capacitance_matrix and other properties
    def run_side_effect(*args, **kwargs):
        mock.sim.capacitance_matrix = sample_capacitance_matrix
        mock.sim.units = "fF"
        mock.sim.capacitance_all_passes = [sample_capacitance_matrix]
        mock.sim.is_converged = True

    mock.sim.run.side_effect = run_side_effect

    return mock


class TestCapacitanceMatrixStudy:
    """Tests for the CapacitanceMatrixStudy class."""

    def test_initialization(self):
        """Test that CapacitanceMatrixStudy initializes with correct parameters."""
        try:
            qiskit_component_names = ["comp1", "comp2"]
            freq_GHz = 5.0
            open_pins = [("comp1", "pin1")]

            study = CapacitanceMatrixStudy(
                qiskit_component_names=qiskit_component_names,
                mode_freq_GHz=freq_GHz,
                open_pins=open_pins,
                x_buffer_width_mm=2.5,
                y_buffer_width_mm=3.0,
                percent_error=0.3,
                nbr_passes=12,
            )

            assert study.qiskit_component_names == qiskit_component_names
            assert study.mode_freq_GHz == freq_GHz
            assert study.open_pins == open_pins
            assert study.x_buffer_width_mm == 2.5
            assert study.y_buffer_width_mm == 3.0
            assert study.percent_error == 0.3
            assert study.nbr_passes == 12
            assert study.capacitance_matrix_fF is None
            assert study.mode_capacitances_matrix_fF is None
        except Exception as e:
            pytest.fail(f"Initialization test failed with exception: {e}")

    def test_set_render_qiskit_metal(self):
        """Test that render_qiskit_metal function can be set correctly."""
        try:
            study = CapacitanceMatrixStudy(
                qiskit_component_names=["comp1"], mode_freq_GHz=5.0
            )

            render_func = MagicMock()
            study.set_render_qiskit_metal(render_func)

            assert study.render_qiskit_metal == render_func
        except Exception as e:
            pytest.fail(f"Set render test failed with exception: {e}")

    def test_simulate_capacitance_matrix(
        self, mock_design, sample_capacitance_matrix, mock_lom_analysis
    ):
        """Test simulate_capacitance_matrix method with default parameters."""
        # Patch where the module is actually imported from, not where it's defined
        with patch(
            "qdesignoptimizer.sim_capacitance_matrix.LOManalysis",
            return_value=mock_lom_analysis,
        ) as mock_lom_class:
            try:
                # Create study
                study = CapacitanceMatrixStudy(
                    qiskit_component_names=["comp1", "comp2"],
                    mode_freq_GHz=5.0,
                    open_pins=[("comp1", "pin1")],
                    percent_error=0.3,
                    nbr_passes=12,
                )

                # Simulate - this should trigger our mocked run method
                result = study.simulate_capacitance_matrix(mock_design)

                # Verify results
                assert result is sample_capacitance_matrix
                assert study.capacitance_matrix_fF is sample_capacitance_matrix

                # Verify LOManalysis setup
                mock_lom_class.assert_called_once_with(mock_design, "q3d")
                assert mock_lom_analysis.sim.setup.max_passes == 12
                assert mock_lom_analysis.sim.setup.percent_error == 0.3
                assert "x_buffer_width_mm" in mock_lom_analysis.sim.renderer.options
                assert "y_buffer_width_mm" in mock_lom_analysis.sim.renderer.options

                # Verify run method was called with correct args
                mock_lom_analysis.sim.run.assert_called_once()
            except Exception as e:
                pytest.fail(
                    f"Simulate capacitance matrix test failed with exception: {e}"
                )

    def test_simulate_capacitance_matrix_with_render(
        self, mock_design, sample_capacitance_matrix, mock_lom_analysis
    ):
        """Test simulate_capacitance_matrix method with custom render function."""
        with patch(
            "qdesignoptimizer.sim_capacitance_matrix.LOManalysis",
            return_value=mock_lom_analysis,
        ) as mock_lom_class:
            try:
                # Create render function mock
                render_func = MagicMock()

                # Create study with render function
                study = CapacitanceMatrixStudy(
                    qiskit_component_names=["comp1"],
                    mode_freq_GHz=5.0,
                    render_qiskit_metal=render_func,
                    render_qiskit_metal_kwargs={"param1": "value1"},
                )

                # Simulate - this should trigger our mocked run method
                result = study.simulate_capacitance_matrix(mock_design)

                # Verify render function was called with correct parameters
                render_func.assert_called_once_with(mock_design, param1="value1")
                assert result is sample_capacitance_matrix
            except Exception as e:
                pytest.fail(f"Simulate with render test failed with exception: {e}")


class TestModeDecayIntoChargeLineStudy:
    """Tests for the ModeDecayIntoChargeLineStudy class."""

    def test_initialization(self):
        """Test initialization of ModeDecayIntoChargeLineStudy with required parameters."""
        try:
            study = ModeDecayIntoChargeLineStudy(
                mode="qubit1",
                mode_freq_GHz=5.0,
                mode_capacitance_name="island_1",
                charge_line_capacitance_name="charge_line",
                charge_line_impedance_Ohm=50.0,
                qiskit_component_names=["comp1"],
                open_pins=[],
                ground_plane_capacitance_name="ground",
            )

            assert study.mode == "qubit1"
            assert study.mode_freq_GHz == 5.0
            assert study.mode_capacitance_name == "island_1"
            assert study.charge_line_capacitance_name == "charge_line"
            assert study.charge_line_impedance_Ohm == 50.0
            assert study.ground_plane_capacitance_name == "ground"
            assert study.freq_GHz == 5.0
        except Exception as e:
            pytest.fail(f"Initialization test failed with exception: {e}")

    @patch(
        "qdesignoptimizer.sim_capacitance_matrix.calculate_t1_limit_floating_lumped_mode_decay_into_chargeline"
    )
    def test_get_t1_limit_floating_design(self, mock_calc, sample_capacitance_matrix):
        """Test T1 limit calculation for floating design with two islands."""
        try:
            mock_calc.return_value = 1e-4  # 100 microseconds

            study = ModeDecayIntoChargeLineStudy(
                mode="qubit1",
                mode_freq_GHz=5.0,
                mode_capacitance_name=[
                    "island_1",
                    "island_2",
                ],  # Floating design with two islands
                charge_line_capacitance_name="charge_line",
                charge_line_impedance_Ohm=50.0,
                qiskit_component_names=["comp1"],
                ground_plane_capacitance_name="ground",
            )

            # Set capacitance matrix directly
            study.capacitance_matrix_fF = sample_capacitance_matrix

            t1_limit = study.get_t1_limit_due_to_decay_into_charge_line()

            # Verify calculate function was called
            mock_calc.assert_called_once()
            # Extract the call arguments
            call_args = mock_calc.call_args[1]

            # Check parameters - using absolute values of the matrix entries
            assert call_args["mode_freq_GHz"] == 5.0
            assert call_args["charge_line_impedance"] == 50.0
            assert call_args["cap_island_a_island_b_fF"] == abs(
                sample_capacitance_matrix.loc["island_1", "island_2"]
            )
            assert call_args["cap_island_a_ground_fF"] == abs(
                sample_capacitance_matrix.loc["island_1", "ground"]
            )
            assert call_args["cap_island_a_line_fF"] == abs(
                sample_capacitance_matrix.loc["island_1", "charge_line"]
            )
            assert call_args["cap_island_b_ground_fF"] == abs(
                sample_capacitance_matrix.loc["island_2", "ground"]
            )
            assert call_args["cap_island_b_line_fF"] == abs(
                sample_capacitance_matrix.loc["island_2", "charge_line"]
            )

            assert t1_limit == 1e-4
        except Exception as e:
            pytest.fail(f"Floating design test failed with exception: {e}")

    @patch(
        "qdesignoptimizer.sim_capacitance_matrix.calculate_t1_limit_grounded_lumped_mode_decay_into_chargeline"
    )
    def test_get_t1_limit_grounded_design(self, mock_calc, sample_capacitance_matrix):
        """Test T1 limit calculation for grounded design with single island."""
        try:
            mock_calc.return_value = 2e-4  # 200 microseconds

            study = ModeDecayIntoChargeLineStudy(
                mode="qubit1",
                mode_freq_GHz=5.0,
                mode_capacitance_name="island_1",  # Grounded design with single island
                charge_line_capacitance_name="charge_line",
                charge_line_impedance_Ohm=50.0,
                qiskit_component_names=["comp1"],
                ground_plane_capacitance_name="ground",
            )

            # Set capacitance matrix directly
            study.capacitance_matrix_fF = sample_capacitance_matrix

            t1_limit = study.get_t1_limit_due_to_decay_into_charge_line()

            # Verify calculate function was called
            mock_calc.assert_called_once()
            call_args = mock_calc.call_args[1]

            # Check parameters
            assert call_args["mode_capacitance_fF"] == abs(
                sample_capacitance_matrix.loc["island_1", "island_1"]
            )
            assert call_args["mode_capacitance_to_charge_line_fF"] == abs(
                sample_capacitance_matrix.loc["island_1", "charge_line"]
            )
            assert call_args["mode_freq_GHz"] == 5.0
            assert call_args["charge_line_impedance"] == 50.0

            assert t1_limit == 2e-4
        except Exception as e:
            pytest.fail(f"Grounded design test failed with exception: {e}")

    def test_get_t1_limit_no_capacitance_matrix(self):
        """Test error handling when trying to calculate T1 limit without setting capacitance matrix."""
        study = ModeDecayIntoChargeLineStudy(
            mode="qubit1",
            mode_freq_GHz=5.0,
            mode_capacitance_name="island_1",
            charge_line_capacitance_name="charge_line",
            charge_line_impedance_Ohm=50.0,
            qiskit_component_names=["comp1"],
        )

        with pytest.raises(AssertionError, match="capacitance_matrix_fF is not set"):
            study.get_t1_limit_due_to_decay_into_charge_line()

    def test_get_t1_limit_invalid_capacitance_name(self, sample_capacitance_matrix):
        """Test error handling with invalid capacitance name format."""
        study = ModeDecayIntoChargeLineStudy(
            mode="qubit1",
            mode_freq_GHz=5.0,
            mode_capacitance_name=[
                1,
                2,
                3,
            ],  # Invalid format - should be string or list of 2 strings
            charge_line_capacitance_name="charge_line",
            charge_line_impedance_Ohm=50.0,
            qiskit_component_names=["comp1"],
        )

        study.capacitance_matrix_fF = sample_capacitance_matrix

        with pytest.raises(
            NotImplementedError,
            match="The mode capacitance name must be a string or a list of string matching the name of the island",
        ):
            study.get_t1_limit_due_to_decay_into_charge_line()
