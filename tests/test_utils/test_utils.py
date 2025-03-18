import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from qdesignoptimizer.utils.utils import (
    close_ansys,
    get_junction_position,
    get_middle_point,
    get_normalized_vector,
    get_value_and_unit,
    rotate_point,
    sum_expression,
)


class TestUtils:
    """Tests for utility functions in the utils module."""

    @patch("os.system")
    def test_close_ansys(self, mock_system):
        """Test that close_ansys calls os.system with correct command."""
        close_ansys()
        mock_system.assert_called_once_with("taskkill /f /im ansysedt.exe")

    def test_get_junction_position_no_junction(self):
        """Test get_junction_position raises error when no junction exists."""
        # Setup mocks
        mock_design = MagicMock()
        mock_qcomponent = MagicMock()
        mock_qcomponent.id = "test_component"

        # Mock the junction table with empty result
        mock_junction = MagicMock()
        mock_junction.__len__ = lambda x: 0  # Make len(mock_junction) return 0

        # Setup the return for the query
        mock_junction_table = MagicMock()
        mock_junction_table.loc.__getitem__.return_value = mock_junction

        mock_design.qgeometry.tables = {"junction": mock_junction_table}

        # Test that an assertion error is raised
        with pytest.raises(AssertionError):
            get_junction_position(mock_design, mock_qcomponent)

    def test_get_middle_point(self):
        """Test get_middle_point calculates midpoint correctly."""
        # Test with integers
        assert get_middle_point((0, 0), (10, 20)) == (5.0, 10.0)

        # Test with floats
        assert get_middle_point((1.5, 2.5), (3.5, 7.5)) == (2.5, 5.0)

        # Test with negative numbers
        assert get_middle_point((-10, -20), (10, 20)) == (0.0, 0.0)

    def test_get_normalized_vector(self):
        """Test get_normalized_vector returns unit vector in correct direction."""
        # Test basic case
        x, y = get_normalized_vector((0, 0), (3, 4))
        assert pytest.approx(x) == 0.6
        assert pytest.approx(y) == 0.8

        # Test vector length
        x, y = get_normalized_vector((1, 1), (5, 6))
        assert pytest.approx(np.sqrt(x**2 + y**2)) == 1.0

        # Test horizontal vector
        x, y = get_normalized_vector((0, 0), (5, 0))
        assert pytest.approx(x) == 1.0
        assert pytest.approx(y) == 0.0

        # Test vertical vector
        x, y = get_normalized_vector((0, 0), (0, -5))
        assert pytest.approx(x) == 0.0
        assert pytest.approx(y) == -1.0

    def test_rotate_point(self):
        """Test rotate_point rotates points correctly around a center."""
        # Test 90 degree rotation around origin
        point = np.array([1.0, 0.0])
        center = np.array([0.0, 0.0])
        rotated = rotate_point(point, center, np.pi / 2)

        assert pytest.approx(rotated[0]) == 0.0
        assert pytest.approx(rotated[1]) == 1.0

        # Test 180 degree rotation around origin
        rotated = rotate_point(point, center, np.pi)

        assert pytest.approx(rotated[0]) == -1.0
        assert pytest.approx(rotated[1]) == 0.0

        # Test rotation around non-origin center
        point = np.array([2.0, 0.0])
        center = np.array([1.0, 0.0])
        rotated = rotate_point(point, center, np.pi / 2)

        assert pytest.approx(rotated[0]) == 1.0
        assert pytest.approx(rotated[1]) == 1.0

    def test_get_value_and_unit(self):
        """Test get_value_and_unit parses values and units correctly."""
        # Test with unit
        val, unit = get_value_and_unit("10.5mm")
        assert val == 10.5
        assert unit == "mm"

        # Test with no unit
        val, unit = get_value_and_unit("42")
        assert val == 42.0
        assert unit == ""

        # Test with multi-character unit
        val, unit = get_value_and_unit("5.2GHz")
        assert val == 5.2
        assert unit == "GHz"

        # Test with invalid format
        with pytest.raises(ValueError):
            get_value_and_unit("invalid")

    def test_sum_expression(self):
        """Test sum_expression combines values with units correctly."""
        # Test with mm units
        result = sum_expression(["10mm", "5mm", "2.5mm"])
        assert result == "17.5mm"

        # Test with GHz units
        result = sum_expression(["1.2GHz", "0.8GHz"])
        assert result == "2.0GHz"

        # Test with float values that need rounding
        result = sum_expression(["1.2um", "2.3um"])
        assert result == "3.5um"

        # Test with inconsistent units
        with pytest.raises(AssertionError):
            sum_expression(["10mm", "5um"])
