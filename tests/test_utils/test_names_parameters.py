import pytest

from qdesignoptimizer.utils.names_parameters import (
    FREQ,
    KAPPA,
    QUBIT,
    RESONATOR,
    get_mode_from_param,
    get_modes_from_param_nonlin,
    mode,
    param,
    param_capacitance,
    param_nonlin,
)


class TestNamesParameters:
    def test_mode_creation(self):
        # Test basic mode creation
        assert mode(QUBIT) == "qubit"
        assert mode(QUBIT, 1) == "qubit_1"
        assert mode(RESONATOR, "test") == "resonator_test"

    def test_mode_creation_validation(self):
        # Test validation constraints
        with pytest.raises(AssertionError):
            mode("invalid_mode_type_with_underscore")

        with pytest.raises(AssertionError):
            mode(QUBIT, "id_with_underscore")

        with pytest.raises(AssertionError):
            mode("mode_with_to", "identifier")

    def test_param_creation(self):
        # Test parameter creation
        qubit_mode = mode(QUBIT, 1)
        resonator_mode = mode(RESONATOR, 2)

        assert param(qubit_mode, FREQ) == "qubit_1_freq"
        assert param(resonator_mode, KAPPA) == "resonator_2_kappa"

    def test_param_validation(self):
        # Test parameter type validation
        qubit_mode = mode(QUBIT, 1)

        with pytest.raises(AssertionError):
            param(qubit_mode, "invalid_param_type")

    def test_param_nonlin(self):
        # Test nonlinear parameter creation
        qubit1 = mode(QUBIT, 1)
        qubit2 = mode(QUBIT, 2)

        # Test that parameters are sorted alphabetically
        assert param_nonlin(qubit1, qubit2) == "qubit_1_to_qubit_2_nonlin"
        assert param_nonlin(qubit2, qubit1) == "qubit_1_to_qubit_2_nonlin"

        # Test self-Kerr/anharmonicity (same mode twice)
        assert param_nonlin(qubit1, qubit1) == "qubit_1_to_qubit_1_nonlin"

    def test_param_capacitance(self):
        # Test capacitance parameter creation
        cap1 = "island_a"
        cap2 = "island_b"

        # Test that parameters are sorted alphabetically
        assert param_capacitance(cap1, cap2) == "island_a_to_island_b_capacitance"
        assert param_capacitance(cap2, cap1) == "island_a_to_island_b_capacitance"

    def test_get_mode_from_param(self):
        # Test extracting mode from parameter
        assert get_mode_from_param("qubit_1_freq") == "qubit_1"
        assert get_mode_from_param("resonator_test_kappa") == "resonator_test"

    def test_get_modes_from_param_nonlin(self):
        # Test extracting modes from nonlinear parameter
        modes = get_modes_from_param_nonlin("qubit_1_to_qubit_2_nonlin")
        assert len(modes) == 2
        assert "qubit_1" in modes
        assert "qubit_2" in modes

        # Test with self-Kerr
        modes = get_modes_from_param_nonlin("qubit_1_to_qubit_1_nonlin")
        assert len(modes) == 2
        assert modes[0] == "qubit_1"
        assert modes[1] == "qubit_1"

    def test_get_modes_from_param_nonlin_validation(self):
        # Test validation
        with pytest.raises(AssertionError):
            get_modes_from_param_nonlin("qubit_1_to_qubit_2_invalid")
