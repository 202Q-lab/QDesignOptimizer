import pytest
from qiskit_metal.renderers.renderer_ansys.ansys_renderer import QAnsysRenderer


def test_keep_originals_property_exists():
    renderer = QAnsysRenderer()
    assert hasattr(renderer, "keep_originals")
