import unittest

from qiskit_metal.renderers.renderer_ansys.ansys_renderer import QAnsysRenderer

class TestQAnsysRenderer(unittest.TestCase):
    def test_property_exists(self):
        renderer = QAnsysRenderer()
        self.assertTrue(hasattr(renderer, 'keep_originals'))
