"""A study switching back to a design must solve that design's setup."""

from types import SimpleNamespace

from qdesignoptimizer.design_analysis import activate_eigenmode_design


class _Pinfo:
    """Like pyEPR's ProjectInfo: the setup is connected apart from the design."""

    def __init__(self, setups):
        self._setups = setups  # design name -> {setup name: setup}
        self.design = None
        self.setup = None
        self.setup_name = None

    def connect_setup(self):
        setups = self._setups[self.design.name]
        if self.setup_name is None:
            self.setup_name = next(iter(setups))
        if self.setup_name not in setups:
            raise ValueError(f"Setup {self.setup_name!r} not found in design")
        self.setup = setups[self.setup_name]


class _Renderer:
    """Like qiskit-metal: an existing design is reconnected, its setup not."""

    def __init__(self, setups):
        self.pinfo = _Pinfo(setups)

    def activate_ansys_design(self, design_name, solution_type):
        self.pinfo.design = SimpleNamespace(name=design_name)
        if self.pinfo.setup is None:
            self.pinfo.connect_setup()


def test_switching_back_to_a_design_solves_its_own_setup():
    renderer = _Renderer(
        {
            "Q1_readout": {"Setup": "Q1 setup"},
            "C_coupler": {"Setup": "C setup"},
        }
    )
    activate_eigenmode_design(renderer, "Q1_readout")
    activate_eigenmode_design(renderer, "C_coupler")

    pinfo = activate_eigenmode_design(renderer, "Q1_readout")

    assert pinfo.setup == "Q1 setup"


def test_a_design_whose_setup_has_another_name_is_connected():
    renderer = _Renderer(
        {
            "Q1_readout": {"Resonator_setup": "Q1 setup"},
            "C_coupler": {"Setup": "C setup"},
        }
    )
    activate_eigenmode_design(renderer, "Q1_readout")

    pinfo = activate_eigenmode_design(renderer, "C_coupler")

    assert pinfo.setup == "C setup"
