"""Recording fakes for the Ansys HFSS / Q3D / pyEPR boundary used by DesignAnalysis.

Everything ``DesignAnalysis`` and ``CapacitanceMatrixStudy`` hand to an external
simulator goes through the fakes in this module. Each fake records what it was
asked to do (number of passes, delta_f, variables sent to HFSS, junctions,
cos/fock truncation, Q3D setup, ...) in :class:`SimulationSession.events`, which
the integration tests compare against stored snapshots.

The fakes answer with numbers from :class:`FakePhysics`, a deterministic model
derived from the example's own ``OptTarget`` relations, so the optimizer has to
actually converge for the examples to pass. Since the model follows the same
relations the optimizer assumes, these tests check the plumbing and the
optimizer's loop, not the physics: a target with a wrong exponent or sign would
still converge here. The fakes deliberately implement only
the simulator API currently used; a new call into Ansys/pyEPR raises
``AttributeError`` so the change cannot slip through unnoticed.
"""

from __future__ import annotations

import math
import re
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import scipy.optimize
from qiskit_metal import Dict as AttrDict

from qdesignoptimizer.design_analysis_types import OptTarget
from qdesignoptimizer.sim_capacitance_matrix import (
    ModeDecayIntoChargeLineStudy,
    ResonatorDecayIntoWaveguideStudy,
)
from qdesignoptimizer.utils.names_parameters import (
    CAPACITANCE,
    FREQ,
    KAPPA,
    NONLIN,
    param,
    param_capacitance,
    param_nonlin,
)

# Eigenmode (linear) frequencies are reported slightly above the EPR (dressed)
# frequencies so the snapshots reveal which of the two ends up in the results.
EIGENMODE_FREQ_FACTOR = 1.003
# Initial design frequencies are this factor above target.
FREQ_OFFSET = 1.05


def to_jsonable(obj: Any) -> Any:
    """Convert nested containers with numpy/pandas scalars into plain JSON types."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.DataFrame):
        return to_jsonable(obj.to_dict())
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    return obj


_VALUE_WITH_UNIT = re.compile(
    r"^\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*([a-zA-Z]*)\s*$"
)


def parse_value_with_unit(text) -> tuple | None:
    """Parse "12.5 um" -> (12.5, "um"); numbers pass through. None if unparsable."""
    if isinstance(text, (int, float)) and not isinstance(text, bool):
        return float(text), ""
    match = _VALUE_WITH_UNIT.match(str(text))
    if match is None:
        return None
    return float(match.group(1)), match.group(2)


def _numeric_design_vars(variables: dict[str, str]) -> dict[str, float]:
    return {key: parse_value_with_unit(val)[0] for key, val in variables.items()}


def _target_param_name(target: OptTarget) -> str:
    if target.target_param_type == NONLIN:
        return param_nonlin(*target.involved_modes)
    if target.target_param_type == CAPACITANCE:
        return param_capacitance(*target.involved_modes)
    return param(target.involved_modes[0], target.target_param_type)


class FakePhysics:
    """Deterministic stand-in for the electromagnetic response of the chip.

    Every targeted parameter follows its own ``prop_to`` relation, evaluated with
    the *target* parameter values, and is calibrated such that the initial design
    misses the target by a parameter-specific offset::

        value(v) = target * offset * prop_to(p_target, v) / prop_to(p_target, v0)

    The optimizer evaluates ``prop_to`` with the *current* parameters instead, so
    it needs several iterations to converge, just like against real HFSS.
    Parameters without an ``OptTarget`` get fixed, deterministic values.
    """

    def __init__(
        self,
        opt_targets: list[OptTarget],
        system_target_params: dict,
        initial_design_variables: dict[str, str],
    ):
        self.system_target_params = dict(system_target_params)
        self.targets: dict[str, OptTarget] = {
            _target_param_name(t): t for t in opt_targets
        }
        v0 = _numeric_design_vars(initial_design_variables)
        self._prop_to_initial = {
            name: target.prop_to(self.system_target_params, v0)
            for name, target in self.targets.items()
        }
        # Distinct offsets so that values assigned to the wrong parameter are noticed.
        # All frequencies share one offset so the modes keep their target order
        # (DesignAnalysis assigns eigenmodes by target frequency, see optimize_target).
        self._offsets = {
            name: (
                FREQ_OFFSET
                if name.endswith(FREQ)
                else 1 + 0.12 * math.sin(1.7 * (idx + 1))
            )
            for idx, name in enumerate(sorted(self.targets))
        }

    def value(self, name: str, design_variables: dict[str, str]) -> float | None:
        """Return the modelled value of a targeted parameter, or None if untargeted."""
        if name not in self.targets:
            return None
        v = _numeric_design_vars(design_variables)
        target = self.targets[name]
        ratio = (
            target.prop_to(self.system_target_params, v) / self._prop_to_initial[name]
        )
        return self.system_target_params[name] * self._offsets[name] * ratio

    def freq(self, mode: str, design_variables: dict[str, str]) -> float:
        name = param(mode, FREQ)
        value = self.value(name, design_variables)
        return value if value is not None else self.system_target_params[name]

    def kappa(self, mode: str, design_variables: dict[str, str]) -> float:
        name = param(mode, KAPPA)
        value = self.value(name, design_variables)
        if value is not None:
            return value
        return self.system_target_params.get(name, 1e5)

    def nonlin(
        self, mode_i: str, mode_j: str, idx_i: int, idx_j: int, design_variables
    ) -> float:
        name = param_nonlin(mode_i, mode_j)
        value = self.value(name, design_variables)
        if value is not None:
            return value
        if name in self.system_target_params:
            return self.system_target_params[name]
        if mode_i == mode_j:
            return 150e6 + 1e6 * idx_i
        return 1e6 / (1 + abs(idx_i - idx_j))

    def capacitance(self, name_1: str, name_2: str, design_variables) -> float | None:
        return self.value(param_capacitance(name_1, name_2), design_variables)


class _Recorder:
    """Base class giving fakes access to the shared event log."""

    def __init__(self, session: SimulationSession):
        self._session = session

    def _log(self, event: str, **payload):
        self._session.log(event, **payload)


# --------------------------------------------------------------------------- HFSS
class FakeOEditor(_Recorder):
    """Raw HFSS 3D modeler script interface; every call is recorded verbatim."""

    _RETURN_VALUES = {"GetModelUnits": "mm"}

    def __init__(self, session):
        super().__init__(session)
        self._group_counter = 0

    def __getattr__(self, name):
        if name.startswith("_") or not name[0].isupper():
            raise AttributeError(name)

        def call(*args):
            self._log(f"hfss.oEditor.{name}", args=to_jsonable(list(args)))
            if name == "CreateGroup":
                self._group_counter += 1
                return f"Group{self._group_counter}"
            if name == "Paste":
                return ["pasted_object"]
            return self._RETURN_VALUES.get(name)

        return call


class FakeHfssModeler(_Recorder):
    """pyEPR ``HfssModeler`` as exposed by the qiskit-metal renderer."""

    METAL_OBJECTS = [
        "pad_top_metal",
        "pad_bot_metal",
        "JJ_rect_Lj_metal",
        "trace_metal",
    ]

    def __init__(self, session):
        super().__init__(session)
        self._modeler = FakeOEditor(session)

    def get_objects_in_group(self, group):
        if group == "Perfect E":
            return list(self.METAL_OBJECTS)
        return [f"{group}_object"]

    def unite(self, names, keep_originals=False):
        self._log(
            "hfss.modeler.unite", names=list(names), keep_originals=keep_originals
        )

    def subtract(self, blank_name, tool_names, keep_originals=False):
        self._log(
            "hfss.modeler.subtract",
            blank_name=blank_name,
            tool_names=list(tool_names),
            keep_originals=keep_originals,
        )

    def translate(self, name, vector):
        self._log("hfss.modeler.translate", name=name, vector=list(vector))

    def mesh_length(self, name, objects, **kwargs):
        self._log(
            "hfss.modeler.mesh_length", name=name, objects=list(objects), **kwargs
        )

    def draw_wirebond(self, **kwargs):
        self._log("hfss.modeler.draw_wirebond", **kwargs)
        return kwargs.get("name")


class FakeHfssDesign(_Recorder):
    """Variables set on the active HFSS design (``pinfo.design``)."""

    def __init__(self, session):
        super().__init__(session)
        self.variables: dict[str, str] = {}

    def set_variable(self, name, value):
        self.variables[name] = value


class FakeEigenmodeSetup(_Recorder):
    """HFSS eigenmode setup (``pinfo.setup``); ``analyze`` is the simulation."""

    def __init__(self, session):
        super().__init__(session)
        self.n_modes = None
        self.passes = None
        self.delta_f = None

    def analyze(self):
        session = self._session
        self._log(
            "hfss.eigenmode.analyze",
            n_modes=self.n_modes,
            passes=self.passes,
            delta_f=self.delta_f,
            hfss_variables=dict(session.pinfo.design.variables),
            junctions=deepcopy(session.eig_solver.setup.junctions),
            renderer_options=dict(session.renderer.options),
            setup_name=session.eig_solver.sim.setup.get("name"),
            sweep_variable=session.eig_solver.setup.get("sweep_variable"),
            dissipative=deepcopy(session.pinfo.dissipative),
        )


class FakePinfo(_Recorder):
    def __init__(self, session):
        super().__init__(session)
        self.setup = FakeEigenmodeSetup(session)
        self.design = FakeHfssDesign(session)
        self.dissipative: dict = {}

    def validate_junction_info(self):
        self._log("hfss.pinfo.validate_junction_info")


class FakeHfssRenderer(_Recorder):
    def __init__(self, session):
        super().__init__(session)
        self.options: dict = {}
        self.pinfo = FakePinfo(session)
        self.modeler = FakeHfssModeler(session)

    def start(self):
        self._log("hfss.renderer.start")

    def activate_ansys_design(self, design_name, solution_type):
        self._log(
            "hfss.renderer.activate_ansys_design",
            design_name=design_name,
            solution_type=solution_type,
        )

    def clean_active_design(self):
        self._log("hfss.renderer.clean_active_design")

    def render_design(self, selection, port_list, open_pins):
        self._session.rendered_design_variables = dict(self._session.design.variables)
        self._log(
            "hfss.renderer.render_design",
            selection=list(selection),
            port_list=to_jsonable(port_list),
            open_pins=to_jsonable(open_pins),
            design_variables=dict(self._session.design.variables),
            wire_bonds={
                key: self.options.get(key)
                for key in ("wb_size", "wb_threshold", "wb_offset")
            },
        )


class FakeEPRanalysis(_Recorder):
    """Replacement for ``qiskit_metal.analyses.quantization.EPRanalysis``."""

    def __init__(self, session, design, solver_name):
        super().__init__(session)
        self._log("qiskit_metal.EPRanalysis", solver=solver_name)
        session.design = design
        session.eig_solver = self
        session.renderer = FakeHfssRenderer(session)
        session.pinfo = session.renderer.pinfo
        self.sim = SimpleNamespace(setup=AttrDict(), renderer=session.renderer)
        self.setup = AttrDict()

    def get_frequencies(self):
        session = self._session
        modes = session.simulated_modes_by_frequency()
        variables = session.rendered_design_variables
        freqs_hz = [
            session.physics.freq(mode, variables) * EIGENMODE_FREQ_FACTOR
            for mode in modes
        ]
        kappas_hz = [session.physics.kappa(mode, variables) for mode in modes]
        self._log("hfss.eigenmode.get_frequencies")
        return pd.DataFrame(
            {
                "Freq. (GHz)": [f / 1e9 for f in freqs_hz],
                "Quality Factor": [f / k for f, k in zip(freqs_hz, kappas_hz)],
            }
        )

    def clear_data(self):
        self._log("epr.eig_solver.clear_data")

    def get_stored_energy(self, no_junctions):
        self._log("epr.eig_solver.get_stored_energy", no_junctions=no_junctions)


# -------------------------------------------------------------------------- pyEPR
class FakeDistributedAnalysis(_Recorder):
    def __init__(self, session, pinfo):
        super().__init__(session)
        assert pinfo is session.pinfo, "EPR must run on the renderer's pinfo"
        self.data_filename = "fake_epr_data.npz"
        self._mode = None

    def do_EPR_analysis(self):
        self._log(
            "epr.do_EPR_analysis",
            junctions=deepcopy(self._session.eig_solver.setup.junctions),
        )

    def set_mode(self, mode):
        self._mode = mode

    def get_Qsurface(self, mode, variation, name):
        self._log("epr.get_Qsurface", mode=mode, variation=variation, name=name)
        # depends on the interface properties handed to pyEPR via pinfo.dissipative
        props = self._session.pinfo.dissipative["dielectric_surfaces"][name]
        loss = props["eps_r"] * props["th"] * props["tan_delta_surf"]
        return pd.Series([(1 + mode) * (1 + len(name)) / loss])

    def get_Qdielectric(self, dielectric, mode, variation):
        self._log(
            "epr.get_Qdielectric", dielectric=dielectric, mode=mode, variation=variation
        )
        return pd.Series([2e6 * (1 + mode)])

    def calc_p_junction_single(self, mode, variation):
        self._log("epr.calc_p_junction_single", mode=mode, variation=variation)
        return {"j0": 0.9 / (1 + mode)}


class FakeQuantumAnalysis(_Recorder):
    def __init__(self, session, data_filename):
        super().__init__(session)
        self._log("epr.QuantumAnalysis", data_filename=data_filename)

    def analyze_all_variations(self, cos_trunc, fock_trunc):
        self._log(
            "epr.analyze_all_variations", cos_trunc=cos_trunc, fock_trunc=fock_trunc
        )

    def plot_hamiltonian_results(self):
        pass

    def get_frequencies(self, numeric=True):
        session = self._session
        variables = session.rendered_design_variables
        # numerical diagonalization leaves tiny imaginary parts
        freqs_mhz = [
            complex(session.physics.freq(mode, variables) / 1e6, 1e-9)
            for mode in session.simulated_modes_by_frequency()
        ]
        return pd.DataFrame({"0": freqs_mhz})

    def get_chis(self, numeric=True):
        session = self._session
        variables = session.rendered_design_variables
        modes = session.simulated_modes_by_frequency()
        chis_mhz = [
            [
                complex(
                    session.physics.nonlin(mode_i, mode_j, i, j, variables) / 1e6, 1e-12
                )
                for j, mode_j in enumerate(modes)
            ]
            for i, mode_i in enumerate(modes)
        ]
        return pd.DataFrame(chis_mhz, index=_variation_mode_index(len(modes)))

    def get_participations(self):
        session = self._session
        n_modes = len(session.simulated_modes_by_frequency())
        n_junctions = len(session.eig_solver.setup.junctions)
        # asymmetric in (mode, junction) so that swapped indices are noticed
        return pd.DataFrame(
            [
                [0.1 * (i + 1) + 0.013 * (j + 1) ** 2 for j in range(n_junctions)]
                for i in range(n_modes)
            ],
            index=_variation_mode_index(n_modes),
        )


def _variation_mode_index(n_modes: int) -> pd.MultiIndex:
    """pyEPR indexes chis and participations by (variation, mode)."""
    return pd.MultiIndex.from_product(
        [["0"], range(n_modes)], names=["variation", None]
    )


# ---------------------------------------------------------------------------- Q3D
class FakeLOManalysis(_Recorder):
    """Replacement for ``qiskit_metal.analyses.quantization.LOManalysis``."""

    def __init__(self, session, design, solver_name):
        super().__init__(session)
        self._design = design
        self._log("qiskit_metal.LOManalysis", solver=solver_name)
        self.sim = SimpleNamespace(
            setup=AttrDict(),
            renderer=SimpleNamespace(options={}),
            run=self._run,
            capacitance_matrix=None,
        )

    def _run(self, components, open_terminations):
        session = self._session
        self._log(
            "q3d.run",
            components=list(components),
            open_terminations=to_jsonable(open_terminations),
            max_passes=self.sim.setup.max_passes,
            percent_error=self.sim.setup.percent_error,
            renderer_options=dict(self.sim.renderer.options),
            design_variables=dict(self._design.variables),
        )
        study = session.find_capacitance_study(components, open_terminations)
        self.sim.capacitance_matrix = session.capacitance_matrix(
            study, dict(self._design.variables)
        )


# ------------------------------------------------------------------------ Session
class SimulationSession:
    """Shared state for all fakes during one example run, including the event log."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Forget all recorded calls and the physics model (start a fresh chip)."""
        self.events: list[list] = []
        self.analysis = None  # the DesignAnalysis currently driving the simulations
        self.physics: FakePhysics | None = None
        self.design = None
        self.eig_solver = None
        self.renderer = None
        self.pinfo = None
        self.rendered_design_variables: dict[str, str] = {}

    def log(self, event: str, **payload):
        self.events.append([event, to_jsonable(payload)])

    # Factories handed to monkeypatch -------------------------------------------
    def epranalysis_factory(self, design, solver_name):
        return FakeEPRanalysis(self, design, solver_name)

    def lomanalysis_factory(self, design, solver_name):
        return FakeLOManalysis(self, design, solver_name)

    def epr_module(self):
        return SimpleNamespace(
            DistributedAnalysis=lambda pinfo: FakeDistributedAnalysis(self, pinfo),
            QuantumAnalysis=lambda filename: FakeQuantumAnalysis(self, filename),
        )

    # Context ---------------------------------------------------------------------
    def register_analysis(self, analysis):
        """Called after each DesignAnalysis is constructed."""
        self.analysis = analysis
        if self.physics is None:
            self.physics = FakePhysics(
                analysis.opt_targets,
                analysis.system_target_params,
                dict(analysis.design.variables),
            )

    def simulated_modes_by_frequency(self) -> list[str]:
        """Simulated modes in the order HFSS returns them: by simulated frequency."""
        variables = self.rendered_design_variables
        return sorted(
            self.analysis.mini_study.modes,
            key=lambda mode: self.physics.freq(mode, variables),
        )

    def find_capacitance_study(self, components, open_terminations):
        for study in self.analysis.mini_study.capacitance_matrix_studies:
            if list(study.qiskit_component_names) == list(components) and list(
                study.open_pins
            ) == list(open_terminations):
                return study
        raise AssertionError(
            f"Q3D run with components={components} does not match any capacitance study"
        )

    def capacitance_matrix(self, study, design_variables) -> pd.DataFrame:
        """Build a symmetric capacitance matrix (fF) consistent with FakePhysics."""
        names: list[str] = []
        entries: dict[tuple, float] = {}

        def add_name(name):
            if name is not None and name not in names:
                names.append(name)

        for target in self.physics.targets.values():
            if target.target_param_type == CAPACITANCE:
                add_name(target.involved_modes[0])
                add_name(target.involved_modes[1])
                entries[tuple(target.involved_modes)] = self.physics.capacitance(
                    *target.involved_modes, design_variables
                )

        if isinstance(study, ResonatorDecayIntoWaveguideStudy):
            add_name(study.resonator_name)
            add_name(study.waveguide_name)
            kappa = self.physics.kappa(study.mode, design_variables)
            omega = study.mode_freq_GHz * 2 * np.pi
            prefactor = study.impedance_ohm**2 * omega**3 / np.pi / (2 * np.pi) * 1e-3
            if study.resonator_type == "lambda_4":
                prefactor *= 2
            entries[(study.resonator_name, study.waveguide_name)] = -np.sqrt(
                kappa / prefactor
            )

        if isinstance(study, ModeDecayIntoChargeLineStudy):
            islands = study.mode_capacitance_name
            islands = [islands] if isinstance(islands, str) else list(islands)
            for name in islands:
                add_name(name)
            add_name(study.charge_line_capacitance_name)
            add_name(study.ground_plane_capacitance_name)
            entries.update(self._charge_line_entries(study, islands, design_variables))

        add_name("ground_main_plane")
        size = len(names)
        matrix = pd.DataFrame(np.zeros((size, size)), index=names, columns=names)
        for i, name_i in enumerate(names):
            matrix.loc[name_i, name_i] = 80.0 + 7.0 * i
            for j in range(i + 1, size):
                default = -(0.3 + 0.1 * (i + j))
                matrix.loc[name_i, names[j]] = default
                matrix.loc[names[j], name_i] = default
        for (name_i, name_j), value in entries.items():
            matrix.loc[name_i, name_j] = value
            matrix.loc[name_j, name_i] = value
        return matrix

    def _charge_line_entries(self, study, islands, design_variables):
        """Solve for island-to-charge-line capacitances that yield the modelled T1."""
        t1_target = self.physics.value(
            param(study.mode, study.get_decay_parameter_type()), design_variables
        )
        line = study.charge_line_capacitance_name
        ground = study.ground_plane_capacitance_name
        if len(islands) == 1:
            island = islands[0]

            def t1(scale):
                return _reference_t1(
                    study.mode_freq_GHz, 80.0, scale, study.charge_line_impedance_Ohm
                )

            scale = _solve_log(lambda s: t1(s) - t1_target)
            return {(island, island): 80.0, (island, line): -scale}

        island_a, island_b = islands
        base = {"ca0": 60.0, "cb0": 50.0, "ca1": 0.5, "cb1": 0.2, "cj": 5.0}

        def t1(scale):
            # series combination of the island-to-ground and island-to-line networks
            cap_a = base["ca0"] + base["ca1"] * scale
            cap_b = base["cb0"] + base["cb1"] * scale
            cap_total = cap_a * cap_b / (cap_a + cap_b) + base["cj"]
            cap_coupling = abs(
                (cap_a * base["cb1"] - cap_b * base["ca1"]) * scale / (cap_a + cap_b)
            )
            return _reference_t1(
                study.mode_freq_GHz,
                cap_total,
                cap_coupling,
                study.charge_line_impedance_Ohm,
            )

        scale = _solve_log(lambda s: t1(s) - t1_target)
        return {
            (island_a, ground): -base["ca0"],
            (island_b, ground): -base["cb0"],
            (island_a, line): -base["ca1"] * scale,
            (island_b, line): -base["cb1"] * scale,
            (island_a, island_b): -base["cj"],
        }


def _reference_t1(freq_ghz, cap_mode_ff, cap_coupling_ff, impedance_ohm) -> float:
    """T1 = 1/gamma with gamma = omega^2 * C * (Cc/C)^2 * Z in SI units.

    Written independently of qdesignoptimizer.estimation so that a regression in
    the library's formula is not cancelled out by the fake.
    """
    omega = 2 * np.pi * freq_ghz * 1e9
    cap_mode = cap_mode_ff * 1e-15
    beta = cap_coupling_ff / cap_mode_ff
    return 1 / (omega**2 * cap_mode * beta**2 * impedance_ohm)


def _solve_log(func) -> float:
    """Find the positive root of a monotonic function by bracketing in log space."""
    return float(
        np.exp(scipy.optimize.brentq(lambda log_s: func(np.exp(log_s)), -20, 20))
    )
