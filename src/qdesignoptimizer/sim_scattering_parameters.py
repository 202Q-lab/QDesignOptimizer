"""Study classes for capacitance matrix based simulations."""

from typing import Callable, List, Optional

import numpy as np
from pyEPR.ansys import HfssSetup
from qiskit_metal.analyses.simulation.scattering_impedance import ScatteringImpedanceSim
from qiskit_metal.designs.design_base import QDesign
from resonator_tools import circuit


class ScatteringParametersStudy:
    """Base class for scattering parameters studies.
    This class is used to define the basic structure and methods for scattering parameters studies.
    It is not intended to be instantiated directly.
    Args :
        qiskit_component_names (list): Names of Qiskit Metal components to include in the
            capacitance simulation.
        open_pins (list, optional): Pin connections to leave open (not grounded or connected),
            specified as tuples of (component_name, pin_name). Defaults to an empty list.
        x_buffer_width_mm (float, optional): Width of simulation buffer space in x-direction
            in millimeters. Defaults to 2mm.
        y_buffer_width_mm (float, optional): Width of simulation buffer space in y-direction
            in millimeters. Defaults to 2mm.
        render_qiskit_metal (Callable, optional): Function for rendering the design before
            simulation. If None, the function from DesignAnalysisState will be used when this
            study is part of a DesignAnalysis optimization. Takes the form
            ``render_qiskit_metal(design, **kwargs)``.
        render_qiskit_metal_kwargs (dict, optional): Keyword arguments for the render_qiskit_metal
            function. Defaults to an empty dict.
        counts (int, optional): Number of frequency points in the sweep. Defaults to 25000.
            Was previously accepted but ignored (add_sweep hardcoded 25000). Lowering it is
            the main lever on scratch-disk usage: every adaptive pass stores field solutions
            for the swept points, and a 20-pass fine-meshed run at 25000 points grew
            C:\\Temp to 35GB and filled the system disk.
        bandwidth (float, optional): Total bandwidth of the sweep in GHz. Default to 1GHz.
        passes (int, optional): Number of passes for the mesh optimization of the Driven Modal setup. Default to 35.
        hfss_wire_bond_size (int, optional): Size parameter for wire bonds. The span is
            hfss_wire_bond_size * (trace_width + 2 * trace_gap) -- the SUBTRACT path's
            width, not the metal trace's. Defaults to 2, matching MiniStudy; see
            MiniStudy for the two geometric constraints that fix it.
        hfss_wire_bond_offset (str, optional): Offset parameter for wire bonds (with unit).
            Defaults to "0um".
        hfss_wire_bond_threshold (str, optional): Minimum path-segment length for a wire
            bond to be placed (with unit). Defaults to "300um".
    """

    def __init__(
        self,
        qiskit_component_names: List[str],
        open_pins: List[str],
        port_list: List[str],
        x_buffer_width_mm: float = 0.5,
        y_buffer_width_mm: float = 0.5,
        render_qiskit_metal: Optional[Callable] = None,
        render_qiskit_metal_kwargs: Optional[dict] = None,
        counts: Optional[int] = 25000,
        component_of_interest: str = None,
        bandwidth: float = 1,
        passes: int = 10,
        hfss_wire_bond_size: int = 2,
        hfss_wire_bond_offset: str = "0um",
        hfss_wire_bond_threshold: str = "300um",
    ):

        self.qiskit_component_names = qiskit_component_names
        self.open_pins = open_pins
        self.x_buffer_width_mm = x_buffer_width_mm
        self.y_buffer_width_mm = y_buffer_width_mm
        self.render_qiskit_metal = render_qiskit_metal
        self.render_qiskit_metal_kwargs = render_qiskit_metal_kwargs or {}
        self.counts = counts
        self.port_list = port_list
        self.component_of_interest = component_of_interest
        self.bandwidth = bandwidth
        self.passes = passes
        # Mirrors MiniStudy's wire-bond options. Without these the DrivenModal
        # render falls through to the qiskit-metal renderer defaults
        # (wb_size=5, wb_threshold="400um"), which draw bonds spanning
        # wb_size * cpw_gap_width. On a tightly folded meander that exceeds the
        # arm-to-arm pitch, the bonds intersect each other and HFSS refuses to
        # solve ("Parts g_wb_N and g_wb_M intersect" -> com_error from
        # Analyze()). The eigenmode path already propagates these, see
        # design_analysis.py's setup_eigenmode.
        self.hfss_wire_bond_size = hfss_wire_bond_size
        self.hfss_wire_bond_offset = hfss_wire_bond_offset
        self.hfss_wire_bond_threshold = hfss_wire_bond_threshold

    def set_render_qiskit_metal(self, render_qiskit_metal: Callable) -> None:
        """Set the rendering function to use before capacitance simulation.
        This method allows updating the rendering function after initialization,
        particularly useful when a DesignAnalysisState is used for within a DesignAnalysis context.
        Args:
            render_qiskit_metal (Callable): Function for rendering the design before simulation.
                Takes the form ``render_qiskit_metal(design, **kwargs)``.
        """
        self.render_qiskit_metal = render_qiskit_metal

    def simulate_scattering_parameters(
        self,
        design: QDesign,
        eigenmode_setup: HfssSetup = None,
        hfss_design_name: str = "Scattering_Study",
        center_frequency: float = 5,
        fine_mesh_names: List[str] = None,
        max_mesh_length: str = "5um",
    ):
        """
        Simulate scattering parameters using HFSS.
        Runs the HFSS simulation for the specified design and returns the results.
        Args:
            design (QDesign): The Qiskit Metal design object to simulate.
            eigenmode_setup (HfssSetup) : eigenmode solution setup pointer to link mesh to
            hfss_design_name (str): Name of the HFSS design. Defaults to "Scattering_Study". If None, an entirely new mesh is generated
            center_frequency (float): Center frequency for the simulation in GHz. Defaults to 5 GHz.
            fine_mesh_names (List[str], optional): Names of rendered geometry objects to apply a
                manual fine-mesh operation to (e.g. coupling capacitor polys), independent of any
                eigenmode_setup link. Applied directly to this DrivenModal design after rendering
                and before the sweep analysis -- unlike DesignAnalysis.run_eigenmodes()'s own
                MeshingMap refinement (which only ever applies to the eigenmode design), this lets
                a scattering-only run (no eigenmode step at all) still get fine mesh on named
                coupling geometry. Defaults to None (no manual mesh operation added here).
            max_mesh_length (str): Max element size (with unit) for fine_mesh_names, mirrors
                MiniStudy.max_mesh_length_lines_to_ports's own default. Defaults to "5um".
        """

        scatteringanalysis = ScatteringImpedanceSim(design, "hfss")

        scattering_analysis_renderer = scatteringanalysis.renderer

        scattering_analysis_renderer.activate_ansys_design(
            hfss_design_name, "drivenmodal"
        )
        # Set design variables through the renderer's existing connection
        # (pinfo.design, a pyEPR HfssDesign) instead of opening a second pyaedt
        # Hfss(...) session. The second session can fail with a gRPC "Failed to
        # connect to Desktop Session" error while the renderer's connection is
        # open, leaving hfss.variable_manager as None.
        for key, value in design.variables.items():
            scattering_analysis_renderer.pinfo.design.set_variable(key, value)
        scattering_analysis_renderer.add_drivenmodal_setup(
            name="Setup_QDO",
            max_delta_s=0.001,
            min_passes=self.passes,
            max_passes=self.passes,
            freq_ghz=center_frequency,
            min_converged=5,
            pct_refinement=20,
            basis_order=-1,
        )
        scattering_analysis_renderer.options["x_buffer_width_mm"] = (
            self.x_buffer_width_mm
        )
        scattering_analysis_renderer.options["y_buffer_width_mm"] = (
            self.y_buffer_width_mm
        )
        scattering_analysis_renderer.options["wb_size"] = self.hfss_wire_bond_size
        scattering_analysis_renderer.options["wb_offset"] = self.hfss_wire_bond_offset
        scattering_analysis_renderer.options["wb_threshold"] = (
            self.hfss_wire_bond_threshold
        )
        # scatteringanalysis.setup_update(max_delta_s = 0.001,
        #                                 freq_ghz=center_frequency,
        #                                 max_passes=20)
        scattering_analysis_renderer.clean_active_design()
        scattering_analysis_renderer.render_design(
            selection=self.qiskit_component_names,
            open_pins=self.open_pins,
            port_list=self.port_list,
            jj_to_port=[],
            ignored_jjs=[],
            box_plus_buffer=True,
        )
        if fine_mesh_names:
            scattering_analysis_renderer.modeler.mesh_length(
                "fine_mesh",
                fine_mesh_names,
                MaxLength=max_mesh_length,
                RefineInside=True,
            )
        start_freq = center_frequency - self.bandwidth / 2
        stop_freq = center_frequency + self.bandwidth / 2
        setup = scattering_analysis_renderer.pinfo.get_setup(name="Setup_QDO")
        try:
            setup.delete_sweep("Sweep")
        except:
            pass
        if eigenmode_setup is not None:
            setup.setup_link(eigenmode_setup)
        scattering_analysis_renderer.add_sweep(
            setup_name="Setup_QDO",
            name="Sweep",
            start_ghz=start_freq,
            stop_ghz=stop_freq,
            count=self.counts,
            type="Fast",
        )

        if len(self.port_list) == 1:
            Sij = ["S11"]

        elif len(self.port_list) == 2:
            Sij = ["S11", "S21", "S12", "S22"]

        else:
            raise ValueError(
                "Invalid number of ports. Only 1 or 2 ports are supported."
            )
        scattering_analysis_renderer.analyze_sweep("Sweep", "Setup_QDO")
        self.Sij = scattering_analysis_renderer.get_params(Sij)
        return self.Sij

    def fit_resonator(self, ports, r_freq):
        """Fit the resonator kappa using the scattering parameters.
        This method fits the resonator kappa using the scattering parameters obtained from the simulation.
        Args:
            ports (list): List of ports to use for fitting.
            r_freq (float) : resonant frequency to look for in GHz. Is used to assess is the fit succeeded or failed
        Returns:
            float: The fitted kappa value.
        """
        if len(ports) == 1:
            Sij = "S11"
            port = circuit.reflection_port()
        elif len(ports) == 2:
            Sij = "S21"
            port = circuit.notch_port()
        else:
            raise ValueError(
                "Invalid number of ports. Only 1 or 2 ports are supported."
            )

        frequency = self.Sij[-1].index
        data = self.Sij[-1][Sij].values

        port.add_data(frequency, data)
        port.autofit()
        fit_result = port.fitresults

        # Implementing a failsafe to prevent fitting when no resonant frequency was found

        if np.abs(fit_result["fr"] - r_freq) > self.bandwidth / 2:
            return None

        print("Scattering analysis" + str(fit_result))

        return fit_result

    def plot(self, title="", Sij=[]):
        """Plot the scattering parameters.
        This method generates a plot of the scattering parameters using matplotlib.
        """
        import matplotlib.pyplot as plt

        frequencies = self.Sij[-1].index
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        if Sij == []:
            if len(self.port_list) == 1:
                Sij = "S11"
            elif len(self.port_list) == 2:
                Sij = "S21"
        else:
            for i in Sij:
                Sij_values = self.Sij[-1][Sij]
                magnitude = 20 * np.log10(np.abs(Sij_values))
                phase = np.angle(Sij_values, deg=True)
                ax[0].plot(frequencies, magnitude)
                ax[0].set_title(f"Magnitude of {Sij}")
                ax[0].set_xlabel("Frequency (GHz)")
                ax[0].set_ylabel("Magnitude (dB)")
                ax[0].grid()
                ax[1].plot(frequencies, phase)
                ax[1].set_title(f"Phase of {Sij}")
                ax[1].set_xlabel("Frequency (GHz)")
                ax[1].set_ylabel("Phase (degrees)")
                ax[1].grid()
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()
