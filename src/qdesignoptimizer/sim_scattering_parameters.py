"""Study classes for capacitance matrix based simulations."""

from abc import ABC, abstractmethod
from typing import Callable, List, Literal, Optional, Union
from qiskit_metal.designs.design_base import QDesign
import numpy as np
from qiskit_metal.analyses.simulation.scattering_impedance import ScatteringImpedanceSim
from pyaedt import Hfss
from resonator_tools import circuit
from qdesignoptimizer.utils.utils import close_ansys

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
        counts (int, optional): Number of simulation runs to perform. Defaults to 25000.

    """

    def __init__(self, 
                 qiskit_component_names: List[str],
                 open_pins: List[str],
                 port_list: List[str] ,
                 x_buffer_width_mm: float = 0.5,
                y_buffer_width_mm: float = 0.5,
                render_qiskit_metal : Optional[Callable] = None,
                render_qiskit_metal_kwargs: Optional[dict] = None,
                counts: Optional[int] = 25000,
                component_of_interest: str = None,


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
        
    def set_render_qiskit_metal(self, render_qiskit_metal: Callable) -> None:
        """Set the rendering function to use before capacitance simulation.

        This method allows updating the rendering function after initialization,
        particularly useful when a DesignAnalysisState is used for within a DesignAnalysis context.

        Args:
            render_qiskit_metal (Callable): Function for rendering the design before simulation.
                Takes the form ``render_qiskit_metal(design, **kwargs)``.

        """
        self.render_qiskit_metal = render_qiskit_metal

    def simulate_scattering_parameters(self,
                                       design :QDesign,
                                       hfss_design_name: str = "Scattering_Study",
                                       center_frequency: float = 5,
                                       bandwidth: float = 1,
                                       ) :
        """ 
        Simulate scattering parameters using HFSS.

        Runs the HFSS simulation for the specified design and returns the results.

        Args:
            design (QDesign): The Qiskit Metal design object to simulate.
            hfss_design_name (str): Name of the HFSS design. Defaults to "Scattering_Study".
            center_frequency (float): Center frequency for the simulation in GHz. Defaults to 5 GHz.
            bandwidth (float): Bandwidth for the simulation in GHz. Defaults to 1 GHz.
        
        """
        

        scatteringanalysis = ScatteringImpedanceSim(design, "hfss")
        scattering_analysis_renderer = scatteringanalysis.renderer
        
        hfss = Hfss(
            designname=hfss_design_name,
            solution_type='DrivenModal',
            
        )
        scattering_analysis_renderer.activate_ansys_design(hfss_design_name, 'drivenmodal')
        for key, value in design.variables.items():
            hfss.variable_manager.set_variable(key, value)
        scattering_analysis_renderer.add_drivenmodal_setup(
            name="Setup_QDO",
            max_delta_s=0.001,
            max_passes=40,
            freq_ghz=center_frequency + bandwidth / 2,
            min_converged = 5,
            pct_refinement = 20,
            basis_order = -1,

        )
        scattering_analysis_renderer.options['x_buffer_width_mm'] = self.x_buffer_width_mm
        scattering_analysis_renderer.options['y_buffer_width_mm'] = self.y_buffer_width_mm
        # scatteringanalysis.setup_update(max_delta_s = 0.001,
        #                                 freq_ghz=center_frequency,
        #                                 max_passes=20)
        scattering_analysis_renderer.clean_active_design()
        scattering_analysis_renderer.render_design(selection=self.qiskit_component_names, 
                   open_pins=self.open_pins, 
                   port_list=self.port_list, 
                   jj_to_port=[], 
                   ignored_jjs=[],
                   box_plus_buffer = True)
        start_freq = center_frequency - bandwidth / 2
        stop_freq = center_frequency + bandwidth / 2
        scattering_analysis_renderer.add_sweep(setup_name="Setup_QDO", 
               name="Sweep", 
               start_ghz=start_freq,
               stop_ghz=stop_freq,
               count=25000,
               type="Fast"
        )

        if len(self.port_list) ==   1:
            Sij = ['S11']
            port = circuit.reflection_port()
        elif len(self.port_list) == 2:
            Sij = ['S11', 'S21', 'S12', 'S22']
            port = circuit.notch_port()
        else:
            raise ValueError("Invalid number of ports. Only 1 or 2 ports are supported.")
        scattering_analysis_renderer.analyze_sweep('Sweep', 'Setup_QDO')
        self.Sij = scattering_analysis_renderer.get_params(Sij)
        return self.Sij
    
    def fit_resonator_kappa(self,ports):
        """Fit the resonator kappa using the scattering parameters.

        This method fits the resonator kappa using the scattering parameters obtained from the simulation.

        Args:
            ports (list): List of ports to use for fitting.

        Returns:
            float: The fitted kappa value.

        """
        if len(ports) == 1:
            Sij = 'S11'
        elif len(ports) == 2:
            Sij = 'S21'
        else:
            raise ValueError("Invalid number of ports. Only 1 or 2 ports are supported.")
        
        frequency = self.Sij[-1].index
        data = self.Sij[-1][Sij].values
        port = circuit.reflection_port()
        port.add_data(frequency, data)
        port.autofit()
        fit_result = port.fitresults
        kappa = fit_result['fr']*1e9 / fit_result['absQc']
        
        return kappa

    def plot(self,title = '',Sij  = []):
        """Plot the scattering parameters.

        This method generates a plot of the scattering parameters using matplotlib.


        """
        import matplotlib.pyplot as plt
        frequencies = self.Sij[-1].index
        fig,ax = plt.subplots(1,2, figsize=(12, 6))
        if Sij == []:
            if len(self.port_list) ==   1:
                Sij = 'S11'
            elif len(self.port_list) == 2:
                Sij = 'S21'
        else :
            for i in Sij:
                Sij_values = self.Sij[-1][Sij]
                magnitude = 20 * np.log10(np.abs(Sij_values))
                phase = np.angle(Sij_values, deg=True)
                ax[0].plot(frequencies, magnitude,label= i)
                ax[0].set_title(f"Magnitude of {Sij}")
                ax[0].set_xlabel("Frequency (GHz)")
                ax[0].set_ylabel("Magnitude (dB)")
                ax[0].grid()
                ax[1].plot(frequencies, phase)
                ax[1].set_title(f"Phase of {Sij}",label = i)
                ax[1].set_xlabel("Frequency (GHz)")
                ax[1].set_ylabel("Phase (degrees)")
                ax[1].grid()
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()
        