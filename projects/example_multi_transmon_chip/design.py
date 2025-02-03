from qiskit_metal.designs.design_planar import DesignPlanar
from qiskit_metal import MetalGUI

from qiskit_metal.qlibrary.qubits.transmon_pocket_teeth import TransmonPocketTeeth
from qiskit_metal.qlibrary.terminations.open_to_ground import OpenToGround
from qiskit_metal.qlibrary.tlines.meandered import RouteMeander
from qiskit_metal.qlibrary.tlines.pathfinder import RoutePathfinder

from qiskit_metal.qt.couplers.qt_route_coupler import QTRouteCoupler
from qiskit_metal.qt.couplers.qt_coupled_line_tee import QTCoupledLineTee
from qiskit_metal.qt.qt_launchpad import QTLaunchpadWirebond
import qiskit_metal.qt.simulation.utils.utils_design_variables as u

import design_variables as dv

def add_branch(design: DesignPlanar, 
               branch: int, 
               gui: MetalGUI):
    
    make_transmon_plus_resonator(design=design, branch=branch)
    
    gui.rebuild()
    gui.autoscale()

def make_transmon_plus_resonator(design:DesignPlanar,
                                 branch: int):

    # make transmon
    transmon_options = dict(
        pos_x = ['-3mm','+3mm','-3mm','+3mm','-3mm'][branch],
        pos_y = ['-4mm','-2mm','0mm','2mm','4mm'][branch],
        orientation = ['-90','+90','-90','+90','-90'][branch],
        pad_gap = '100um',
        inductor_width = '30um',
        pad_width = dv.DESIGN_VARS[dv.design_var_qb_pad_width(branch)],
        pad_height = '120um',
        pocket_width = '1200um',
        pocket_height = '1200um',
        coupled_pad_width = '45um',
        coupled_pad_height = dv.DESIGN_VARS[dv.design_var_res_qb_coupl_length(branch)],
        coupled_pad_gap = '200um',   
        connection_pads=dict(
            readout = dict(loc_W=0, loc_H=+1, pad_gap = '100um', pad_width = '40um', pad_height = '170um',
                        cpw_width = dv.RESONATOR_WIDTH, cpw_gap = dv.RESONATOR_GAP, cpw_extend = '0.0um', pocket_extent = '5um')),
        gds_cell_name = f'Manhattan_{branch}' ,
        hfss_inductance = u.design_var_lj(u.name_qb(branch)),
        hfss_capacitance = u.design_var_cj(u.name_qb(branch)),
        )    

    qub = TransmonPocketTeeth(design, u.name_qb(branch), options=transmon_options)

    # make open end of resonator
    cltee_options = dict(pos_x='0mm',  
                    pos_y=['-4mm','-2mm','0mm','2mm','4mm'][branch], 
                    orientation=['-90','+90','-90','+90','-90'][branch],
                    second_width=dv.RESONATOR_WIDTH,
                    second_gap=dv.RESONATOR_GAP,
                    prime_width=dv.LINE_50_OHM_WIDTH, 
                    prime_gap=dv.LINE_50_OHM_GAP,
                    coupling_space='10um',
                    fillet = dv.BEND_RADIUS,
                    coupling_length=dv.design_var_res_coupl_length(branch) )    
     
    cltee = QTCoupledLineTee(design, u.name_tee(branch), options=cltee_options)

    # make resonator 
    resonator_options = dict(
        pin_inputs=dict(
            start_pin=dict(component=qub.name, pin='readout'),
            end_pin=dict(component=cltee.name, pin='second_end')),
        fillet=dv.BEND_RADIUS,
        total_length = dv.design_var_res_length(branch),
        lead = dict(start_straight = '150um'),
        trace_width = dv.RESONATOR_WIDTH,
        trace_gap = dv.RESONATOR_GAP,
        meander = dict(spacing = '200um')
        )
    
    RouteMeander(design, u.name_res(branch), options= resonator_options)

def add_route_interconnects(design: DesignPlanar, branch: int, gui: MetalGUI):
    if branch%2 ==0: 
        pins = dict( 
                    start_pin=dict(component=u.name_tee(branch), pin='prime_start'),
                    end_pin=dict(  component=u.name_tee(branch+1), pin='prime_start') )
    else:
        pins = dict( 
                    start_pin=dict(component=u.name_tee(branch), pin='prime_end'),
                    end_pin=dict(  component=u.name_tee(branch+1), pin='prime_end') )
    options_rpf = dict(
                fillet='49um', hfss_wire_bonds = False,
                trace_width = dv.LINE_50_OHM_WIDTH,
                trace_gap = dv.LINE_50_OHM_GAP,
                pin_inputs = pins)
    RoutePathfinder(design, u.name_tee_to_tee(branch, branch+1), options = options_rpf)

    gui.rebuild()
    gui.autoscale()

def add_launch_pads(design: DesignPlanar, gui: MetalGUI):
                         
    launch_options = dict(chip='main', 
                        trace_width = dv.LINE_50_OHM_WIDTH, 
                        trace_gap = dv.LINE_50_OHM_GAP,
                        lead_length='30um', pad_gap='125um',
                        pad_width='260um', pad_height='260um',
                        pos_x='-5000um', pos_y='1800um', orientation='270' )
    
    launch_options['pos_x']='0mm'
    launch_options['pos_y']='4.9mm'
    launch_options['orientation']='270'
    QTLaunchpadWirebond(design, u.name_lp(0),   options = launch_options)

    launch_options['pos_x']='0mm'
    launch_options['pos_y']='-4.9mm'
    launch_options['orientation']='90'
    QTLaunchpadWirebond(design, u.name_lp(1),   options = launch_options)

    pins_top = dict( 
                    start_pin=dict(component=u.name_lp(0), pin='tie'),
                    end_pin=dict(  component=u.name_tee(4), pin='prime_start') )
    
    options_top = dict(
                fillet='49um', hfss_wire_bonds = False,
                trace_width = dv.LINE_50_OHM_WIDTH,
                trace_gap = dv.LINE_50_OHM_GAP,
                pin_inputs = pins_top)
    
    pins_bottom = dict( 
                    start_pin=dict(component=u.name_lp(1), pin='tie'),
                    end_pin=dict(  component=u.name_tee(0), pin='prime_end') )
    
    options_bottom = dict(
                fillet='49um', hfss_wire_bonds = False,
                trace_width = dv.LINE_50_OHM_WIDTH,
                trace_gap = dv.LINE_50_OHM_GAP,
                pin_inputs = pins_bottom)
    RoutePathfinder(design, u.name_lp_to_tee(0,4), options = options_top)
    RoutePathfinder(design, u.name_lp_to_tee(1,0), options = options_bottom)

    gui.rebuild()
    gui.autoscale()

