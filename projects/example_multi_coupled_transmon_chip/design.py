from qiskit_metal.designs.design_planar import DesignPlanar
from qiskit_metal.qlibrary.qubits.transmon_pocket_teeth import TransmonPocketTeeth
from qiskit_metal.qlibrary.tlines.meandered import RouteMeander
from qiskit_metal.qlibrary.tlines.pathfinder import RoutePathfinder
from qiskit_metal.qlibrary.terminations.launchpad_wb import LaunchpadWirebond
from qiskit_metal.qlibrary.couplers.coupled_line_tee import CoupledLineTee

import design_variables as dv

def add_branch(design: DesignPlanar, 
               branch: str):
    
    make_transmon_plus_resonator(design=design, branch=branch)
    

def make_transmon_plus_resonator(design:DesignPlanar,
                                 branch: int):

    # make transmon
    transmon_options = dict(
        pos_x = ['-1.75mm','+1.75mm','-1.75mm','+1.75mm'][branch],
        pos_y = ['-1.5mm','-1mm','1.5mm','2mm'][branch],
        orientation = ['180','180','0','0'][branch],
        pad_gap = '100um',
        inductor_width = '30um',
        pad_width = dv.design_var_qb_pad_width(branch),
        pad_height = '120um',
        pocket_width = '1600um',
        pocket_height = '1200um',
        coupled_pad_width = '45um',
        coupled_pad_height = dv.design_var_res_qb_coupl_length(branch),
        coupled_pad_gap = '100um',   
        connection_pads=dict(
            readout = dict(loc_W=0, loc_H=+1, pad_gap = '100um', pad_gap_w = '0um', pad_width = '40um', pad_height = '170um',
                        cpw_width = dv.RESONATOR_WIDTH, cpw_gap = dv.RESONATOR_GAP, cpw_extend = '0.0um', pocket_extent = '5um'),
            coupler = dict(loc_W=0, loc_H=-1, pad_gap = '100um', pad_gap_w = '0um', pad_width = '40um', pad_height = '170um',
                        cpw_width = dv.RESONATOR_WIDTH, cpw_gap = dv.RESONATOR_GAP, cpw_extend = '0.0um', pocket_extent = '5um'), 
            charge_line = dict(loc_W=[1,-1,-1,1][branch], loc_H=-1, 
                               pad_gap = dv.DESIGN_VARS[dv.design_var_cl_pos_x(branch)], 
                               pad_gap_w = dv.DESIGN_VARS[dv.design_var_cl_pos_y(branch)], pad_width = '40um', pad_height = '170um',
                        cpw_width = dv.LINE_50_OHM_WIDTH, cpw_gap = dv.LINE_50_OHM_GAP, cpw_extend = '0.0um', pocket_extent = '5um', pocket_rise='0um', pad_cpw_shift='150um',)),
        gds_cell_name = f'Manhattan_{branch}' ,
        hfss_inductance = dv.design_var_lj(dv.name_qb(branch)),
        hfss_capacitance = dv.design_var_cj(dv.name_qb(branch)),
        )    

    qub = TransmonPocketTeeth(design, dv.name_qb(branch), options=transmon_options)

    # make open end of resonator
    cltee_options = dict(pos_x='0mm',  
                    pos_y=['-2.5mm','-2.5mm','2.5mm','3mm'][branch], 
                    orientation=['-90','+90','-90','+90','-90'][branch],
                    second_width=dv.RESONATOR_WIDTH,
                    second_gap=dv.RESONATOR_GAP,
                    prime_width=dv.LINE_50_OHM_WIDTH, 
                    prime_gap=dv.LINE_50_OHM_GAP,
                    coupling_space='10um',
                    fillet = dv.BEND_RADIUS,
                    coupling_length=dv.design_var_res_coupl_length(branch) )    
     
    cltee = CoupledLineTee(design, dv.name_tee(branch), options=cltee_options)

    # make resonator 
    resonator_options = dict(
        pin_inputs=dict(
            start_pin=dict(component=qub.name, pin='readout'),
            end_pin=dict(component=cltee.name, pin='second_end')),
        fillet=dv.BEND_RADIUS,
        hfss_wire_bonds = False,
        wb_size = 2,
        total_length = dv.design_var_res_length(branch),
        lead = dict(start_straight = '600um', end_straight='100um'),
        trace_width = dv.RESONATOR_WIDTH,
        trace_gap = dv.RESONATOR_GAP,
        meander = dict(spacing = '200um')
        )
    
    RouteMeander(design, dv.name_res(branch), options= resonator_options)

def add_route_interconnects(design: DesignPlanar, 
                            branch_start: int, 
                            branch_end: int, 
                            type: str):
    if type== 'start_start': 
        pins = dict( 
                    start_pin=dict(component=dv.name_tee(branch_start), pin='prime_start'),
                    end_pin=dict(  component=dv.name_tee(branch_end), pin='prime_start') )
    elif type== 'end_end': 
        pins = dict( 
                    start_pin=dict(component=dv.name_tee(branch_start), pin='prime_end'),
                    end_pin=dict(  component=dv.name_tee(branch_end), pin='prime_end') )
    elif type== 'start_end': 
        pins = dict( 
                    start_pin=dict(component=dv.name_tee(branch_start), pin='prime_start'),
                    end_pin=dict(  component=dv.name_tee(branch_end), pin='prime_end') )
    else: 
        ValueError('This type has not been implemented.')
        
    options_rpf = dict(
                fillet='49um', 
                hfss_wire_bonds = False,
                trace_width = dv.LINE_50_OHM_WIDTH,
                trace_gap = dv.LINE_50_OHM_GAP,
                pin_inputs = pins)
    RoutePathfinder(design, dv.name_tee_to_tee(branch_start, branch_end), options = options_rpf)


def add_launch_pads(design: DesignPlanar):
                         
    launch_options = dict(chip='main', 
                        trace_width = dv.LINE_50_OHM_WIDTH, 
                        trace_gap = dv.LINE_50_OHM_GAP,
                        lead_length='30um', pad_gap='125um',
                        pad_width='260um', pad_height='260um',
                        pos_x='-5000um', pos_y='1800um', orientation='270' )
    
    launch_options['pos_x']='0mm'
    launch_options['pos_y']='4.5mm'
    launch_options['orientation']='270'
    LaunchpadWirebond(design, dv.name_lp(0),   options = launch_options)

    launch_options['pos_x']='0mm'
    launch_options['pos_y']='-4.5mm'
    launch_options['orientation']='90'
    LaunchpadWirebond(design, dv.name_lp(1),   options = launch_options)

    pins_top = dict( 
                    start_pin=dict(component=dv.name_lp(0), pin='tie'),
                    end_pin=dict(  component=dv.name_tee(2), pin='prime_start') )
    
    options_top = dict(
                fillet='49um', 
                hfss_wire_bonds = False,
                trace_width = dv.LINE_50_OHM_WIDTH,
                trace_gap = dv.LINE_50_OHM_GAP,
                pin_inputs = pins_top)
    
    pins_bottom = dict( 
                    start_pin=dict(component=dv.name_lp(1), pin='tie'),
                    end_pin=dict(  component=dv.name_tee(0), pin='prime_end') )
    
    options_bottom = dict(
                fillet='49um', hfss_wire_bonds = False,
                trace_width = dv.LINE_50_OHM_WIDTH,
                trace_gap = dv.LINE_50_OHM_GAP,
                pin_inputs = pins_bottom)
    RoutePathfinder(design, dv.name_lp_to_tee(0,3), options = options_top)
    RoutePathfinder(design, dv.name_lp_to_tee(1,0), options = options_bottom)


def add_resonator_coupler(design: DesignPlanar, 
                          branch: int, 
                          branch_start: int, 
                          branch_end: int):
    resonator_options = dict(
        pin_inputs=dict(
            start_pin=dict(component=dv.name_qb(branch_start), pin='coupler'),
            end_pin=dict(component=dv.name_qb(branch_end), pin='coupler')),
        fillet=dv.BEND_RADIUS,
        hfss_wire_bonds = True,
        total_length = dv.design_var_res_coupl_length(branch),
        lead = dict(start_straight = '200um', end_straight='200um'),
        trace_width = dv.RESONATOR_WIDTH,
        trace_gap = dv.RESONATOR_GAP,
        meander = dict(spacing = '200um')
        )
    
    RouteMeander(design, dv.name_res(branch), options= resonator_options)


def add_chargeline(design: DesignPlanar, 
                   branch: int):
                         
    launch_options = dict(chip='main', 
                        trace_width = dv.LINE_50_OHM_WIDTH, 
                        trace_gap = dv.LINE_50_OHM_GAP,
                        lead_length='30um', pad_gap='125um',
                        pad_width='260um', pad_height='260um',
                        pos_x='-5000um', pos_y='1800um', orientation='270' )
    
    launch_options['pos_x']=['-4.5mm','4.5mm','-4.5mm','4.5mm'][branch]
    launch_options['pos_y']=['-2mm','-2mm','2mm','2mm'][branch]
    launch_options['orientation']=['0','180','0','180'][branch]

    LaunchpadWirebond(design, dv.name_lp_chargeline(branch),   options = launch_options)

    pins_top = dict( 
                    start_pin=dict(component=dv.name_lp_chargeline(branch), pin='tie'),
                    end_pin=dict(  component=dv.name_qb(branch), pin='charge_line') )
    
    options_chargeline = dict(
                fillet='50um', hfss_wire_bonds = False,
                trace_width = dv.LINE_50_OHM_WIDTH,
                trace_gap = dv.LINE_50_OHM_GAP,
                pin_inputs = pins_top, 
                step_size='20um', 
                lead=dict(start_straight='100um', end_straight='100um'))    
    
    RoutePathfinder(design, dv.name_charge_line(branch), options = options_chargeline)



