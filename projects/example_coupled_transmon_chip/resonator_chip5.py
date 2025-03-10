#%%
import numpy as np
import qiskit_metal as metal
from qiskit_metal import designs
from qiskit_metal import MetalGUI, Dict, open_docs
from qiskit_metal.qlibrary.terminations.short_to_ground import ShortToGround
from qiskit_metal.qlibrary.tlines.meandered import RouteMeander
from qiskit_metal.qlibrary.tlines.straight_path import RouteStraight
from qiskit_metal.qlibrary.tlines.framed_path import RouteFramed
from qiskit_metal.qt.qt_launchpad import QTLaunchpadWirebond
from qiskit_metal.qt.couplers.qt_coupled_line_tee import QTCoupledLineTee
from qiskit_metal.qt.constants import LINE_50_OHM_GAP, LINE_50_OHM_WIDTH
from qiskit_metal.qt.fabrication.gds_generation.chip_generator import render_single_chip_to_gds
import klayout.db as db
from qiskit_metal.analyses import EPRanalysis

design = designs.DesignPlanar()
design.overwrite_enabled = True

design.chips.main.size['size_x'] = '7mm'
design.chips.main.size['size_y'] = '7mm'
design.chips.main.size['size_z'] = '-280um'

gui = MetalGUI(design)

RES_WIDTH = '20um'
RES_GAP = '20um'

#%%
"""ESTIMATE LENGTH"""
eps_eff = 6.44 # assuming eps_r = 11.9
vp = 3e8/np.sqrt(eps_eff)

def freq2length(f, vp=vp):
    return int(vp/f*1e6/4) #length of lbda/4 resonator

def length2freq(l, vp=vp):
    return int(vp/4/(l*1e-6))

l = freq2length(5e9)
print(l)
f = print(length2freq(l))

#%%
"""CREATE LAUNCHERS"""

launch_options = dict(
    chip='main', 
    trace_width = LINE_50_OHM_WIDTH, 
    trace_gap = LINE_50_OHM_GAP,
    lead_length='30um', 
    pad_gap='125um',
    pad_width='260um', 
    pad_height='200um',
    )

launch_options['pos_x']='-3050um'
launch_options['pos_y']='-1800um'
launch_options['orientation']='0'
lp1 = QTLaunchpadWirebond(design, 'lp1',   options = launch_options)

launch_options['pos_x']='3050um'
launch_options['pos_y']='1800um'
launch_options['orientation']='180'
lp2 = QTLaunchpadWirebond(design, 'lp2',   options = launch_options)

gui.rebuild()
gui.autoscale() 

#%% 
"""CREATE RESONATORS"""

def resonator(ID, f, tee_pos, coupling_length, prime_length, coupling_space, short_pos, res_gap, down_length = 100, fillet = 60,  right = False, orientation = '90'):
    
    if right:
        mirror = False
    else:
        mirror = True
    tee_options = Dict(prime_width=LINE_50_OHM_WIDTH, prime_gap=LINE_50_OHM_GAP, prime_length_1=str(prime_length)+'um', prime_length_2=str(prime_length)+'um',
                    second_width=RES_WIDTH, second_gap=res_gap, coupling_space=str(coupling_space)+'um', coupling_length=str(coupling_length)+'um', 
                    down_length=str(down_length)+'um', fillet=str(fillet)+'um', mirror=mirror, open_termination=True,
                    pos_x=str(tee_pos[0])+'um', pos_y=str(tee_pos[1])+'um', orientation=orientation)
    tee1 = QTCoupledLineTee(design, 'tee'+ID, options=tee_options)

    spacing = 200
    L = int(freq2length(f)) - coupling_length - down_length

    short1_options = Dict(pos_x=str(short_pos[0])+'um', pos_y=str(short_pos[1])+'um', orientation='-90', termination_gap='10um')
    short1 = ShortToGround(design,'short'+ID,options=short1_options)

    #f = 5e9
    res1_options = Dict(total_length=str(L)+'um', fillet=str(-5+spacing/2)+'um', lead = dict(start_straight='100um', end_straight='0um'),
                        pin_inputs=Dict(start_pin=Dict(component='tee'+ID, pin='second_end'),
                                        end_pin=Dict(component='short'+ID, pin='short')),
                        meander=Dict(spacing=str(spacing)+'um', asymmetry='0um'),
                        trace_width=RES_WIDTH, trace_gap=res_gap)
    res1 = RouteMeander(design, 'res'+ID,  options=res1_options)

simulation = False

### 7.25 GHz ###
ID = '1'
f = 7.25e9-0.175e9
coupling_length = 300
if simulation:
    prime_length = coupling_length+700
else:
    prime_length = 0
down_length = 200
coupling_space = 70
tee_pos = [-2500, 500]
tee_fillet = 80
right = True
orientation = '90'
res_gap = '30um'
short_pos = [-1000, tee_pos[1]-coupling_length+200]

resonator(ID, f, tee_pos, coupling_length, prime_length, coupling_space, short_pos, res_gap=res_gap, down_length = down_length, fillet = tee_fillet, right = right, orientation=orientation)

### 7 GHz ###
ID = '2'
f = 7e9-0.12e9
coupling_length = 300
if simulation:
    prime_length = coupling_length+700
else:
    prime_length = 0
down_length = 150
coupling_space = 70
tee_pos = [-2500, 2500]
right = True
orientation = '90'
res_gap = '20um'
short_pos = [-900, tee_pos[1]-coupling_length+150]

resonator(ID, f, tee_pos, coupling_length, prime_length, coupling_space, short_pos, res_gap=res_gap, down_length = down_length, right = right, orientation=orientation)

### 4.5 GHz ###
ID = '3'
f = 4.5e9-0.05e9
coupling_length = 300
if simulation:
    prime_length = coupling_length+700
else:
    prime_length = 0
down_length = 200
coupling_space = 33
tee_pos = [0, 2500]
right = True
orientation = '90'
res_gap = '10um'
short_pos = [2250, tee_pos[1]-coupling_length]

resonator(ID, f, tee_pos, coupling_length, prime_length, coupling_space, short_pos, res_gap=res_gap, down_length = down_length, right = right, orientation=orientation)

### 6 GHz ###
ID = '4'
f = 6e9-0.11e9
coupling_length = 200
if simulation:
    prime_length = coupling_length+700
else:
    prime_length = 0
down_length = 150
coupling_space = 35
tee_pos = [0, 1500]
right = False
orientation = '-90'
res_gap = '20um'
short_pos = [-1950, tee_pos[1]-coupling_length]

resonator(ID, f, tee_pos, coupling_length, prime_length, coupling_space, short_pos, res_gap=res_gap, down_length = down_length, right = right, orientation=orientation)

### 4.75 GHz ###
ID = '5'
f = 4.75e9-0.08e9
coupling_length = 200
if simulation:
    prime_length = coupling_length+700
else:
    prime_length = 0
down_length = 150
coupling_space = 30
tee_pos = [0, 1500]
right = True
orientation = '90'
res_gap = '20um'
short_pos = [2000, tee_pos[1]-coupling_length]

resonator(ID, f, tee_pos, coupling_length, prime_length, coupling_space, short_pos, res_gap=res_gap, down_length = down_length, right = right, orientation=orientation)

### 5.75 GHz ###
ID = '6'
f = 5.75e9-0.081e9
coupling_length = 200
if simulation:
    prime_length = coupling_length+700
else:
    prime_length = 0
down_length = 150
coupling_space = 32
tee_pos = [0, 500]
right = True
orientation = '90'
res_gap = '10um'
short_pos = [1550, tee_pos[1]-coupling_length]

resonator(ID, f, tee_pos, coupling_length, prime_length, coupling_space, short_pos, res_gap=res_gap, down_length = down_length, right = right, orientation=orientation)

### 5.5 GHz ###
ID = '7'
f = 5.5e9-0.14e9
coupling_length = 300
if simulation:
    prime_length = coupling_length+700
else:
    prime_length = 0
down_length = 150
coupling_space = 50
tee_pos = [0, -500]
tee_fillet = 80
right = False
orientation = '-90'
res_gap = '30um'
short_pos = [-1800, tee_pos[1]]

resonator(ID, f, tee_pos, coupling_length, prime_length, coupling_space, short_pos, res_gap=res_gap, down_length = down_length, fillet=tee_fillet, right = right, orientation=orientation)

### 5.4 GHz ###
ID = '8'
f = 5.4e9-0.105e9
coupling_length = 300
if simulation:
    prime_length = coupling_length+700
else:
    prime_length = 0
down_length = 150
coupling_space = 53
tee_pos = [0, -1500]
right = False
orientation = '-90'
res_gap = '20um'
short_pos = [-2000, tee_pos[1]]
resonator(ID, f, tee_pos, coupling_length, prime_length, coupling_space, short_pos,  res_gap=res_gap, down_length = down_length, right = right, orientation=orientation)

### 5.3 GHz ###
ID = '9'
f = 5.3e9-0.065e9
coupling_length = 200
if simulation:
    prime_length = coupling_length+700
else:
    prime_length = 0
down_length = 150
coupling_space = 35
tee_pos = [0, -1500]
right = True
orientation = '90'
res_gap = '10um'
short_pos = [1950, tee_pos[1]-coupling_length]

resonator(ID, f, tee_pos, coupling_length, prime_length, coupling_space, short_pos, res_gap=res_gap, down_length = down_length, right = right, orientation=orientation)

### 5 GHz ###
ID = '10'
f = 5e9-0.12e9
coupling_length = 200
if simulation:
    prime_length = coupling_length+700
else:
    prime_length = 0
down_length = 150
coupling_space = 40
tee_pos = [0, -2500]
tee_fillet = 80
right = False
orientation = '-90'
res_gap = '30um'
short_pos = [-1950, tee_pos[1]-coupling_length]

resonator(ID, f, tee_pos, coupling_length, prime_length, coupling_space, short_pos, res_gap=res_gap, down_length = down_length, fillet=tee_fillet, right = right, orientation=orientation)

### 6.75 GHz ###
ID = '11'
f = 6.75e9-0.08e9
coupling_length = 300
if simulation:
    prime_length = coupling_length+700
else:
    prime_length = 0
down_length = 150
coupling_space = 48
tee_pos = [2500, -2500]
right = False
orientation = '-90'
res_gap = '10um'
short_pos = [950, tee_pos[1]]

resonator(ID, f, tee_pos, coupling_length, prime_length, coupling_space, short_pos, res_gap=res_gap, down_length = down_length, right = right, orientation=orientation)

### 6.5 GHz ###
ID = '12'
f = 6.5e9-0.17e9
coupling_length = 200
if simulation:
    prime_length = coupling_length+700
else:
    prime_length = 0
down_length = 150
coupling_space = 50
tee_pos = [2500, -500]
tee_fillet = 80
right = False
orientation = '-90'
res_gap = '30um'
short_pos = [900, tee_pos[1]]

resonator(ID, f, tee_pos, coupling_length, prime_length, coupling_space, short_pos, res_gap=res_gap, down_length = down_length, fillet=tee_fillet, right = right, orientation=orientation)

gui.rebuild()
gui.autoscale() 

#%%
# """CHECK QUALITY FACTOR"""

# eig_res = EPRanalysis(design, "hfss")
# em_p = eig_res.sim.setup
# em_p.name = 'Setup'#'TeeRes'
# em_p.min_freq_ghz = 4
# em_p.n_modes = 1
# em_p.max_passes = 15
# em_p.max_delta_f = 0.1
# #em_p.pct_refinement = 30
# em_p.min_converged = 2
# em_p.basis_order = 2
# em_p.vars = Dict({'Lj': '0nH', 'Cj': '0 fF'})

# hfss = eig_res.sim.renderer
# hfss.options['x_buffer_width_mm'] = 0.4
# hfss.options['y_buffer_width_mm'] = 0.4
# # hfss.options['max_mesh_length_jj'] = '7um'
# # hfss.options['max_mesh_length_port'] = '7um'

# for ID in [11]:#np.arange(1,13):

#     eig_res.sim.run(name="res_sim",
#                 components=['tee'+str(ID), 'res'+str(ID), 'short'+str(ID)],
#                 open_terminations=[],
#                 port_list=[('tee'+str(ID), 'prime_start','50'), ('tee'+str(ID), 'prime_end','50')],
#                 box_plus_buffer = True)
    
#     print(ID)
#     print(eig_res.get_frequencies())
#     hfss = eig_res.sim.renderer
#     hfss.modeler._modeler.ShowWindow()
#     hfss.plot_ansys_fields('main')

#%% 
"""CONNECT ELEMENTS"""

meander_options_lp11 = Dict(fillet='200um',
                            pin_inputs=Dict(start_pin=Dict(component='lp1', pin='tie'),
                                        end_pin=Dict(component='tee1', pin='prime_start')),
                        trace_width=LINE_50_OHM_WIDTH, trace_gap=LINE_50_OHM_GAP)
line12= RouteFramed(design, 'lp11',  options=meander_options_lp11)

meander_options_12 = Dict(fillet='200um',
                            pin_inputs=Dict(start_pin=Dict(component='tee1', pin='prime_end'),
                                        end_pin=Dict(component='tee2', pin='prime_start')),
                        trace_width=LINE_50_OHM_WIDTH, trace_gap=LINE_50_OHM_GAP)
line12= RouteFramed(design, '12',  options=meander_options_12)

meander_options_23 = Dict(total_length='2 mm', fillet='250 um', lead = dict(start_straight='600um', end_straight='600um'),
                       pin_inputs=Dict(start_pin=Dict(component='tee2', pin='prime_end'),
                                        end_pin=Dict(component='tee3', pin='prime_end')),
                        meander=Dict(spacing='1000um', asymmetry='00um'),
                        trace_width=LINE_50_OHM_WIDTH, trace_gap=LINE_50_OHM_GAP)
line23 = RouteMeander(design, 'line23',  options=meander_options_23)

meander_options_34 = Dict(pin_inputs=Dict(start_pin=Dict(component='tee3', pin='prime_start'),
                                        end_pin=Dict(component='tee4', pin='prime_start')),
                        trace_width=LINE_50_OHM_WIDTH, trace_gap=LINE_50_OHM_GAP)
line34 = RouteStraight(design, 'line34',  options=meander_options_34)

meander_options_56 = Dict(pin_inputs=Dict(start_pin=Dict(component='tee5', pin='prime_start'),
                                        end_pin=Dict(component='tee6', pin='prime_end')),
                        trace_width=LINE_50_OHM_WIDTH, trace_gap=LINE_50_OHM_GAP)
line56 = RouteStraight(design, 'line56',  options=meander_options_56)

meander_options_67 = Dict(pin_inputs=Dict(start_pin=Dict(component='tee6', pin='prime_start'),
                                        end_pin=Dict(component='tee7', pin='prime_start')),
                        trace_width=LINE_50_OHM_WIDTH, trace_gap=LINE_50_OHM_GAP)
line67 = RouteStraight(design, 'line67',  options=meander_options_67)

meander_options_78 = Dict(pin_inputs=Dict(start_pin=Dict(component='tee7', pin='prime_end'),
                                        end_pin=Dict(component='tee8', pin='prime_start')),
                        trace_width=LINE_50_OHM_WIDTH, trace_gap=LINE_50_OHM_GAP)
line78 = RouteStraight(design, 'line78',  options=meander_options_78)

meander_options_910 = Dict(pin_inputs=Dict(start_pin=Dict(component='tee9', pin='prime_start'),
                                        end_pin=Dict(component='tee10', pin='prime_start')),
                        trace_width=LINE_50_OHM_WIDTH, trace_gap=LINE_50_OHM_GAP)
line910 = RouteStraight(design, 'line910',  options=meander_options_910)

meander_options_1011 = Dict(total_length='1 mm', fillet='250 um', lead = dict(start_straight='600um', end_straight='400um'),
                       pin_inputs=Dict(start_pin=Dict(component='tee10', pin='prime_end'),
                                        end_pin=Dict(component='tee11', pin='prime_end')),
                        meander=Dict(spacing='2100um', asymmetry='00um'),
                        trace_width=LINE_50_OHM_WIDTH, trace_gap=LINE_50_OHM_GAP)
line1011 = RouteMeander(design, 'line1011',  options=meander_options_1011)

meander_options_1112 = Dict(pin_inputs=Dict(start_pin=Dict(component='tee11', pin='prime_start'),
                                        end_pin=Dict(component='tee12', pin='prime_end')),
                        trace_width=LINE_50_OHM_WIDTH, trace_gap=LINE_50_OHM_GAP)
line1112 = RouteStraight(design, 'line1112',  options=meander_options_1112)

meander_options_12lp2 = Dict(fillet='200um',
                            pin_inputs=Dict(start_pin=Dict(component='tee12', pin='prime_start'),
                                        end_pin=Dict(component='lp2', pin='tie')),
                        trace_width=LINE_50_OHM_WIDTH, trace_gap=LINE_50_OHM_GAP)
line12lp2= RouteFramed(design, '12lp2',  options=meander_options_12lp2)

gui.rebuild()
gui.autoscale() 

#%% 
"""EXPORT TO GDS"""

gds_filename = 'HighQ_res_4PortsHolder'
render_single_chip_to_gds(design=design, 
                          path='./', 
                          gds_filename=gds_filename, 
                          junction_filename='Junctions_TransmonChip', 
                          cheesing=True)

path = "./" + gds_filename + ".gds"
layout = db.Layout()
layout.read(path)
top_cell_index = layout.cell_by_name('TOP')
top_cell = layout.cell(top_cell_index)

def make_markers():
        layer = layout.layer(2, 0)        
        displacements_x = np.array([-3500, 3500], dtype=float)
        displacements_y = np.array([-3500,3500], dtype=float)
        for i, disp_x in enumerate(displacements_x):
            for j, disp_y in enumerate(displacements_y):
                square_size = 20
                rect_length = 450
                rect_width = 6 
                marker_distance = 40
                square_box = db.DBox(-square_size / 2, -square_size / 2, square_size / 2, square_size / 2) 
                north_rect = db.DBox(-rect_width / 2, square_size / 2 + marker_distance, rect_width / 2, square_size / 2 + rect_length + marker_distance)       
                south_rect = db.DBox(-rect_width / 2, -square_size / 2 - rect_length - marker_distance, rect_width / 2, -square_size / 2 - marker_distance)
                east_rect = db.DBox(square_size / 2 + marker_distance, -rect_width / 2, square_size / 2 + rect_length + marker_distance, rect_width / 2)
                west_rect = db.DBox(-square_size / 2 - rect_length - marker_distance, -rect_width / 2, -square_size / 2 - marker_distance, rect_width / 2)

                markers = [square_box, north_rect, south_rect, east_rect, west_rect] 
                markers = [marker * db.DTrans(0,0,int(disp_x),int(disp_y)) for marker in markers] 
                [top_cell.shapes(layer).insert(db.DPolygon(marker)) for marker in markers]

make_markers()
layout.write(path)

# %%
