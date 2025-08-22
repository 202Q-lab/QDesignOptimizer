import names as n
from qiskit_metal.designs.design_planar import DesignPlanar
from qiskit_metal.qlibrary.couplers.coupled_line_tee import CoupledLineTee
from qiskit_metal.qlibrary.couplers.tunable_coupler_01 import TunableCoupler01
from qiskit_metal.qlibrary.qubits.transmon_cross import TransmonCross
from qiskit_metal.qlibrary.terminations.launchpad_wb import LaunchpadWirebond
from qiskit_metal.qlibrary.terminations.open_to_ground import OpenToGround
from qiskit_metal.qlibrary.terminations.short_to_ground import ShortToGround
from qiskit_metal.qlibrary.tlines.meandered import RouteMeander
from qiskit_metal.qlibrary.tlines.pathfinder import RoutePathfinder
from qiskit_metal.toolbox_python.attr_dict import Dict

from qdesignoptimizer.utils.chip_generation import ChipType
from qdesignoptimizer.utils.utils import sum_expression

# Fixed design constants
LINE_50_OHM_WIDTH = "16.51um"
LINE_50_OHM_GAP = "10um"

RESONATOR_WIDTH = LINE_50_OHM_WIDTH
RESONATOR_GAP = LINE_50_OHM_GAP

BEND_RADIUS = "99um"

chip_type = ChipType(size_x="10mm", size_y="10mm", size_z="-300um", material="silicon")


def add_transmon_plus_resonator(design: DesignPlanar, group: int):
    nbr_idx = group - 1  # zero indexed

    qubit = [n.QUBIT_1, n.QUBIT_2][nbr_idx]
    resonator = [n.RESONATOR_1, n.RESONATOR_2][nbr_idx]

    # default_options = dict(

    #         _default_connection_pads=dict(
    #             connector_type='0',  # 0 = Claw type, 1 = gap type
    #             claw_length='30um',
    #             ground_spacing='5um',
    #             claw_width='10um',
    #             claw_gap='6um',
    #             claw_cpw_length='40um',
    #             claw_cpw_width='10um',
    #             connector_location=
    #             '0'  # 0 => 'west' arm, 90 => 'north' arm, 180 => 'east' arm
    #         ))

    # make transmon
    start_offset = ["-65um", "65um"][nbr_idx]
    coupler_width_offset = ["-250um", "250um"][nbr_idx]
    coupl_dist = design.variables[n.design_var_coupl_length(qubit, n.COUPLER_12)]
    y_pos_xmon = sum_expression(["-80um", "-" + coupl_dist])
    transmon_options = Dict(
        pos_x=sum_expression(
            [start_offset, coupler_width_offset, ["-", ""][nbr_idx] + coupl_dist]
        ),
        pos_y=y_pos_xmon,
        cross_width="30um",
        cross_length=n.design_var_width(qubit),
        cross_gap="30um",
        chip="main",
        connection_pads={
            "pad1": Dict(
                connector_type="0",  # 0 = Claw type, 1 = gap type
                claw_length=design.variables[
                    n.design_var_coupl_length(resonator, qubit)
                ],
                ground_spacing="5um",
                claw_width=LINE_50_OHM_WIDTH,  #
                claw_gap=LINE_50_OHM_GAP,
                claw_cpw_length="40um",
                claw_cpw_width=LINE_50_OHM_WIDTH,
                connector_location=["0", "180"][
                    nbr_idx
                ],  # 0 => 'west' arm, 90 => 'north' arm, 180 => 'east' arm
            )
        },
        # orientation=["180", "0"][nbr_idx],
        # pad_gap="100um",
        # inductor_width="30um",
        # pad_width=n.design_var_width(qubit),
        # pad_height="120um",
        # pocket_width="1200um",
        # pocket_height="1200um",
        # coupled_pad_width="0um",
        # coupled_pad_height="0um",
        # coupled_pad_gap="100um",
        # connection_pads=dict(
        #     readout=dict(
        #         loc_W=0,
        #         loc_H=+1,
        #         pad_gap="120um",
        #         pad_gap_w="0um",
        #         pad_width=n.design_var_coupl_length(resonator, qubit),
        #         pad_height="40um",
        #         cpw_width=RESONATOR_WIDTH,
        #         cpw_gap=RESONATOR_GAP,
        #         cpw_extend="300um",
        #         pocket_extent="5um",
        #     ),
        #     coupler=dict(
        #         loc_W=0,
        #         loc_H=-1,
        #         pad_gap="100um",
        #         pad_gap_w="0um",
        #         pad_width="40um",
        #         pad_height="170um",
        #         cpw_width=RESONATOR_WIDTH,
        #         cpw_gap=RESONATOR_GAP,
        #         cpw_extend="0.0um",
        #         pocket_extent="5um",
        #     ),
        # ),
        gds_cell_name=f"Manhattan_{group}",
        hfss_inductance=n.design_var_lj(qubit),
        hfss_capacitance=n.design_var_cj(qubit),
    )

    qub = TransmonCross(design, n.name_mode(qubit), options=transmon_options)

    # # make open end of resonator
    # cltee_options = dict(
    #     pos_x=["-1.5mm", "1.5mm"][nbr_idx],
    #     pos_y="-1mm",
    #     orientation=["180", "180"][nbr_idx],
    #     second_width=RESONATOR_WIDTH,
    #     second_gap=RESONATOR_GAP,
    #     prime_width=LINE_50_OHM_WIDTH,
    #     prime_gap=LINE_50_OHM_GAP,
    #     coupling_space=n.design_var_length(f"{resonator}_capacitance"),
    #     fillet=BEND_RADIUS,
    #     coupling_length=n.design_var_coupl_length(resonator, "tee"),
    #     mirror=[True, False][nbr_idx],
    # )
    # cltee = CoupledLineTee(design, n.name_tee(group), options=cltee_options)

    otg_options = dict(
        pos_x=["-2mm", "2mm"][nbr_idx],
        pos_y=y_pos_xmon,
        orientation=["180", "0"][nbr_idx],
        width=LINE_50_OHM_WIDTH,
        gap=LINE_50_OHM_GAP,
        termination_gap=LINE_50_OHM_GAP,
    )

    ShortToGround(design, n.name_tee(group), options=otg_options)

    # make resonator
    resonator_options = dict(
        pin_inputs=dict(
            start_pin=dict(component=qub.name, pin="pad1"),
            end_pin=dict(component=n.name_tee(group), pin="short"),
        ),
        fillet=BEND_RADIUS,
        total_length=n.design_var_length(resonator),
        lead=dict(start_straight="100um", end_straight="100um"),
        trace_width=RESONATOR_WIDTH,
        trace_gap=RESONATOR_GAP,
        meander=dict(spacing="200um"),
    )

    RouteMeander(design, n.name_mode(resonator), options=resonator_options)


def add_route_interconnects(design: DesignPlanar):

    pins = dict(
        start_pin=dict(component=n.name_tee(1), pin="prime_start"),
        end_pin=dict(component=n.name_tee(2), pin="prime_end"),
    )

    options_rpf = dict(
        fillet="49um",
        hfss_wire_bonds=False,
        trace_width=LINE_50_OHM_WIDTH,
        trace_gap=LINE_50_OHM_GAP,
        pin_inputs=pins,
    )
    RoutePathfinder(design, n.name_tee_to_tee(1, 2), options=options_rpf)


def add_launch_pads(design: DesignPlanar):

    launch_options = dict(
        chip="main",
        trace_width=LINE_50_OHM_WIDTH,
        trace_gap=LINE_50_OHM_GAP,
        lead_length="30um",
        pad_gap="125um",
        pad_width="260um",
        pad_height="260um",
        pos_x="-5000um",
        pos_y="1800um",
        orientation="270",
    )

    launch_options["pos_x"] = "0mm"
    launch_options["pos_y"] = "4.5mm"
    launch_options["orientation"] = "270"
    LaunchpadWirebond(design, n.name_lp(0), options=launch_options)

    launch_options["pos_x"] = "0mm"
    launch_options["pos_y"] = "-4.5mm"
    launch_options["orientation"] = "90"
    LaunchpadWirebond(design, n.name_lp(1), options=launch_options)

    pins_top = dict(
        start_pin=dict(component=n.name_lp(0), pin="tie"),
        end_pin=dict(component=n.name_tee(2), pin="prime_start"),
    )

    options_top = dict(
        fillet="49um",
        hfss_wire_bonds=False,
        trace_width=LINE_50_OHM_WIDTH,
        trace_gap=LINE_50_OHM_GAP,
        pin_inputs=pins_top,
    )

    pins_bottom = dict(
        start_pin=dict(component=n.name_lp(1), pin="tie"),
        end_pin=dict(component=n.name_tee(1), pin="prime_end"),
    )

    options_bottom = dict(
        fillet="49um",
        hfss_wire_bonds=False,
        trace_width=LINE_50_OHM_WIDTH,
        trace_gap=LINE_50_OHM_GAP,
        pin_inputs=pins_bottom,
    )
    RoutePathfinder(design, n.name_lp_to_tee(0, 2), options=options_top)
    RoutePathfinder(design, n.name_lp_to_tee(1, 1), options=options_bottom)


def add_coupler(design):
    length = design.variables[n.design_var_length(n.COUPLER_12)]
    default_options = Dict(
        c_width="500um",
        l_width="30um",
        l_gap="20um",
        a_height=length,
        cp_height="9um",
        cp_arm_length="30um",
        cp_arm_width="6um",
        cp_gap="6um",
        cp_gspace="3um",
        fl_width="5um",
        fl_gap="3um",
        fl_length="10um",
        fl_ground="2um",
        pos_x="0mm",
        pos_y="0mm",
        hfss_inductance=n.design_var_lj(n.COUPLER_12),
        hfss_capacitance=n.design_var_cj(n.COUPLER_12),
    )
    # resonator_options = dict(
    #     pin_inputs=dict(
    #         start_pin=dict(component=n.name_mode(n.QUBIT_1), pin="coupler"),
    #         end_pin=dict(component=n.name_mode(n.QUBIT_2), pin="coupler"),
    #     ),
    #     fillet=BEND_RADIUS,
    #     hfss_wire_bonds=True,
    #     total_length=n.design_var_length(n.COUPLER_12),
    #     lead=dict(start_straight="200um", end_straight="200um"),
    #     trace_width=RESONATOR_WIDTH,
    #     trace_gap=RESONATOR_GAP,
    #     meander=dict(spacing="200um"),
    # )

    TunableCoupler01(design, n.name_mode(n.COUPLER_12), options=default_options)


def add_chargeline(design: DesignPlanar, group: int):
    nbr_idx = group - 1
    qubit = [n.QUBIT_1, n.QUBIT_2][nbr_idx]
    lp_nbr = [2, 3][nbr_idx]

    launch_options = dict(
        chip="main",
        trace_width=LINE_50_OHM_WIDTH,
        trace_gap=LINE_50_OHM_GAP,
        lead_length="30um",
        pad_gap="125um",
        pad_width="260um",
        pad_height="260um",
        pos_x="-4.5mm",
        pos_y="-2mm",
        orientation="0",
    )
    launch_options["pos_y"] = ["-2mm", "2mm"][nbr_idx]
    LaunchpadWirebond(design, n.name_lp(lp_nbr), options=launch_options)
    x_cl_offset = ["-2350um", "-2350um"][nbr_idx]
    x_cl_absolute = sum_expression(
        [design.variables[n.design_var_cl_pos_x(qubit)], x_cl_offset]
    )  # transmon position + pocket width/2

    y_cl_offset = ["-1500um", "+1500um"][nbr_idx]
    y_cl_absolute = sum_expression(
        [design.variables[n.design_var_cl_pos_y(qubit)], y_cl_offset]
    )  # transmon position + pocket height/2

    otg_options = dict(
        pos_x=x_cl_absolute,
        pos_y=y_cl_absolute,
        orientation="0",
        width=LINE_50_OHM_WIDTH,
        gap=LINE_50_OHM_GAP,
        termination_gap=LINE_50_OHM_GAP,
    )

    OpenToGround(design, n.name_id("otg_" + qubit), options=otg_options)

    pins_top = dict(
        start_pin=dict(component=n.name_lp(lp_nbr), pin="tie"),
        end_pin=dict(component=n.name_id("otg_" + qubit), pin="open"),
    )

    options_chargeline = dict(
        fillet="90um",
        hfss_wire_bonds=True,
        trace_width=LINE_50_OHM_WIDTH,
        trace_gap=LINE_50_OHM_GAP,
        pin_inputs=pins_top,
        step_size="20um",
        lead=dict(start_straight="100um", end_straight="600um"),
    )

    RoutePathfinder(design, n.name_charge_line(group), options=options_chargeline)


# Function for meshing names for mapping for finer meshing
def CoupledLineTee_mesh_names(comp_names):
    all_names_to_mesh = [f"prime_cpw_{comp_names}", f"second_cpw_{comp_names}"]
    return all_names_to_mesh


# Function to render the design
def render_qiskit_metal_design(design, gui, capacitance_or_surface_p_ratio=False):
    add_transmon_plus_resonator(design, group=n.NBR_1)
    add_transmon_plus_resonator(design, group=n.NBR_2)
    add_coupler(design)
    # add_route_interconnects(design)
    # add_launch_pads(design)
    # add_chargeline(design, group=n.NBR_1)
    # add_chargeline(design, group=n.NBR_2)

    if capacitance_or_surface_p_ratio == True:
        for component in design.components.values():
            if "hfss_wire_bonds" in component.options:
                component.options["hfss_wire_bonds"] = False

    gui.rebuild()
    gui.autoscale()
