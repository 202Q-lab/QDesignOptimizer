import names as n
from qiskit_metal.designs.design_planar import DesignPlanar
from qiskit_metal.qlibrary.couplers.coupled_line_tee import CoupledLineTee
from qiskit_metal.qlibrary.qubits.transmon_pocket_teeth import TransmonPocketTeeth
from qiskit_metal.qlibrary.terminations.launchpad_wb import LaunchpadWirebond
from qiskit_metal.qlibrary.terminations.open_to_ground import OpenToGround
from qiskit_metal.qlibrary.tlines.meandered import RouteMeander
from qiskit_metal.qlibrary.tlines.pathfinder import RoutePathfinder

from qdesignoptimizer.utils.chip_generation import ChipType
from qdesignoptimizer.utils.utils import sum_expression

# Fixed design constants
LINE_50_OHM_WIDTH = "16.51um"
LINE_50_OHM_GAP = "10um"

RESONATOR_WIDTH = "20um"
RESONATOR_GAP = "20um"

BEND_RADIUS = "99um"

chip_type = ChipType(size_x="10mm", size_y="10mm", size_z="-300um", material="silicon")


def add_transmon_plus_resonator(design: DesignPlanar, group: int):
    nbr_idx = group - 1  # zero indexed

    qubit = [n.QUBIT_1, n.QUBIT_2][nbr_idx]
    resonator = [n.RESONATOR_1, n.RESONATOR_2][nbr_idx]

    # make transmon
    transmon_options = dict(
        pos_x=["-1.75mm", "-1.75mm"][nbr_idx],
        pos_y=["-1.5mm", "1.5mm"][nbr_idx],
        orientation=["180", "0"][nbr_idx],
        pad_gap="100um",
        inductor_width="30um",
        pad_width=n.design_var_width(qubit),
        pad_height="120um",
        pocket_width="1200um",
        pocket_height="1200um",
        coupled_pad_width="0um",
        coupled_pad_height="0um",
        coupled_pad_gap="100um",
        connection_pads=dict(
            readout=dict(
                loc_W=0,
                loc_H=+1,
                pad_gap="120um",
                pad_gap_w="0um",
                pad_width=n.design_var_coupl_length(resonator, qubit),
                pad_height="40um",
                cpw_width=RESONATOR_WIDTH,
                cpw_gap=RESONATOR_GAP,
                cpw_extend="300um",
                pocket_extent="5um",
            ),
            coupler=dict(
                loc_W=0,
                loc_H=-1,
                pad_gap="100um",
                pad_gap_w="0um",
                pad_width="40um",
                pad_height="170um",
                cpw_width=RESONATOR_WIDTH,
                cpw_gap=RESONATOR_GAP,
                cpw_extend="0.0um",
                pocket_extent="5um",
            ),
        ),
        gds_cell_name=f"Manhattan_{group}",
        hfss_inductance=n.design_var_lj(qubit),
        hfss_capacitance=n.design_var_cj(qubit),
    )

    qub = TransmonPocketTeeth(design, n.name_mode(qubit), options=transmon_options)

    # make open end of resonator
    cltee_options = dict(
        pos_x="0mm",
        pos_y=["-2.9mm", "2.9mm"][nbr_idx],
        orientation=["-90", "-90"][nbr_idx],
        second_width=RESONATOR_WIDTH,
        second_gap=RESONATOR_GAP,
        prime_width=LINE_50_OHM_WIDTH,
        prime_gap=LINE_50_OHM_GAP,
        coupling_space=n.design_var_length(f"{resonator}_capacitance"),
        fillet=BEND_RADIUS,
        coupling_length=n.design_var_coupl_length(resonator, "tee"),
    )

    cltee = CoupledLineTee(design, n.name_tee(group), options=cltee_options)

    # make resonator
    resonator_options = dict(
        pin_inputs=dict(
            start_pin=dict(component=qub.name, pin="readout"),
            end_pin=dict(component=cltee.name, pin="second_end"),
        ),
        fillet=BEND_RADIUS,
        hfss_wire_bonds=True,
        total_length=n.design_var_length(resonator),
        lead=dict(start_straight="600um", end_straight="100um"),
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


def add_coupler(design: DesignPlanar):
    resonator_options = dict(
        pin_inputs=dict(
            start_pin=dict(component=n.name_mode(n.QUBIT_1), pin="coupler"),
            end_pin=dict(component=n.name_mode(n.QUBIT_2), pin="coupler"),
        ),
        fillet=BEND_RADIUS,
        hfss_wire_bonds=True,
        total_length=n.design_var_length(n.COUPLER_12),
        lead=dict(start_straight="200um", end_straight="200um"),
        trace_width=RESONATOR_WIDTH,
        trace_gap=RESONATOR_GAP,
        meander=dict(spacing="200um"),
    )

    RouteMeander(design, n.name_mode(n.COUPLER_12), options=resonator_options)


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
        hfss_wire_bonds=False,
        trace_width=LINE_50_OHM_WIDTH,
        trace_gap=LINE_50_OHM_GAP,
        pin_inputs=pins_top,
        step_size="20um",
        lead=dict(start_straight="100um", end_straight="1600um"),
    )

    RoutePathfinder(design, n.name_charge_line(group), options=options_chargeline)


# Function for meshing names for mapping for finer meshing
def CoupledLineTee_mesh_names(comp_names):
    all_names_to_mesh = [f"prime_cpw_{comp_names}", f"second_cpw_{comp_names}"]
    return all_names_to_mesh


# Function to render the design
def render_qiskit_metal_design(design, gui, capacitance=False):
    add_transmon_plus_resonator(design, group=n.NBR_1)
    add_transmon_plus_resonator(design, group=n.NBR_2)
    add_coupler(design)
    add_route_interconnects(design)
    add_launch_pads(design)
    add_chargeline(design, group=n.NBR_1)
    add_chargeline(design, group=n.NBR_2)

    if capacitance == True:
        for component in design.components.values():
            if "hfss_wire_bonds" in component.options:
                component.options["hfss_wire_bonds"] = False

    gui.rebuild()
    gui.autoscale()
