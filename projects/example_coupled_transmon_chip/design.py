import design_constants as c
import design_variable_names as u
from qiskit_metal.designs.design_planar import DesignPlanar
from qiskit_metal.qlibrary.couplers.coupled_line_tee import CoupledLineTee
from qiskit_metal.qlibrary.qubits.transmon_pocket_teeth import TransmonPocketTeeth
from qiskit_metal.qlibrary.terminations.launchpad_wb import LaunchpadWirebond
from qiskit_metal.qlibrary.tlines.meandered import RouteMeander
from qiskit_metal.qlibrary.tlines.pathfinder import RoutePathfinder


def add_transmon_plus_resonator(design: DesignPlanar, nbr: int):
    nbr_idx = nbr - 1  # zero indexed

    qubit = [c.QUBIT_1, c.QUBIT_2][nbr_idx]
    resonator = [c.RESONATOR_1, c.RESONATOR_2][nbr_idx]

    # make transmon
    transmon_options = dict(
        pos_x=["-1.75mm", "-1.75mm"][nbr_idx],
        pos_y=["-1.5mm", "1.5mm"][nbr_idx],
        orientation=["180", "0"][nbr_idx],
        pad_gap="100um",
        inductor_width="30um",
        pad_width=u.design_var_width(qubit),
        pad_height="120um",
        pocket_width="1600um",
        pocket_height="1200um",
        coupled_pad_width="45um",
        coupled_pad_height=u.design_var_coupl_length(resonator, qubit),
        coupled_pad_gap="100um",
        connection_pads=dict(
            readout=dict(
                loc_W=0,
                loc_H=+1,
                pad_gap="100um",
                pad_gap_w="0um",
                pad_width="40um",
                pad_height="170um",
                cpw_width=c.RESONATOR_WIDTH,
                cpw_gap=c.RESONATOR_GAP,
                cpw_extend="0.0um",
                pocket_extent="5um",
            ),
            coupler=dict(
                loc_W=0,
                loc_H=-1,
                pad_gap="100um",
                pad_gap_w="0um",
                pad_width="40um",
                pad_height="170um",
                cpw_width=c.RESONATOR_WIDTH,
                cpw_gap=c.RESONATOR_GAP,
                cpw_extend="0.0um",
                pocket_extent="5um",
            ),
            charge_line=dict(
                loc_W=[1, -1][nbr_idx],
                loc_H=-1,
                pad_gap=u.design_var_cl_pos_x(qubit),
                pad_gap_w=u.design_var_cl_pos_y(qubit),
                pad_width="40um",
                pad_height="170um",
                cpw_width=c.LINE_50_OHM_WIDTH,
                cpw_gap=c.LINE_50_OHM_GAP,
                cpw_extend="1500.0um",
                pocket_extent="5um",
                pocket_rise="0um",
                pad_cpw_shift="150um",
            ),
        ),
        gds_cell_name=f"Manhattan_{nbr}",
        hfss_inductance=u.design_var_lj(qubit),
        hfss_capacitance=u.design_var_cj(qubit),
    )

    qub = TransmonPocketTeeth(design, u.name_mode(qubit), options=transmon_options)

    # make open end of resonator
    cltee_options = dict(
        pos_x="0mm",
        pos_y=["-2.9mm", "2.9mm"][nbr_idx],
        orientation=["-90", "-90"][nbr_idx],
        second_width=c.RESONATOR_WIDTH,
        second_gap=c.RESONATOR_GAP,
        prime_width=c.LINE_50_OHM_WIDTH,
        prime_gap=c.LINE_50_OHM_GAP,
        coupling_space="10um",
        fillet=c.BEND_RADIUS,
        coupling_length=u.design_var_coupl_length(resonator, "tee"),
    )

    cltee = CoupledLineTee(design, u.name_tee(nbr), options=cltee_options)

    # make resonator
    resonator_options = dict(
        pin_inputs=dict(
            start_pin=dict(component=qub.name, pin="readout"),
            end_pin=dict(component=cltee.name, pin="second_end"),
        ),
        fillet=c.BEND_RADIUS,
        hfss_wire_bonds=True,
        total_length=u.design_var_length(resonator),
        lead=dict(start_straight="600um", end_straight="100um"),
        trace_width=c.RESONATOR_WIDTH,
        trace_gap=c.RESONATOR_GAP,
        meander=dict(spacing="200um"),
    )

    RouteMeander(design, u.name_mode(resonator), options=resonator_options)


def add_route_interconnects(design: DesignPlanar):

    pins = dict(
        start_pin=dict(component=u.name_tee(1), pin="prime_start"),
        end_pin=dict(component=u.name_tee(2), pin="prime_end"),
    )

    options_rpf = dict(
        fillet="49um",
        hfss_wire_bonds=False,
        trace_width=c.LINE_50_OHM_WIDTH,
        trace_gap=c.LINE_50_OHM_GAP,
        pin_inputs=pins,
    )
    RoutePathfinder(design, u.name_tee_to_tee(1, 2), options=options_rpf)


def add_launch_pads(design: DesignPlanar):

    launch_options = dict(
        chip="main",
        trace_width=c.LINE_50_OHM_WIDTH,
        trace_gap=c.LINE_50_OHM_GAP,
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
    LaunchpadWirebond(design, u.name_lp(0), options=launch_options)

    launch_options["pos_x"] = "0mm"
    launch_options["pos_y"] = "-4.5mm"
    launch_options["orientation"] = "90"
    LaunchpadWirebond(design, u.name_lp(1), options=launch_options)

    pins_top = dict(
        start_pin=dict(component=u.name_lp(0), pin="tie"),
        end_pin=dict(component=u.name_tee(2), pin="prime_start"),
    )

    options_top = dict(
        fillet="49um",
        hfss_wire_bonds=False,
        trace_width=c.LINE_50_OHM_WIDTH,
        trace_gap=c.LINE_50_OHM_GAP,
        pin_inputs=pins_top,
    )

    pins_bottom = dict(
        start_pin=dict(component=u.name_lp(1), pin="tie"),
        end_pin=dict(component=u.name_tee(1), pin="prime_end"),
    )

    options_bottom = dict(
        fillet="49um",
        hfss_wire_bonds=False,
        trace_width=c.LINE_50_OHM_WIDTH,
        trace_gap=c.LINE_50_OHM_GAP,
        pin_inputs=pins_bottom,
    )
    RoutePathfinder(design, u.name_lp_to_tee(0, 2), options=options_top)
    RoutePathfinder(design, u.name_lp_to_tee(1, 1), options=options_bottom)


def add_coupler(design: DesignPlanar):
    resonator_options = dict(
        pin_inputs=dict(
            start_pin=dict(component=u.name_mode(c.QUBIT_1), pin="coupler"),
            end_pin=dict(component=u.name_mode(c.QUBIT_2), pin="coupler"),
        ),
        fillet=c.BEND_RADIUS,
        hfss_wire_bonds=True,
        total_length=u.design_var_length(c.COUPLER_12),
        lead=dict(start_straight="200um", end_straight="200um"),
        trace_width=c.RESONATOR_WIDTH,
        trace_gap=c.RESONATOR_GAP,
        meander=dict(spacing="200um"),
    )

    RouteMeander(design, u.name_mode(c.COUPLER_12), options=resonator_options)


def add_chargeline(design: DesignPlanar, nbr: int):
    nbr_idx = nbr - 1
    qubit = [c.QUBIT_1, c.QUBIT_2][nbr_idx]
    lp_nbr = [2, 3][nbr_idx]

    launch_options = dict(
        chip="main",
        trace_width=c.LINE_50_OHM_WIDTH,
        trace_gap=c.LINE_50_OHM_GAP,
        lead_length="30um",
        pad_gap="125um",
        pad_width="260um",
        pad_height="260um",
        pos_x="-4.5mm",
        pos_y="-2mm",
        orientation="0",
    )
    launch_options["pos_y"] = ["-2mm", "2mm"][nbr_idx]
    LaunchpadWirebond(design, u.name_lp(lp_nbr), options=launch_options)

    pins_top = dict(
        start_pin=dict(component=u.name_lp(lp_nbr), pin="tie"),
        end_pin=dict(component=u.name_mode(qubit), pin="charge_line"),
    )

    options_chargeline = dict(
        fillet="50um",
        hfss_wire_bonds=False,
        trace_width=c.LINE_50_OHM_WIDTH,
        trace_gap=c.LINE_50_OHM_GAP,
        pin_inputs=pins_top,
        step_size="20um",
        lead=dict(start_straight="100um", end_straight="100um"),
    )

    RoutePathfinder(design, u.name_charge_line(nbr), options=options_chargeline)
