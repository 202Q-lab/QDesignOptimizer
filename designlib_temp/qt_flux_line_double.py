# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
from qiskit_metal import Dict, draw
from qiskit_metal.qlibrary.core import QComponent

from designlib_temp.qt_component import QTComponent
from qdesignoptimizer.utils.utils import get_middle_point


class QTFluxLineDouble(QComponent, QTComponent):
    default_options = Dict(
        connector_cpw_ext="25um",
        flux_cpw_ext="25um",
        flux_cpw_width="12um",
        flux_cpw_gap="10um",
        flux_gap="3um",
        flux_ground_width="2um",
        flux_line_length="50um",
        flux_rounding="1um",
        flux_t_width="30um",
        flux_t_width_neck="10um",
        flux_taper_length="50um",
        flux_width="5um",
        squid_side="+1.",
        squid_pos_x="0um",
        trace_gap="10um",
        separation_of_flux_lines="10um",
        # trace_width = '20um',
        arc_angle="20",
        orientation="0",
        pos_x="0um",
        pos_y="0um",
    )

    def make(self):
        p = self.p

        def get_single_flux_line(mirror: bool):
            # self.make_elements(self.get_points())
            # Draw the flux line
            flux_y_bottom = p.squid_side * (  # p.trace_width / 2. +
                p.trace_gap
                # + p.flux_ground_width + p.flux_gap
            )
            flux_y_top = flux_y_bottom + p.squid_side * (
                p.flux_line_length + p.flux_width + p.flux_taper_length + p.flux_cpw_ext
            )

            before_bend_left = (
                p.squid_pos_x - 0.5 * p.flux_width,
                flux_y_bottom + p.squid_side * (p.flux_line_length + p.flux_width),
            )
            before_bend_right = (
                p.squid_pos_x + 0.5 * p.flux_width,
                flux_y_bottom + p.squid_side * (p.flux_line_length + p.flux_width),
            )

            radius_nbr = 10
            arc_radius_left = radius_nbr * p.flux_width
            arc_radius_right = (radius_nbr + 1) * p.flux_width
            arc_center = (before_bend_left[0] - arc_radius_left, before_bend_left[1])

            # # Generate points along the arc. Here we use np.linspace to create angles from 0 to 180 degrees (for a semicircle)
            # # and then calculate the x and y coordinates of the points along the arc
            # linspace include end point
            arc_angle_rad = np.radians(p.arc_angle)
            angles = np.linspace(
                0, arc_angle_rad, num=5, endpoint=True
            )  # Increase num for a smoother arc
            angles_left = np.flip(angles[1:])
            angles_right = angles[1:]
            arc_points_left = [
                (
                    arc_center[0] + arc_radius_left * np.cos(angle),
                    arc_center[1] - arc_radius_left * np.sin(angle),
                )
                for angle in angles_left
            ]
            arc_points_right = [
                (
                    arc_center[0] + arc_radius_right * np.cos(angle),
                    arc_center[1] - arc_radius_right * np.sin(angle),
                )
                for angle in angles_right
            ]
            width_diff = (p.flux_cpw_width - p.flux_width) / 2  # half at each side

            wide_left = (
                arc_points_left[0][0]
                - np.sin(arc_angle_rad) * p.flux_taper_length
                - np.cos(arc_angle_rad) * width_diff,
                arc_points_left[0][1]
                - np.cos(arc_angle_rad) * p.flux_taper_length
                + np.sin(arc_angle_rad) * width_diff,
            )
            wide_right = (
                arc_points_right[-1][0]
                - np.sin(arc_angle_rad) * p.flux_taper_length
                + np.cos(arc_angle_rad) * width_diff,
                arc_points_right[-1][1]
                - np.cos(arc_angle_rad) * p.flux_taper_length
                - np.sin(arc_angle_rad) * width_diff,
            )
            end_left = (
                wide_left[0] - np.sin(arc_angle_rad) * p.flux_cpw_ext,
                wide_left[1] - np.cos(arc_angle_rad) * p.flux_cpw_ext,
            )
            end_right = (
                wide_right[0] - np.sin(arc_angle_rad) * p.flux_cpw_ext,
                wide_right[1] - np.cos(arc_angle_rad) * p.flux_cpw_ext,
            )
            # Draw the taper and tip
            flux_line = draw.Polygon(
                [
                    (p.squid_pos_x + 0.5 * p.flux_width, flux_y_bottom),
                    before_bend_right,
                    *arc_points_right,
                    wide_right,
                    end_right,
                    end_left,
                    wide_left,
                    *arc_points_left,
                    before_bend_left,
                    (p.squid_pos_x - 0.5 * p.flux_width, flux_y_bottom),
                    # (p.squid_pos_x - 0.5 * p.flux_width - p.flux_gap,
                    #     flux_y_bottom + p.flux_width),
                    # (p.squid_pos_x - 0.5 * p.flux_width - p.flux_gap,
                    #     flux_y_bottom)
                ]
            )

            # Round the object
            flux_line = self.round_object(flux_line, p.flux_rounding)
            # ``Sharpen'' the connection to the CPW again
            flux_cpw_extension_top_half = draw.Polygon(
                [
                    (
                        np.mean([wide_left[0], end_left[0]]),
                        np.mean([wide_left[1], end_left[1]]),
                    ),
                    end_left,
                    end_right,
                    (
                        np.mean([wide_right[0], end_right[0]]),
                        np.mean([wide_right[1], end_right[1]]),
                    ),
                    # (p.squid_pos_x + 0.5 * p.flux_cpw_width,
                    #     flux_y_bottom + p.squid_side * (
                    #         p.flux_line_length + p.flux_width + p.flux_taper_length
                    #         + p.flux_cpw_ext)),
                    # (p.squid_pos_x + 0.5 * p.flux_cpw_width,
                    #     flux_y_bottom + p.squid_side * (
                    #         p.flux_line_length + p.flux_width + p.flux_taper_length
                    #         + 0.5 * p.flux_cpw_ext)),
                    # (p.squid_pos_x - 0.5 * p.flux_cpw_width,
                    #     flux_y_bottom + p.squid_side * (
                    #         p.flux_line_length + p.flux_width + p.flux_taper_length
                    #         + 0.5 * p.flux_cpw_ext)),
                    # (p.squid_pos_x - 0.5 * p.flux_cpw_width,
                    #     flux_y_bottom + p.squid_side * (
                    #         p.flux_line_length + p.flux_width + p.flux_taper_length
                    #         + p.flux_cpw_ext))
                ]
            )
            flux_tip = draw.Polygon(
                [
                    (
                        p.squid_pos_x
                        - 0.5 * p.flux_width
                        - p.flux_gap
                        - p.flux_rounding,
                        flux_y_bottom,
                    ),
                    (p.squid_pos_x, flux_y_bottom),
                    (p.squid_pos_x, flux_y_bottom + p.squid_side * p.flux_width),
                    (
                        p.squid_pos_x
                        - 0.5 * p.flux_width
                        - p.flux_gap
                        - p.flux_rounding,
                        flux_y_bottom + p.squid_side * p.flux_width,
                    ),
                ]
            )
            flux_line = draw.union(
                flux_line,
                flux_cpw_extension_top_half,
                flux_tip,
            )

            # Draw the flux line pocket
            arc_radius_left_pocket = arc_radius_left - p.flux_gap
            arc_radius_right_pocket = arc_radius_right + p.flux_gap
            arc_points_left_pocket = [
                (
                    arc_center[0] + arc_radius_left_pocket * np.cos(angle),
                    arc_center[1] - arc_radius_left_pocket * np.sin(angle),
                )
                for angle in angles_left
            ]
            arc_points_right_pocket = [
                (
                    arc_center[0] + arc_radius_right_pocket * np.cos(angle),
                    arc_center[1] - arc_radius_right_pocket * np.sin(angle),
                )
                for angle in angles_right
            ]
            sin_shift_pocket = np.sin(arc_angle_rad) * p.flux_cpw_gap
            cos_shift_pocket = np.cos(arc_angle_rad) * p.flux_cpw_gap
            wide_left_pocket = (
                wide_left[0] - cos_shift_pocket,
                wide_left[1] + sin_shift_pocket,
            )
            wide_right_pocket = (
                wide_right[0] + cos_shift_pocket,
                wide_right[1] - sin_shift_pocket,
            )

            sin_shift = np.sin(arc_angle_rad) * p.flux_cpw_gap
            cos_shift = np.cos(arc_angle_rad) * p.flux_cpw_gap
            end_left_pocket = (end_left[0] - cos_shift, end_left[1] + sin_shift)
            end_right_pocket = (end_right[0] + cos_shift, end_right[1] - sin_shift)

            flux_line_pocket = draw.Polygon(
                [
                    (p.squid_pos_x + 0.5 * p.flux_width + p.flux_gap, flux_y_bottom),
                    (
                        p.squid_pos_x + 0.5 * p.flux_width + p.flux_gap,
                        flux_y_bottom
                        + p.squid_side * (p.flux_line_length + p.flux_width),
                    ),
                    *arc_points_right_pocket,
                    wide_right_pocket,
                    end_right_pocket,
                    end_left_pocket,
                    wide_left_pocket,
                    *arc_points_left_pocket,
                    (
                        p.squid_pos_x - 0.5 * p.flux_width - p.flux_gap,
                        flux_y_bottom
                        + p.squid_side * (p.flux_line_length + p.flux_width),
                    ),
                    (p.squid_pos_x - 0.5 * p.flux_width - p.flux_gap, flux_y_bottom),
                ]
            )
            # end_left = (wide_left[0] - np.sin(arc_angle_rad) * p.flux_cpw_ext, #this should be another meas
            #                 wide_left[1] - np.cos(arc_angle_rad) * p.flux_cpw_ext)
            # end_right = (wide_right[0] - np.sin(arc_angle_rad) * p.flux_cpw_ext,
            #                 wide_right[1] - np.cos(arc_angle_rad) * p.flux_cpw_ext)

            flux_line_end_hard_corner = draw.Polygon(
                [
                    # from flux_cpw_extension_top_half
                    (
                        np.mean([wide_left[0], end_left[0]]) - cos_shift,
                        np.mean([wide_left[1], end_left[1]]) + sin_shift,
                    ),
                    end_left_pocket,
                    end_right_pocket,
                    (
                        np.mean([wide_right[0], end_right[0]]) + cos_shift,
                        np.mean([wide_right[1], end_right[1]]) - sin_shift,
                    ),
                ]
            )
            # Merge the flux line pocket and hard corner
            flux_line_pocket = draw.union(flux_line_pocket)
            # Draw the termination T to the flux line
            flux_line_termination_t = draw.Polygon(
                [
                    (
                        p.squid_pos_x - 0.5 * p.flux_t_width,
                        flux_y_bottom - p.squid_side * p.flux_gap,
                    ),
                    (
                        p.squid_pos_x + 0.5 * p.flux_t_width_neck,
                        flux_y_bottom - p.squid_side * p.flux_gap,
                    ),
                    (p.squid_pos_x + 0.5 * p.flux_t_width_neck, flux_y_bottom),
                    (p.squid_pos_x - 0.5 * p.flux_t_width, flux_y_bottom),
                ]
            )
            # Merge the flux line pocket and termination T
            flux_line_pocket = flux_line_pocket.union(flux_line_termination_t)
            # Round the flux line pocket
            flux_line_pocket = self.round_object(flux_line_pocket, p.flux_rounding)
            # Merge the hard corner flux line pocket
            flux_line_pocket = flux_line_pocket.union(flux_line_end_hard_corner)
            # # Sharpen the connection to the CPW again
            # flux_line_pocket_cpw_top = draw.Polygon([
            #     (p.squid_pos_x + 0.5 * p.flux_cpw_width + p.flux_cpw_gap,
            #         flux_y_top),
            #     (p.squid_pos_x + 0.5 * p.flux_cpw_width + p.flux_cpw_gap,
            #         flux_y_top - 0.5 * p.squid_side * p.flux_cpw_ext),
            #     (p.squid_pos_x - 0.5 * p.flux_cpw_width - p.flux_cpw_gap,
            #         flux_y_top - 0.5 * p.squid_side * p.flux_cpw_ext),
            #     (p.squid_pos_x - 0.5 * p.flux_cpw_width - p.flux_cpw_gap,
            #         flux_y_top),
            # ])
            # flux_line_pocket = flux_line_pocket.union(flux_line_pocket_cpw_top)
            # Rotate and translate all geometries

            # Generate the flux line pin to connect the CPW to
            middle_point = get_middle_point(wide_left, wide_right)
            flux_line_pin = draw.LineString(
                [
                    middle_point,
                    (
                        middle_point[0] - np.sin(arc_angle_rad) * p.flux_cpw_ext,
                        middle_point[1] - np.cos(arc_angle_rad) * p.flux_cpw_ext,
                    ),
                    # (p.squid_pos_x - p.squid_side * 0.5 * p.flux_cpw_width,
                    #     flux_y_top),
                    # (p.squid_pos_x + p.squid_side * 0.5 * p.flux_cpw_width,
                    #     flux_y_top)
                ]
            )

            c_items = [flux_line, flux_line_pin, flux_line_pocket]
            separation = -p.separation_of_flux_lines / 2
            c_items = draw.translate(c_items, separation, 0)

            def mirror_in_x_axis(shapes: list):
                return [
                    draw.shapely.ops.transform(lambda x, y: (-x, y), shape)
                    for shape in shapes
                ]

            if mirror:
                c_items = mirror_in_x_axis(c_items)

            c_items = draw.rotate(c_items, p.orientation, origin=(0, 0))
            c_items = draw.translate(c_items, p.pos_x, p.pos_y)
            [flux_line, flux_line_pin, flux_line_pocket] = c_items
            return flux_line, flux_line_pin, flux_line_pocket

        # Get the single flux line
        flux_line_left, flux_line_pin_left, flux_line_pocket_left = (
            get_single_flux_line(mirror=False)
        )
        flux_line_right, flux_line_pin_right, flux_line_pocket_right = (
            get_single_flux_line(mirror=True)
        )
        # Add all geometries to qgeometry
        self.add_qgeometry("poly", {"flux_line_left": flux_line_left})
        self.add_qgeometry("poly", {"flux_line_right": flux_line_right})
        self.add_qgeometry(
            "poly", {"flux_line_pocket_left": flux_line_pocket_left}, subtract=True
        )
        self.add_qgeometry(
            "poly", {"flux_line_pocket_right": flux_line_pocket_right}, subtract=True
        )
        # Add all pins to qgeometry
        self.add_pin(
            "flux_left", flux_line_pin_left.coords, p.flux_cpw_width, input_as_norm=True
        )
        self.add_pin(
            "flux_right",
            flux_line_pin_right.coords,
            p.flux_cpw_width,
            input_as_norm=True,
        )
        # # If the coupler has a readout resonator, draw its connector claw
        # if p.coupler_readout:
        #     self.make_connector()
