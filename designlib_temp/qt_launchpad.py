# -*- coding: utf-8 -*-

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

from qiskit_metal import draw
from designlib_temp.qt_component import QTComponent
from qiskit_metal.toolbox_python.attr_dict import Dict
from qiskit_metal.qlibrary.core import QComponent
import numpy as np

# Define class and options for the launch geometry


class QTLaunchpadWirebond(QComponent, QTComponent):
    """Based on class QTLaunchpadWirebond.

    Inherits classes QComponent, QTComponent.

    .. image::
        QTLaunchpadWirebond.png

    .. meta::
        QT Launchpad Wirebond

    Creates a 50 ohm launch pad with a ground pocket cutout.
    Limited but expandable parameters to control the launchpad polygons.
    The (0,0) point is the center of the necking of the launch tip.
    The pin attaches directly to the built-in lead length at its midpoint

    Pocket and pad:
        Pocket and launch pad geometries are currently fixed.
        (0,0) point is the midpoint of the necking of the launch tip.
        Pocket is a negative shape that is cut out of the ground plane

    Values (unless noted) are strings with units included, (e.g., '30um')

    Sketch:
        Below is a sketch of the launch
        ::

            -----------
            |          \
            |      ---------\\
            |      |    0    |    (0,0) pin at midpoint of necking, before 
            |      ---------//    the lead
            |          /
            -----------

            y
            ^
            |
            |------> x

    .. image::
        LaunchpadWirebond.png

    Default Options:
        * p.trace_width: 'cpw_width' -- Width of the transmission line attached to the launch pad
        * p.trace_gap: 'cpw_gap' -- Gap of the transmission line
        * p.lead_length: '25um' -- Length of the transmission line attached to the launch pad
        * pad_width: '80um' -- Width of the launch pad
        * p.pad_height: '80um' -- Height of the launch pad
        * p.pad_gap: '58um' -- Gap of the launch pad
        * p.taper_height: '122um' -- Height of the taper from the launch pad to the transmission line
    """

    default_options = Dict(trace_width = 'cpw_width',
                           trace_gap = 'cpw_gap',
                           lead_length = '25um',
                           pad_width = '80um',
                           pad_height = '80um',
                           pad_gap = '58um',
                           rounding = '50um',
                           taper_height = '122um')
    """Default options"""

    TOOLTIP = """Launch pad to feed/read signals to/from the chip."""

    def make(self):
        """Build the component."""

        p = self.p
        trace_width_half = p.trace_width / 2.
        pad_width_half = p.pad_width / 2.

        # Draw the geometry of the launch pad
        launch_pad = draw.Polygon([
            (0, trace_width_half),
            (-p.taper_height, pad_width_half),
            (-(p.pad_height + p.taper_height), pad_width_half),
            (-(p.pad_height + p.taper_height), -pad_width_half),
            (-p.taper_height, -pad_width_half),
            (0, -trace_width_half),
            (p.lead_length, -trace_width_half),
            (p.lead_length, trace_width_half),
            (0, trace_width_half)
        ])
        # Calculate the amount by which the straight part of the diagonal taper
        # is reduced by the rounding. This is required for the subsequent
        # re-sharpening of the launch pad tip.
        taper_rounding_angle = 0.5 * np.arctan(
            (pad_width_half - trace_width_half) / p.taper_height)
        taper_rounding_distance = p.rounding * np.tan(taper_rounding_angle)
        taper_diagonal = np.sqrt(
            (pad_width_half - trace_width_half) ** 2 + p.taper_height ** 2)
        taper_fraction_rounded = taper_rounding_distance / taper_diagonal
        # Draw a launch pad tip with rounded "outer corners" (angles >pi,
        # measured from the inside of the geometry), but sharp "inner corners".
        launch_pad_tip = draw.Polygon([
            (0, trace_width_half),
            (-(1 - taper_fraction_rounded) * p.taper_height,
                taper_fraction_rounded * trace_width_half
                + (1 - taper_fraction_rounded) * pad_width_half),
            (-(1 - taper_fraction_rounded) * p.taper_height,
                -taper_fraction_rounded * trace_width_half
                - (1 - taper_fraction_rounded) * pad_width_half),
            (0, -trace_width_half),
            (p.lead_length, -trace_width_half),
            (p.lead_length, trace_width_half),
            (0, trace_width_half)
        ]).buffer(p.rounding).buffer(-p.rounding)

        # Draw the launch pad pocket
        pocket = draw.Polygon([
            (0, trace_width_half + p.trace_gap),
            (-p.taper_height, pad_width_half + p.pad_gap),
            (-(p.pad_height + p.taper_height + p.pad_gap),
                pad_width_half + p.pad_gap),
            (-(p.pad_height + p.taper_height + p.pad_gap),
                -(pad_width_half + p.pad_gap)),
            (-p.taper_height, -(pad_width_half + p.pad_gap)),
            (0, -(trace_width_half + p.trace_gap)),
            (p.lead_length, -(trace_width_half + p.trace_gap)),
            (p.lead_length, trace_width_half + p.trace_gap),
            (0, trace_width_half + p.trace_gap)
        ])
        # Calculate the amount by which the straight part of the diagonal taper
        # pocket is reduced by the rounding. This is required for the
        # subsequent re-sharpening of the launch pad pocket.
        taper_rounding_angle = 0.5 * np.arctan(
            (pad_width_half + p.pad_gap - trace_width_half - p.trace_gap) /
                p.taper_height)
        taper_rounding_distance = p.rounding * np.tan(taper_rounding_angle)
        taper_diagonal = np.sqrt(
            (pad_width_half + p.pad_gap - trace_width_half - p.trace_gap) ** 2
                + p.taper_height ** 2)
        taper_fraction_rounded = taper_rounding_distance / taper_diagonal
        # Draw a pocket tip with rounded "outer corners" (angles >pi,
        # measured from the inside of the geometry), but sharp "inner corners".
        pocket_tip = draw.Polygon([
            (0, trace_width_half + p.trace_gap),
            (- (1 - taper_fraction_rounded) * p.taper_height,
                (1 - taper_fraction_rounded) * (pad_width_half + p.pad_gap)
                + taper_fraction_rounded 
                    * (trace_width_half + p.trace_gap)),
            (- (1 - taper_fraction_rounded) * p.taper_height,
                - (1 - taper_fraction_rounded) * (pad_width_half + p.pad_gap)
                - taper_fraction_rounded
                    * (trace_width_half + p.trace_gap)),
            (0, -(trace_width_half + p.trace_gap)),
            (p.lead_length, -(trace_width_half + p.trace_gap)),
            (p.lead_length, trace_width_half + p.trace_gap),
            (0, trace_width_half + p.trace_gap)
        ]).buffer(p.rounding).buffer(-p.rounding)

        # Define the pin locations
        main_pin_line = draw.LineString([(p.lead_length, trace_width_half),
                                         (p.lead_length, -trace_width_half)])

        # Round the launchpad and pocket, then add the sharp tips again
        launch_pad = self.round_object(launch_pad, p.rounding).union(
            launch_pad_tip)
        pocket = self.round_object(pocket, p.rounding).union(
            pocket_tip)

        # Rotate and translate all the objects
        polys1 = [main_pin_line, launch_pad, pocket]
        polys1 = draw.rotate(polys1, p.orientation, origin=(0, 0))
        polys1 = draw.translate(polys1, xoff=p.pos_x, yoff=p.pos_y)
        [main_pin_line, launch_pad, pocket] = polys1

        # Add the geometries to the qgeometry table
        self.add_qgeometry('poly', dict(launch_pad=launch_pad), layer=p.layer)
        self.add_qgeometry('poly',
                           dict(pocket=pocket),
                           subtract=True,
                           layer=p.layer)

        # Generate the pins
        self.add_pin('tie', main_pin_line.coords, p.trace_width)