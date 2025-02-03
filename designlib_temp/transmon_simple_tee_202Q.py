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

"""
Designed by : Kazi Rafsanjani Amin
"""

import numpy as np
from qiskit_metal import draw, Dict
from qiskit_metal.qlibrary.core import BaseQubit


class TransmonTeemon202Q(BaseQubit):  # pylint: disable=invalid-name
    """The base `TransmonTeemon202Q` class.

    Inherits `BaseQubit` class.

    Simple Metal Transmon Teemon object. Creates the T-shaped island,
    the "junction" at the base of the T, and up to 2 connectors on the
    remaining arms (claw or gap).

    'claw_width' and 'claw_gap' define the width/gap of the CPW line
    that makes up the connector. Note, DC SQUID currently represented by
    single inductance sheet

    Add connectors to it using the `connection_pads` dictionary. See
    BaseQubit for more information.

    .. meta::
        Teemon

    BaseQubit Default Options:
        * connection_pads: Empty Dict -- The dictionary which contains
        all active connection lines for the qubit.
        * _default_connection_pads: empty Dict -- The default values for
        the (if any) connection lines of the qubit.

    Default Options:
        * tee_width: '20um' -- Width of the CPW center trace making up
        the top of the 'T' of the Teemon
        * tee_length: '200um' -- Length of one Teemon arm (from
        center)
        * tee_arm_length: '20um' -- Length of the CPW center trace making
        up the base of the 'T' of the Teemon
        * tee_width2: '20um' -- Width of the CPW center trace making up
        the base of the 'T' of the Teemon
        * tee_gap: '20um' -- Width of the CPW gap making up the Teemon
    """

    default_options = Dict(
        tee_width="20um",
        tee_length="200um",
        tee_arm_length="20um",
        tee_width2="5um",
        tee_gap="20um",
        junction_length='20um',
        chip="main",
    )
    """Default options."""

    component_metadata = Dict(
        short_name="Cross",
        _qgeometry_table_poly="True",
        _qgeometry_table_junction="True",
    )
    """Component metadata"""

    TOOLTIP = """Simple Tee shaped capacitor."""

    ##############################################MAKE#########################

    def make(self):
        """This is executed by the GUI/user to generate the qgeometry for the
        component."""
        self.make_pocket()

        if self.options.make_pin == True:
            self.make_pins()

    ###################################TRANSMON################################

    def make_pocket(self):
        """Makes a basic Tee shaped capacitor."""

        # self.p allows us to directly access parsed values (string -> numbers)
        # form the user option
        p = self.p

        # access to chip name
        chip = p.chip

        tee_arm = draw.box(
            -p.tee_length / 2, p.tee_width/2, p.tee_length / 2, -p.tee_width/2
        )
        tee_arm_etcher = draw.box(
            -p.tee_length / 2, p.tee_width/2 +p.tee_gap, p.tee_length / 2, -p.tee_width/2 -p.tee_gap
        )
        tee_bottom_arm = draw.box(
            -p.tee_width2 / 2,
            -p.tee_width/2,
            p.tee_width2 / 2,
            -p.tee_arm_length - p.tee_width/2,
        )

        tee_bottom_arm_etcher = draw.box(
            -(p.tee_width2+p.tee_gap) / 2,
            -p.tee_width/2,
            (p.tee_width2+p.tee_gap) / 2,
            -p.tee_width/2 - p.tee_arm_length - p.junction_length,
        )

        tee_capacitor = draw.shapely.ops.unary_union([tee_arm, tee_bottom_arm])
        tee_etcher = draw.shapely.ops.unary_union([tee_arm_etcher, tee_bottom_arm_etcher])
        
        # Creates the cross and the etch equivalent.
        rect_jj = draw.LineString(
            [
                (0, -p.tee_arm_length - p.tee_width/2),
                (0, -p.tee_arm_length - p.tee_width/2 - p.junction_length),
            ]
        )

        # rotate and translate
        polys = [tee_capacitor, tee_etcher, rect_jj]
        polys = draw.rotate(polys, p.orientation, origin=(0, 0))
        polys = draw.translate(polys, p.pos_x, p.pos_y)

        [tee_capacitor, tee_etcher, rect_jj] = polys

        # generate qgeometry
        self.add_qgeometry("poly", dict(cross=tee_capacitor), chip=chip)
        self.add_qgeometry(
            "poly", dict(tee_etcher=tee_etcher), subtract=True, chip=chip
        )
        self.add_qgeometry(
            "junction", dict(rect_jj=rect_jj), width=p.tee_width2, chip=chip
        )

    def make_pins(self):
        for name in self.options.connection_pins:
            self.make_single_pin(name)

    def make_single_pin(self, name):
        p = self.p
        tee_length = p.tee_length
        tee_gap = p.tee_gap

        pc = self.p.connection_pins[name]
        con_loc = pc.connector_location

        if con_loc == 180:
            con_rotation = 180
        elif con_loc == 0:
            con_rotation = 0 
        else:
            ValueError('Other pin locations have not been implemented. Choose 0 or 180 for west and east')

        port_line = draw.LineString([(0, -0.001),
                                     (0,0.001)])
       
        # Rotates and translates the connector polygons (and temporary port_line)
        polys = [port_line]
        polys = draw.translate(polys, -(tee_length)/2, 0)
        polys = draw.rotate(polys, con_rotation, origin=(0, 0))
        polys = draw.rotate(polys, p.orientation, origin=(0, 0))
        polys = draw.translate(polys, p.pos_x, p.pos_y)
        [port_line] = polys

        self.add_pin(name, port_line.coords, 0.002)