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

from qiskit_metal import draw, Dict
from qiskit_metal.qlibrary.core import QComponent
from designlib_temp.qt_component import QTComponent


class QTChargeLineOpenToGround(QComponent, QTComponent):
    """ Tapered CPW segment ending in an open to ground.


    Description:
        Creates a tapered CPW ending in an open to ground. Connect with CPW such as
        created by RoutePathfinder for a full charge line.
    ::

    X marks the location of end_pos_x and end_pos_y. It is the end of the tapered CPW, 
    with width cpw_end_width. The beginning is marked by pin 'prime_end'.
                ........................
                ---------------------\  \...........................
            ^   |                     \__<-end_line_length->_____   |
            |   |                      _________________________X|  |   
  cpw_width |   |                     /   ..........................| ^ taper_gap 
            v   -----<-cpw_ext->-----/  / :
       cpw_gap  ......................./  :
                                       :  :
                                    taper_length


    Default Options:
        * end_pos_x: '0mm' 
        * end_pos_y: '0mm'
        * cpw_width: '15um' -- Width of the main CPW center trace.
        * cpw_gap: '10um' -- Ground plane gap around main CPW until beginning of taper
        * fillet: '99um' -- Rounding of possible curves in RoutePathfinder object
        * hfss_wire_bonds_ True -- Add wire bounds for ground in HFSS simulation
        * orientation: '0' -- Direction of tapered end of the CPW
        * taper_gap: '5um' -- Ground plane gap around tapered end of the CPW
        * taper_length: '50um' -- Length of the taper itself
        * cpw_end_width: '5um' -- Width of the CPW at the end of taper
        * end_line_length: '100um' -- Extension of narrow CPW after tapering
        * end_rounding: '1um' -- Fillet of tapered end
        * cpw_ext: '1um' -- Extension of main CPW before taper. Needs to be > 0. 
    """
    default_options = Dict(chip = 'main', 
                            end_pos_x='0mm',
                            end_pos_y='0mm',
                            cpw_width='15um',
                            cpw_gap='10um',
                            fillet = '99um',
                            hfss_wire_bonds = True,
                            orientation = '0', 
                            taper_gap = '5um',
                            taper_length = '50um',
                            cpw_end_width = '5um',
                            end_line_length = '100um',
                            end_rounding = '1um',
                            cpw_ext='1um'
                            )
    
    component_metadata = Dict(short_name='taper', _qgeometry_table_poly='True')
    """Component metadata"""

    def make(self):
        p = self.p

       # Draw the tapered end

        y_top =  (p.end_line_length + p.cpw_end_width + p.taper_length
            + p.cpw_ext)
        # Draw the taper and tip
        tapered_line = draw.Polygon([
            ( 0.5 * p.cpw_end_width, 0),
            (0.5 * p.cpw_end_width,(
                    p.end_line_length + p.cpw_end_width)),
            (0.5 * p.cpw_width,
                y_top - 1 * p.cpw_ext),
            (0.5 * p.cpw_width,
                y_top),
            ( - 0.5 * p.cpw_width,
                y_top),
            (- 0.5 * p.cpw_width,
                y_top - 1 * p.cpw_ext),
            ( - 0.5 * p.cpw_end_width,
                 (
                    p.end_line_length + p.cpw_end_width)),
            ( - 0.5 * p.cpw_end_width,
                 p.cpw_end_width ),
            ( - 0.5 * p.cpw_end_width ,
                 p.cpw_end_width),
            ( - 0.5 * p.cpw_end_width , 0)
        ])
        # Round the object
        tapered_line = self.round_object(tapered_line, p.end_rounding)
        # ``Sharpen'' the connection to the CPW again
        cpw_extension_top_half = draw.Polygon([
            ( + 0.5 * p.cpw_width,
                 (
                    p.end_line_length + p.cpw_end_width + p.taper_length
                    + p.cpw_ext)),
            ( + 0.5 * p.cpw_width,
                 (
                    p.end_line_length + p.cpw_end_width + p.taper_length
                    + 0.5 * p.cpw_ext)),
            ( - 0.5 * p.cpw_width,
                 (
                    p.end_line_length + p.cpw_end_width + p.taper_length
                    + 0.5 * p.cpw_ext)),
            ( - 0.5 * p.cpw_width,
                 (
                    p.end_line_length + p.cpw_end_width + p.taper_length
                    + p.cpw_ext))
        ])
 
        tapered_line = draw.union(
            tapered_line, cpw_extension_top_half)
        # Generate the line pin to connect the CPW to
        tapered_line_pin = draw.LineString([
            (- 1 * 0.5 * p.cpw_width,
                y_top),
            ( + 1 * 0.5 * p.cpw_width,
                y_top)
        ])
        
        # Draw the line pocket
        tapered_line_pocket = draw.Polygon([
            ( 0.5 * p.cpw_end_width + p.taper_gap,
                - p.taper_gap),
            ( 0.5 * p.cpw_end_width + p.taper_gap,
                 (
                    p.end_line_length + p.cpw_end_width)),
            ( 0.5 * p.cpw_width + p.cpw_gap,
                y_top - 1 * p.cpw_ext),
            ( 0.5 * p.cpw_width + p.cpw_gap,
                y_top),
            ( - 0.5 * p.cpw_width - p.cpw_gap,
                y_top),
            ( - 0.5 * p.cpw_width - p.cpw_gap,
                y_top - 1 * p.cpw_ext),
            ( - 0.5 * p.cpw_end_width - p.taper_gap,
                 (
                    p.end_line_length + p.cpw_end_width - p.taper_gap)),
            ( - 0.5 * p.cpw_end_width - p.taper_gap,
                - p.taper_gap)
        ])

        # Round the line pocket
        tapered_line_pocket = self.round_object(tapered_line_pocket, p.end_rounding)
        # Sharpen the connection to the CPW again
        tapered_line_pocket_cpw_top = draw.Polygon([
            ( 0.5 * p.cpw_width + p.cpw_gap,
                y_top),
            ( 0.5 * p.cpw_width + p.cpw_gap,
                y_top - 0.5 * 1 * p.cpw_ext),
            ( - 0.5 * p.cpw_width - p.cpw_gap,
                y_top - 0.5 * 1 * p.cpw_ext),
            ( - 0.5 * p.cpw_width - p.cpw_gap,
                y_top),
        ])
        tapered_line_pocket = tapered_line_pocket.union(tapered_line_pocket_cpw_top)


        # Rotate and translate all geometries
        c_items = [tapered_line, tapered_line_pin, tapered_line_pocket]
        c_items = draw.rotate(c_items, p.orientation, origin=(0, 0))
        c_items = draw.translate(c_items, p.end_pos_x, p.end_pos_y)
        [tapered_line, tapered_line_pin, tapered_line_pocket] = c_items

        # Add all geometries to qgeometry
        self.add_qgeometry('poly',
                           {'tapered_line': tapered_line})
        self.add_qgeometry('poly',
                           {'tapered_line_pocket': tapered_line_pocket},
                           subtract = True)

        # Add all pins to qgeometry
        self.add_pin('prime_end', tapered_line_pin.coords, p.cpw_width)
