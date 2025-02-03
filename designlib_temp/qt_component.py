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

from typing import Dict, List
import numpy as np

from src.utils.utils import get_value_and_unit


class QTComponent():
    """The QTComponent class collects all shared QT methods.

    Args:
        None
    
    Attributes:
        None
    """

    def central_coordinates(self):
        """Add pos_x, pos_y, and orientation to self.options."""
        p = self.p

        # Extract the (x,y) coordinates of the center of the coupler
        start_pin_points = self._get_connected_pin(
            self.options.pin_inputs.start_pin).points[-2::]
        start_pin_x = np.mean([start_pin_points[0][0], start_pin_points[1][0]])
        start_pin_y = np.mean([start_pin_points[0][1], start_pin_points[1][1]])
        end_pin_points = self._get_connected_pin(
            self.options.pin_inputs.end_pin).points[-2::]
        end_pin_x = np.mean([end_pin_points[0][0], end_pin_points[1][0]])
        end_pin_y = np.mean([end_pin_points[0][1], end_pin_points[1][1]])
        p.pos_x = np.mean([start_pin_x, end_pin_x])
        p.pos_y = np.mean([start_pin_y, end_pin_y])

        # Extract the orientation of the coupler 
        p.orientation = np.rad2deg(np.arctan(
            (end_pin_y - start_pin_y) / (end_pin_x - start_pin_x)))

        # Make the coordinates available in the QComponents dict
        self.options.pos_x = self.convert_to_um(p.pos_x)
        self.options.pos_y = self.convert_to_um(p.pos_y)
        self.options.orientation = p.orientation

    def convert_to_um(self, length_without_unit):
        """Take a unitless length and return its value in micron."""
        length_in_um = length_without_unit / self.parse_value("1um")
        return f"{length_in_um }um"

    def get_end_pin_center(self):
        """Add end_pin_x and end_pin_y to self.options."""
        end_pin_points = self._get_connected_pin(
            self.options.pin_inputs.end_pin).points[-2::]
        end_pin_x = np.mean([end_pin_points[0][0], end_pin_points[1][0]])
        end_pin_y = np.mean([end_pin_points[0][1], end_pin_points[1][1]])
        
        # Make the coordinates available in the QComponents dict
        self.options.end_pin_x = self.convert_to_um(end_pin_x)
        self.options.end_pin_y = self.convert_to_um(end_pin_y)

    def get_end_pin_orientation(self):
        """Add end_orientation to self.options."""
        end_pin_points = self._get_connected_pin(
            self.options.pin_inputs.end_pin).points[-2::]
        
        # Extract the orientation of the coupler 
        orientation = np.rad2deg(np.arctan(
            (end_pin_points[1][1] - end_pin_points[0][1]) /
            (end_pin_points[1][0] - end_pin_points[0][0])))

        # Make the coordinates available in the QComponents dict
        self.options.end_pin_orientation = orientation

    def get_start_pin_center(self):
        """Add start_pin_x and start_pin_y to self.options."""
        start_pin_points = self._get_connected_pin(
            self.options.pin_inputs.start_pin).points[-2::]
        start_pin_x = np.mean([start_pin_points[0][0], start_pin_points[1][0]])
        start_pin_y = np.mean([start_pin_points[0][1], start_pin_points[1][1]])
        
        # Make the coordinates available in the QComponents dict
        self.options.start_pin_x = self.convert_to_um(start_pin_x)
        self.options.start_pin_y = self.convert_to_um(start_pin_y)

    def get_start_pin_orientation(self):
        """Add start_orientation to self.options."""
        start_pin_points = self._get_connected_pin(
            self.options.pin_inputs.start_pin).points[-2::]
        
        # Extract the orientation of the coupler 
        orientation = np.rad2deg(np.arctan(
            (start_pin_points[1][1] - start_pin_points[0][1]) /
            (start_pin_points[1][0] - start_pin_points[0][0])))

        # Make the coordinates available in the QComponents dict
        self.options.start_pin_orientation = orientation

    def log_route(self):
        """Log the coordinates of routes to .txt files."""
        ptsfilename = (f"{self.design.gds_path}{self.design.chip_name}_"
            f"{self.p.route_type}.txt")
        # Add a comma and newline before the line if the file has already been
        # created by a previous operation
        printcomma = True
        try:
            ptsfile = open(ptsfilename,"r")
        except:
            printcomma = False
        else:
            ptsfile.close()
        ptsfile = open(ptsfilename,"a")
        if printcomma:
            ptsfile.write(",\n")
        # Write the coordinates of the CPW to the file
        ptsfile.write(str(self.get_points().tolist()))
        ptsfile.close()

    def round_object(self, input_polygon, rounding):
        """Take a shapely object, return a shapely object with corners rounded
        by 'rounding'. Requires rounding to be smaller than half the object
        width.
        """
        return input_polygon.buffer(rounding).buffer(-2 * rounding).buffer(
            rounding)
    
    def get_value_and_unit(self, val_unit: str) -> tuple:
        """Get the unitless numeric values."""
        return get_value_and_unit(val_unit)
    
    def get_connected_pin(self, pin_data: Dict):
        """Recovers a pin from the dictionary.

        Args:
            pin_data: dict {component: string, pin: string}

        Return:
            The actual pin object.
        """
        return self.design.components[pin_data.component].pins[pin_data.pin]
    
    @property
    def airbridge_coordinates(self):
        if not hasattr(self, '_airbridge_coordinates'):
            self._airbridge_coordinates = []
        return self._airbridge_coordinates

    def add_air_bridge_coordinates(self, airbridge_coordinates: List[list]):
        """Add the coordinates of the airbridge.

        Args:
            airbridge_coordinates: List of coordinates of the airbridge.
            Example: [ [x_start,y_start,z_start], [x_end,y_end,z_end] ]"""
        assert len(airbridge_coordinates) == 2, "The airbridge coordinates should have 1 start and 1 end point."
        assert len(airbridge_coordinates[0]) == 3, "The start point should have x,z,y coordinates."
        assert len(airbridge_coordinates[1]) == 3, "The end point should have x,z,y coordinates."
        if airbridge_coordinates not in self.airbridge_coordinates:
            self.airbridge_coordinates.append(airbridge_coordinates)
    
    def get_air_bridge_coordinates(self) -> List[List[list]]:
        """Get the coordinates of the airbridge.
        Example: 
        [ 
            [ [x1_start,y1_start,z1_start], [x1_end,y1_end,z1_end] ],
            [ [x1_start,y1_start,z1_start], [x1_end,y1_end,z1_end] ],
        ]"""
        return self.airbridge_coordinates
    