import os


def close_ansys():
    os.system("taskkill /f /im ansysedt.exe")


def get_junction_position(design, qcomponent):
    """Calculates the position of the Josephson junction in the component where a flux line could be placed.

    Args:
        design (QDesign): The Qiskit metal design.
        qcomponent (QComponent): The component to calculate the junction position for.

    Returns:
        tuple: The x and y coordinates of the junction as strings ending with "mm", to be used as qcomponents options.

    Raises:
        AssertionError: If the component does not have a junction.
        AssertionError: If the component has more than one junction.
    """

    junction_table = design.qgeometry.tables["junction"]
    rect_jj_junction = junction_table.loc[
        junction_table["component"] == qcomponent.id, "geometry"
    ]
    assert len(rect_jj_junction) == 1, "Only supports a single junction per component"
    coords = list(rect_jj_junction.iloc[0].coords)
    x, y = coords[1]

    return f"{x}mm", f"{y}mm"


def get_middle_point(point1, point2):
    """Calculates the middle point between two points.

    Args:
        point1 (tuple): The first point.
        point2 (tuple): The second point.

    Returns:
        tuple: The middle point.
    """
    x1, y1 = point1
    x2, y2 = point2
    return (x1 + x2) / 2, (y1 + y2) / 2


def get_normalized_vector(point1, point2):
    """Calculates the normalized vector between two points.

    Args:
        point1 (tuple): The first point.
        point2 (tuple): The second point.

    Returns:
        tuple: The normalized vector.
    """
    x1, y1 = point1
    x2, y2 = point2
    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx**2 + dy**2)
    return dx / length, dy / length


def rotate_point(point, rotation_center, angle_rad):
    """
    Rotate a point counterclockwise by a given angle around a given center.

    The angle should be given in radians.

    Args:
        point (np.array): The point to rotate.
        rotation_center (np.array): The center of rotation.
        angle_rad (float): The angle of rotation in radians.
    Returns:
        np.array: The rotated point.
    """
    # Translate the point so that the rotation center is at the origin
    point_translated = point - rotation_center

    # Perform the rotation
    rotation_matrix = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ]
    )
    rotated_point_translated = np.dot(rotation_matrix, point_translated)

    # Translate back
    rotated_point = rotated_point_translated + rotation_center

    return rotated_point


def get_value_and_unit(val_unit: str) -> tuple:
    """Get the value and unit from string."""
    try:
        if str.isalpha(val_unit[-1]):
            idx = 1
            while str.isalpha(val_unit[-idx - 1]):
                idx += 1

            unit = val_unit[-idx:]
            val = float(val_unit.replace(unit, ""))
        else:
            val = float(val_unit)
            unit = ""
        return val, unit
    except:
        raise ValueError(f"Could not parse value and unit from {val_unit}")


def sum_expression(vals: list):
    """Sum a list of values and units.

    Args:
        vals (list): A list of values and units.

    Returns:
        str: The sum of the values with unit.
    """
    sum_val = 0
    _, unit_0 = get_value_and_unit(vals[0])
    for val in vals:
        val, unit = get_value_and_unit(val)
        assert unit_0 == unit, "Units must be the same"
        sum_val += val
        sum_unit = unit

    return f"{sum_val}{sum_unit}"
