"""
Utility functions for geometry calculations and runtime operations in quantum circuit designs.

This module provides helper functions for common geometric calculations needed in
quantum circuit design and simulation, such as finding junction positions, calculating
midpoints, normalizing vectors, and performing rotations. It also includes utilities
for string parsing of values with units and process management.
"""

import glob
import io
import os
import re
import time
from typing import List, Tuple

import numpy as np

from qdesignoptimizer.logger import log


def _aedt_project_directories() -> List[str]:
    """Folders AEDT writes its projects, and therefore its recovery files, into.

    Read from AEDT's own user configuration rather than hard-coded: every
    install records
    ``<VALUE ObjectName="ProjectDirectory" Value="..."/>`` in
    ``~/Documents/Ansoft/ElectronicsDesktop<version>/config/*_user.XML``.

    Returns:
        Existing directories, newest config first. Falls back to AEDT's own
        default ``C:/Ansoft`` when no configuration can be read.
    """
    found: List[str] = []
    pattern = os.path.join(
        os.path.expanduser("~"), "Documents", "Ansoft",
        "ElectronicsDesktop*", "config", "*_user.XML",
    )
    for cfg in sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True):
        try:
            text = io.open(cfg, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for match in re.finditer(
            r'ObjectName="ProjectDirectory"\s+Value="([^"]+)"', text
        ):
            directory = match.group(1)
            if os.path.isdir(directory) and directory not in found:
                found.append(directory)
    if not found and os.path.isdir(r"C:/Ansoft"):
        found.append(r"C:/Ansoft")
    return found


def _release_pyaedt_desktops() -> int:
    """Shut down every AEDT session pyaedt opened, gracefully.

    A clean shutdown writes no ``.aedt.auto``/``.aedt.lock`` at all, so this
    is what keeps the next launch from blocking on a recovery dialog. Sessions
    pyaedt never owned are not visible here and are left to the taskkill below.

    Returns:
        Number of sessions released.
    """
    try:
        from ansys.aedt.core.desktop import _desktop_sessions
    except ImportError:
        return 0
    released = 0
    for session in list(_desktop_sessions.values()):
        try:
            session.release_desktop(close_projects=True, close_on_exit=True)
            released += 1
        except Exception:  # noqa: BLE001 - a dead session must not stop the rest
            pass
    return released


def close_ansys(sweep_recovery_files: bool = True) -> None:
    """Shut down Ansys Electronics Desktop.

    Releases any session pyaedt is holding first, which shuts AEDT down
    cleanly and leaves no recovery files, then force-kills whatever is still
    running (a GUI left open by hand, or an orphan from an earlier run).
    Recovery files from those killed sessions are swept afterwards, because
    AEDT blocks on a modal "found a recovery file" dialog on its next launch
    -- invisible to a headless caller, and surfacing minutes later as an
    opaque COM error from ``Analyze()``.

    Args:
        sweep_recovery_files: Delete stale ``.aedt.auto``/``.aedt.lock`` from
            the project folders AEDT itself reports. Set False on a shared
            machine if another user may have a session open that this process
            cannot see.

    Note:
        The force-kill step is Windows-only.
    """
    released = _release_pyaedt_desktops()
    if released:
        log.info("Released %d pyaedt-owned AEDT session(s) gracefully", released)

    os.system("taskkill /f /im ansysedt.exe")
    time.sleep(1)

    if not sweep_recovery_files:
        return
    for directory in _aedt_project_directories():
        for fname in os.listdir(directory):
            if fname.endswith((".aedt.auto", ".aedt.lock")):
                try:
                    os.remove(os.path.join(directory, fname))
                    log.info("Removed stale recovery file %s", fname)
                except OSError:
                    pass


def get_junction_position(design, qcomponent) -> Tuple[str, str]:
    """
    Calculate the position of a Josephson junction in a quantum component.

    Extracts the geometric coordinates of a Josephson junction from a quantum
    component's geometry tables. This is particularly useful for determining
    where flux lines should be placed in relation to the junction.

    Args:
        design: The Qiskit Metal design containing the component.
        qcomponent: The quantum component containing the junction.

    Returns:
        A tuple of (x, y) coordinates as strings with "mm" units,
        suitable for use in QComponent options.

    Note:
        Supports only components containing a single junction.

    Example:
        >>> x_pos, y_pos = get_junction_position(design, transmon)
        >>> print(x_pos, y_pos)
        '2.5mm' '3.7mm'
    """
    junction_table = design.qgeometry.tables["junction"]
    rect_jj_junction = junction_table.loc[
        junction_table["component"] == qcomponent.id, "geometry"
    ]
    assert len(rect_jj_junction) == 1, "Only supports a single junction per component"
    coords = list(rect_jj_junction.iloc[0].coords)
    x, y = coords[1]
    return f"{x}mm", f"{y}mm"


def get_middle_point(
    point1: Tuple[float, float], point2: Tuple[float, float]
) -> Tuple[float, float]:
    """
    Calculate the midpoint between two points in a 2D plane.

    Args:
        point1: The (x, y) coordinates of the first point.
        point2: The (x, y) coordinates of the second point.

    Returns:
        The (x, y) coordinates of the midpoint.

    Example:
        >>> get_middle_point((0, 0), (10, 20))
        (5.0, 10.0)
    """
    x1, y1 = point1
    x2, y2 = point2
    return (x1 + x2) / 2, (y1 + y2) / 2


def get_normalized_vector(
    point1: Tuple[float, float], point2: Tuple[float, float]
) -> Tuple[float, float]:
    """
    Calculate the normalized unit vector pointing from point1 to point2.

    Computes a vector of length 1 that points in the direction from
    the first point to the second point in a 2D plane.

    Args:
        point1: The (x, y) coordinates of the origin point.
        point2: The (x, y) coordinates of the target point.

    Returns:
        A tuple (dx, dy) representing the normalized vector components.

    Example:
        >>> get_normalized_vector((0, 0), (3, 4))
        (0.6, 0.8)
    """
    x1, y1 = point1
    x2, y2 = point2
    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx**2 + dy**2)
    return dx / length, dy / length


def rotate_point(
    point: np.ndarray, rotation_center: np.ndarray, angle_rad: float
) -> np.ndarray:
    """
    Rotate a point counterclockwise around a center point by a specified angle.

    Applies a 2D rotation transformation to a point around a specified center
    using standard rotation matrix operations.

    Args:
        point: A numpy array [x, y] representing the point to rotate.
        rotation_center: A numpy array [x, y] representing the center of rotation.
        angle_rad: The angle of rotation in radians (positive for counterclockwise).

    Returns:
        A numpy array with the coordinates of the rotated point.

    Example:
        >>> import numpy as np
        >>> rotate_point(np.array([1, 0]), np.array([0, 0]), np.pi/2)
        array([0., 1.])
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


def get_value_and_unit(val_unit: str | float | int) -> Tuple[float, str]:
    """
    Extract numerical value and unit from a string representation.

    Parses a string containing a number followed by an optional unit
    (e.g., "10.5mm") and separates it into the numerical value and unit string.

    Args:
        val_unit: A string representing a value with optional unit (e.g., "10.5mm").

    Returns:
        A tuple (value, unit) where value is a float and unit is a string.
        If no unit is present, the unit string will be empty..

    Example:
        >>> get_value_and_unit("10.5mm")
        (10.5, 'mm')
        >>> get_value_and_unit(42)
        (42.0, '')
    """
    try:
        if isinstance(val_unit, str) and str.isalpha(val_unit[-1]):
            idx = 1
            while idx < len(val_unit) and str.isalpha(val_unit[-idx - 1]):
                idx += 1

            unit = val_unit[-idx:]
            val = float(val_unit.replace(unit, ""))
        else:
            val = float(val_unit.strip(" ") if isinstance(val_unit, str) else val_unit)
            unit = ""
        return val, unit
    except Exception as exc:
        raise ValueError(f"Could not parse value and unit from {val_unit}") from exc


def sum_expression(vals: List[str]) -> str:
    """
    Calculate the sum of values with consistent units.

    Takes a list of strings representing values with units (e.g., ["10mm", "5mm"]),
    extracts the numerical values while preserving the unit, computes their sum,
    and returns the result with the same unit.

    Args:
        vals: A list of strings representing values with the same unit.

    Returns:
        A string representing the sum with the preserved unit.

    Raises:
        AssertionError: If the units of the provided values are not the same.

    Example:
        >>> sum_expression(["10mm", "5mm", "2.5mm"])
        '17.5mm'
        >>> sum_expression(["1.2GHz", "0.8GHz"])
        '2.0GHz'
    """
    sum_val = 0.0
    _, unit_0 = get_value_and_unit(vals[0])

    for val_str in vals:
        val, unit = get_value_and_unit(val_str)
        assert unit_0 == unit, f"Units must be the same: {unit_0} != {unit}"
        sum_val += val
        sum_unit = unit

    return f"{sum_val}{sum_unit}"


def get_save_path(out_folder: str, chip_name: str, time_format: str = "%Y%m%d-%H%M%S"):
    """Create a path to save simulation results by appending the start time of the simulation to the identifier name."""
    return out_folder + chip_name + "_" + time.strftime(time_format)
