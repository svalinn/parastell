import numpy as np

# from matplotlib import pyplot as plt
# from matplotlib.ticker import FormatStrFormatter
from . import magnet_coils
from . import invessel_build as ivb
import pystell.read_vmec as read_vmec

import cubit


# function to get reordered filaments
def calc_z_radius(point):
    """
    Calculate the distance from the z axis.

    Arguments:
        point (iterable of [x,y,z] float): point to find distance for
    Returns:
        (float): distance to z axis.
    """
    return (point[0] ** 2 + point[1] ** 2) ** 0.5


def get_start_index(filament):
    """
    Find the index at which the filament crosses the xy plane on the OB
    side of the filament

    Arguments:
        filament (list of list of float): List of points defining the filament

    Returns:
        max_z_index (int): index at which the filament crosses the xy plane
    """
    max_z_index = None
    max_z_radius = None
    for index, point in enumerate(filament[0:-1]):
        next_point = filament[index + 1]
        if point[2] / next_point[2] < 0:
            z_radius = calc_z_radius(point)
            if max_z_radius is None:
                max_z_index = index
                max_z_radius = z_radius

            elif max_z_radius < z_radius:
                max_z_index = index
                max_z_radius = z_radius
    return max_z_index


def sort_filaments_toroidally(filaments):
    """
    Reorder filaments in order of increasing toroidal angle

    Arguments:
        filaments (list of list of list of float): List of filaments, which are
            lists of points defining each filament.
    Returns:
        filaments (list of list of list of float): filaments in order of
            increasing toroidal angle.
    """
    com_list = []

    for fil in filaments:
        com = np.average(fil, axis=0)
        com_list.append(com)

    com_list = np.array(com_list)
    phi_arr = np.arctan2(com_list[:, 1], com_list[:, 0])

    filaments = np.array([x for _, x in sorted(zip(phi_arr, filaments))])

    return filaments


def reorder_filaments(filaments):
    """
    Reorder the filaments so they start near the outboard xy plane crossing,
    and begin by increasing z value.

    Arguments:
        filaments (list of list of list of float): List of filaments, which are
            lists of points defining each filament.
    Returns:
        filaments (list of list of list of float): Reorderd list of filaments,
            suitable for building the magnet surface.
    """
    for filament_index, filament in enumerate(filaments):
        # start the filament at the outboard
        max_z_index = get_start_index(filament)
        reordered_filament = np.concatenate(
            [filament[max_z_index:], filament[0:max_z_index]]
        )

        # make sure z is increasing initially
        if filament[max_z_index, 2] > filament[max_z_index + 1, 2]:
            reordered_filament = np.flip(reordered_filament, axis=0)

        # remove duplicate point since the start point might be different
        _, idx = np.unique(reordered_filament, return_index=True, axis=0)
        reordered_filament = reordered_filament[np.sort(idx)]

        # ensure filament is a closed loop
        reordered_filament = np.concatenate(
            [reordered_filament, [reordered_filament[0]]]
        )

        filaments[filament_index] = reordered_filament

    filaments = sort_filaments_toroidally(filaments)

    return filaments


def get_reordered_filaments(magnet_set):
    """
    Convenience function to get the reordered filament data from a magnet
    """
    magnet_set._extract_filaments()
    magnet_set._set_average_radial_distance()
    magnet_set._set_filtered_filaments()

    filtered_filaments = magnet_set.filtered_filaments
    filaments = reorder_filaments(filtered_filaments)

    return filaments


def build_magnet_surface(reordered_filaments):
    for fil_index, _ in enumerate(reordered_filaments[0:-1]):
        fil1 = reordered_filaments[fil_index]
        fil2 = reordered_filaments[fil_index + 1]
        for index, _ in enumerate(fil1):
            x1 = fil1[index, 0]
            x2 = fil2[index, 0]
            y1 = fil1[index, 1]
            y2 = fil2[index, 1]
            z1 = fil1[index, 2]
            z2 = fil2[index, 2]
            cubit.cmd(
                f"create curve location {x1} {y1} {z1} location {x2} {y2} {z2}"
            )

    lines = np.array(cubit.get_entities("curve"))
    lines = np.reshape(
        lines, (len(reordered_filaments) - 1, len(reordered_filaments[0]))
    )
    for loop in lines:
        for line in loop[0:-1]:
            cubit.cmd(f"create surface skin curve {line} {line + 1}")


def measure_radial_distance(ribs):
    distances = []
    for rib in ribs:
        distance_subset = []
        for point, direction in zip(rib.rib_loci, rib._normals()):
            cubit.cmd(f"create vertex {point[0]} {point[1]} {point[2]}")
            vertex_id = max(cubit.get_entities("vertex"))
            cubit.cmd(
                f"create curve location at vertex {vertex_id} "
                f"location fire ray location at vertex {vertex_id} "
                f"direction {direction[0]} {direction[1]} {direction[2]} at "
                "surface all maximum hits 1"
            )
            curve_id = max(cubit.get_entities("curve"))
            distance = cubit.get_curve_length(curve_id)
            distance_subset.append(distance)
        distances.append(distance_subset)
    return np.array(distances)