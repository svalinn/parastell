import numpy as np
import cubit
import pystell.read_vmec as read_vmec

from . import magnet_coils
from . import invessel_build as ivb
from . import cubit_io


def calc_radius(point):
    """Calculates the cylindrical radius of a reference point.

    Arguments:
        point (iterable of float): Cartesian coordinates of reference
            point.

    Returns:
        (float): cylindrical radius.
    """
    return np.linalg.norm(point[0:2])


def get_start_index(coil):
    """Finds the index of the outboard midplane coordinate on a coil filament.

    Arguments:
        coil (object): MagnetCoil class object.

    Returns:
        outboard_index (int): index of the outboard midplane point.
    """
    radii = [calc_radius(point) for point in coil.coords]
    # Determine whether adjacent points cross the midplane
    midplane_flags = np.less(
        coil.coords[:, 2] / np.append(coil.coords[1:, 2], coil.coords[1, 2]),
        np.zeros(len(coil.coords)),
    )
    # Find index of outboard midplane point
    outboard_index = np.argmax(midplane_flags * radii)

    return outboard_index


def reorder_filament(coil):
    """Reorders the filament coordinates of a MagnetCoil class object such that
    they begin near the outboard midplane, and initially progress positively in
    the z-direction.

    Arguments:
        coil (object): MagnetCoil class object.

    Returns:
        reordered_coords (2-D iterable of float): reordered list of Cartesian
            coordinates defining a MagnetCoil filament.
    """
    # Start the filament at the outboard midplane
    outboard_index = get_start_index(coil)
    reordered_coords = np.concatenate(
        [coil.coords[outboard_index:], coil.coords[1:outboard_index]]
    )
    # Ensure filament is a closed loop
    if outboard_index != 0:
        reordered_coords = np.concatenate(
            [reordered_coords, [reordered_coords[0]]]
        )
    # Ensure points initially progress in positive z-direction
    if reordered_coords[0, 2] > reordered_coords[1, 2]:
        reordered_coords = np.flip(reordered_coords, axis=0)
    coil.coords = reordered_coords


def sort_coils_toroidally(magnet_coils):
    """Reorders list of coils by toroidal angle on range [-pi, pi] (coils
    ordered in MagnetSet class by toroidal angle on range [0, 2*pi]).

    Arguments:
        magnet_coils (list of object): list of MagnetCoil class objects.

    Returns:
        magnet_coils (list of object): sorted list of MagnetCoil class objects.
    """
    com_list = np.array([coil.center_of_mass for coil in magnet_coils])
    com_toroidal_angles = np.arctan2(com_list[:, 1], com_list[:, 0])

    magnet_coils = np.array(
        [x for _, x in sorted(zip(com_toroidal_angles, magnet_coils))]
    )

    return magnet_coils


def reorder_coils(magnet_set):
    """Reorders a set of magnetic coils toroidally and reorders their filament
    coordinates such that they begin near the outboard midplane, and initially
    progress positively in the z-direction.

    Arguments:
        magnet_set (object): MagnetSet class object.

    Returns:
        magnet_coils (list of object): reordered list of MagnetCoil class
            objects.
    """
    magnet_set.populate_magnet_coils()
    magnet_coils = magnet_set.magnet_coils

    [reorder_filament(coil) for coil in magnet_coils]
    magnet_coils = sort_coils_toroidally(magnet_coils)

    return magnet_coils


def build_line(point_1, point_2):
    """Builds a line between two points in Coreform Cubit.

    Arguments:
        point_1 (1-D iterable of float): Cartesian coordinates of first point.
        point_2 (1-D iterable of float): Cartesian coordinates of second point.

    Returns:
        curve_id (int): index of curve created in Coreform Cubit.
    """
    point_1 = " ".join(str(val) for val in point_1)
    point_2 = " ".join(str(val) for val in point_2)
    cubit.cmd(f"create curve location {point_1} location {point_2}")
    curve_id = cubit.get_last_id("curve")

    return curve_id


def build_magnet_surface(magnet_coils, sample_mod=1):
    """Builds a surface in Coreform Cubit spanning a list of coil filaments.

    Arguments:
        magnet_coils (list of object): list of MagnetCoil class objects,
            ordered toroidally. Each MagnetCoil object must also have its
            filament coordinates ordered poloidally (see reorder_coils
            function).
        sample_mod (int): sampling modifier for filament points (defaults to
            1). For a user-defined value n, every nth point will be sampled.
    """
    cubit_io.init_cubit()

    surface_lines = [
        [
            build_line(coord, next_coord)
            for coord, next_coord in zip(
                np.concatenate(
                    [coil.coords[:-1:sample_mod], [coil.coords[0]]]
                ),
                np.concatenate(
                    [next_coil.coords[:-1:sample_mod], [next_coil.coords[0]]]
                ),
            )
        ]
        for coil, next_coil in zip(magnet_coils[:-1], magnet_coils[1:])
    ]

    surface_lines = np.array(surface_lines)
    surface_sections = np.reshape(
        surface_lines,
        (
            len(magnet_coils) - 1,
            len(magnet_coils[0].coords[:-1:sample_mod]) + 1,
        ),
    )

    [
        [
            cubit.cmd(f"create surface skin curve {line} {next_line}")
            for line, next_line in zip(lines[:-1], lines[1:])
        ]
        for lines in surface_sections
    ]


def fire_ray(point, direction):
    """Performs a ray-firing operation in Coreform Cubit from a reference point
    and along a reference direction to determine the distance from that point
    to the magnet surface.

    Arguments:
        point (iterable of float): Cartesian coordinates of reference point.
        direction (iterable of float): reference direction in
            Cartesian-coordinate system.

    Returns:
        distance (float): distance between reference point and magnet surface,
            along reference direction.
    """
    cubit.cmd(f"create vertex {point[0]} {point[1]} {point[2]}")
    vertex_id = cubit.get_last_id("vertex")

    cubit.cmd(
        f"create curve location at vertex {vertex_id} "
        f"location fire ray location at vertex {vertex_id} "
        f"direction {direction[0]} {direction[1]} {direction[2]} at "
        "surface all maximum hits 1"
    )

    curve_id = cubit.get_last_id("curve")
    distance = cubit.get_curve_length(curve_id)

    return distance


def measure_surface_coils_separation(surface):
    """Determines the distance between a given Surface class object and a
    surface spanning a set of MagnetCoil class objects via a ray-firing
    operation in Coreform Cubit.

    Arguments:
        surface (object): Surface class object.

    Returns:
        distance_matrix (2-D np.array of float): matrix of distances between
            points defining surface.Ribs and magnet surface, along
            Rib._normals.
    """
    distance_matrix = [
        [
            fire_ray(point, distance)
            for point, distance in zip(rib.rib_loci, rib._normals())
        ]
        for rib in surface.Ribs
    ]

    return np.array(distance_matrix)


def measure_fw_coils_separation(
    vmec_file,
    toroidal_angles,
    poloidal_angles,
    wall_s,
    coils_file,
    width,
    thickness,
    sample_mod=1,
):
    """Measures the distance between a given first wall definition and a set of
    magnet filaments, at specified angular locations and along the profile
    normal at those angular locations, using ray-firing in Coreform Cubit.

    Arguments:
        vmec_file (str): path to plasma equilibrium VMEC file.
        toroidal_angles (array of float): toroidal angles at which distances
            should be computed [deg].
        poloidal_angles (array of float): poloidal angles at which distances
            should be computed [deg].
        wall_s (float): closed flux surface label extrapolation at wall.
        coils_file (str): path to coil filament data file.
        width (float): width of coil cross-section in toroidal direction [cm].
        thickness (float): thickness of coil cross-section in radial direction
            [cm].
        sample_mod (int): sampling modifier for filament points (defaults to
            1). For a user-defined value n, every nth point will be sampled.

    Returns:
        radial_distance_matrix (2-D np.array of float):
    """
    vmec = read_vmec.VMECData(vmec_file)
    radial_build_dict = {}

    radial_build = ivb.RadialBuild(
        toroidal_angles,
        poloidal_angles,
        wall_s,
        radial_build_dict,
        split_chamber=False,
    )
    invessel_build = ivb.InVesselBuild(
        vmec,
        radial_build,
        # Set num_ribs and num_rib_pts to be less than length of corresponding
        # array to ensure that only defined angular locations are used
        num_ribs=len(toroidal_angles) - 1,
        num_rib_pts=len(poloidal_angles) - 1,
    )
    invessel_build.populate_surfaces()
    invessel_build.calculate_loci()
    surface = invessel_build.Surfaces["chamber"]

    magnet_set = magnet_coils.MagnetSet(
        coils_file, width, thickness, toroidal_angles[-1] - toroidal_angles[0]
    )

    reordered_coils = reorder_coils(magnet_set)
    build_magnet_surface(reordered_coils, sample_mod=sample_mod)

    radial_distance_matrix = measure_surface_coils_separation(surface)

    return radial_distance_matrix
