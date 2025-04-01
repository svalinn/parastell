import math
from pathlib import Path
import concurrent.futures
import os

import openmc
from scipy.optimize import direct
import numpy as np
import h5py
from pystell import read_vmec
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt

from . import log
from .utils import m2cm


def fire_rays(dagmc_geom, source_mesh, toroidal_extent, strengths, num_parts):
    """Fires rays from sampled neutron source mesh to first wall geometry via
    OpenMC. The first wall should be tagged as a vacuum boundary during the
    creation of the DAGMC geometry to avoid multiple surface crossings from
    single histories.

    Arguments:
        dagmc_geom (str): path to DAGMC geometry file.
        source_mesh (str): path to source mesh file.
        toroidal_extent (float): toroidal extent of model [deg].
        strengths (array): source strengths for all tetrahedra [1/s].
        num_parts (int): number of source particles to simulate.
    """
    toroidal_extent = np.deg2rad(toroidal_extent)

    model = openmc.model.Model()

    dag_univ = openmc.DAGMCUniverse(dagmc_geom, auto_geom_ids=False)

    # Define problem boundaries
    # Assign large IDs to avoid ID overlaps
    per_init = openmc.YPlane(boundary_type="periodic", surface_id=9990)
    per_fin = openmc.Plane(
        a=np.sin(toroidal_extent),
        b=-np.cos(toroidal_extent),
        c=0,
        d=0,
        boundary_type="periodic",
        surface_id=9991,
    )
    # A small fraction (<0.005%) of particles tend to escape first wall vacuum
    # boundary.
    # Include additional vacuum boundary to avoid lost particles. Note that
    # this is effectively a cosmetic fix as the particles still escape the FW.
    vacuum_surface = openmc.Sphere(
        r=10_000, surface_id=9992, boundary_type="vacuum"
    )

    region = -vacuum_surface & +per_init & +per_fin
    period = openmc.Cell(cell_id=9999, region=region, fill=dag_univ)
    geometry = openmc.Geometry([period])
    model.geometry = geometry

    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.particles = num_parts
    settings.batches = 1

    mesh = openmc.UnstructuredMesh(source_mesh, "moab")
    src = openmc.IndependentSource()
    src.space = openmc.stats.MeshSpatial(
        mesh, strengths=strengths, volume_normalized=False
    )
    src.angle = openmc.stats.Isotropic()
    src.energy = openmc.stats.Discrete([14.1e6], [1.0])
    settings.source = [src]

    # Track surface crossings
    settings.surf_source_write = {
        "surface_ids": [1],
        "max_particles": num_parts * 2,
    }

    model.settings = settings

    model.run()

    return "surface_source.h5"


def compute_residual(poloidal_guess, vmec_obj, wall_s, toroidal_angle, point):
    """Minimization problem to solve for the poloidal angle.

    Arguments:
        poloidal_guess (float): poloidal angle guess [rad].
        vmec_obj (object): plasma equilibrium VMEC object.
        wall_s (float): closed flux surface label extrapolation at wall.
        toroidal_angle (float): toroidal angle [rad].
        point (numpy.array): Cartesian coordinates [cm].

    Returns:
        (float): L2 norm of difference between coordinates of interest and
            computed point [cm].
    """
    fw_guess = (
        np.array(vmec_obj.vmec2xyz(wall_s, poloidal_guess, toroidal_angle))
        * m2cm
    )

    return np.linalg.norm(point - fw_guess)


def solve_poloidal_angles(data):
    """Solves for poloidal angle of plasma equilibrium corresponding to
    specified coordinates. Takes a single argument so it works with
    ProcessPoolExecutor.

    Arguments:
        data (iterable): data for root-finding algorithm. Entries in order:
            1) path to plasma equilibrium VMEC file (str). Because the VMEC
               dataset is not picklable, a separate object is created for each
               thread.
            2) first wall CFS reference value (float).
            3) convergence tolerance for root-finding (float).
            4) toroidal angles at which to solve for poloidal angles
               (iterable). Must be in same order as Cartesian coordinates.
            5) Cartesian coordinates at which to solve for poloidal angles
               (iterable). Must be in same order as toroidal angles.

    Returns:
        poloidal_angles (list): poloidal angles corresponding to supplied
            coordinates [rad].
    """
    vmec_obj = read_vmec.VMECData(data[0])
    wall_s = data[1]
    conv_tol = data[2]
    toroidal_angles = data[3]
    coords = data[4]

    poloidal_angles = []

    for toroidal_angle, point in zip(toroidal_angles, coords):
        result = direct(
            compute_residual,
            bounds=[(0, 2 * np.pi)],
            args=(vmec_obj, wall_s, toroidal_angle, point),
            vol_tol=conv_tol,
        )
        poloidal_angles.append(result.x[0])

    return poloidal_angles


def compute_flux_coordinates(vmec_file, wall_s, coords, num_threads, conv_tol):
    """Computes flux coordinates of specified Cartesian coordinates.

    Arguments:
        vmec_file (str): path to plasma equilibrium VMEC file.
        wall_s (float): closed flux surface extrapolation at first wall.
        coords (numpy.array): Cartesian coordinates of surface crossings [cm].
        conv_tol (float): convergence tolerance for root-finding.

    Returns:
        toroidal_angles (list): toroidal angles of surface crossings [rad].
        poloidal_angles (list): poloidal angles of surface crossings [rad].
    """
    toroidal_angles = np.arctan2(coords[:, 1], coords[:, 0])
    chunk_size = math.ceil(len(toroidal_angles) / num_threads)

    chunks = []

    for i in range(num_threads):
        chunks.append(
            (
                vmec_file,
                wall_s,
                conv_tol,
                toroidal_angles[i * chunk_size : (i + 1) * chunk_size],
                coords[i * chunk_size : (i + 1) * chunk_size],
            )
        )

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_threads
    ) as executor:
        poloidal_chunks = list(executor.map(solve_poloidal_angles, chunks))

        poloidal_angles = [
            poloidal_angle
            for chunk in poloidal_chunks
            for poloidal_angle in chunk
        ]

    return toroidal_angles.tolist(), poloidal_angles


def extract_surface_crossings(source_file):
    """Extracts Cartesian coordinates of particle surface crossings given an
    OpenMC surface source file.

    Arguments:
        source_file (str): path to OpenMC surface source file.

    Returns:
        coords (np.array): Cartesian coordinates of all particle surface
            crossings.
    """
    file = h5py.File(source_file, "r")
    dataset = file["source_bank"]["r"]

    coords = np.empty((len(dataset), 3))
    coords[:, 0] = dataset["x"]
    coords[:, 1] = dataset["y"]
    coords[:, 2] = dataset["z"]

    return coords


def compute_quadrilateral_area(corners):
    """Approximates the area of a non-planar region bounded by four points.

    Arguments:
        corners (numpy.array): Cartesian coordinates of four points. Connecting
            the points in the order given should result in a polygon.

    Returns:
        area (float): approximation of area.
    """
    # Triangle 1
    edge_1 = corners[3] - corners[0]
    edge_2 = corners[2] - corners[0]
    normal_vector = np.cross(edge_1, edge_2)
    area_1 = np.linalg.norm(normal_vector) / 2

    # Triangle 2
    edge_1 = corners[1] - corners[0]
    edge_2 = corners[2] - corners[0]
    normal_vector = np.cross(edge_1, edge_2)
    area_2 = np.linalg.norm(normal_vector) / 2

    total_area = area_1 + area_2

    return total_area


def compute_nwl(
    source_file,
    vmec_file,
    wall_s,
    toroidal_extent,
    neutron_power,
    num_toroidal_bins=101,
    num_poloidal_bins=101,
    conv_tol=1e-6,
    num_batches=1,
    num_crossings=None,
    num_threads=None,
    logger=None,
):
    """Computes and plots NWL. Assumes toroidal extent is less than 360 degrees.

    Arguments:
        source_file (str): path to OpenMC surface source file.
        vmec_file (str): path to plasma equilibrium VMEC file.
        wall_s (float): closed flux surface label extrapolation at wall.
        toroidal_extent (float): toroidal extent of model [deg].
        neutron_power (float): reference neutron power [MW].
        num_toroidal_bins (int): number of toroidal angle bins (defaults to 101).
        num_poloidal_bins (int): number of poloidal angle bins (defaults to 101).
        conv_tol (float): tolerence for convergence in poloidal angle root-
            finding routine. Must lie in range (0.0, 1.0]. Smaller values
            correspond to a stricter tolerance. This parameter corresponds to
            the hyperrectangle volume tolerance defining the termination
            criterion for SciPy's DIRECT algorithm. Once the volume of the
            hyperrectangle containing the lowest function value (root-finding
            residual) falls below this tolerance, root-finding will terminate.
        num_batches (int): number of batches across which crossing coordinates
            will be solved (defaults to 1). Helps alleviate memory burden.
        num_crossings (int): number of crossings to use from the surface source
            (defaults to None). If None, all crossings will be used.
        num_threads (int): number of threads to use for parallelizing
            coordinate-solving routine (defaults to None). If None, the maximum
            number of threads will be used.

    Returns:
        nwl_mat (numpy array): array used to create the NWL plot
        toroidal_centroids (numpy array): phi axis of NWL plot
        poloidal_centroids (numpy array): theta axis of NWL plot
        area_mat (numpy array): area array used to normalize nwl_mat
    """
    logger = log.check_init(logger)

    toroidal_extent = np.deg2rad(toroidal_extent)
    poloidal_extent = 2 * np.pi

    if not num_threads:
        num_threads = os.cpu_count()

    coords = extract_surface_crossings(source_file)
    if num_crossings is not None:
        coords = coords[0:num_crossings]

    toroidal_angles = []
    poloidal_angles = []

    batch_size = math.ceil(len(coords) / num_batches)

    for i in range(num_batches):
        logger.info(f"Processing batch {i + 1}")

        toroidal_angle_batch, poloidal_angle_batch = compute_flux_coordinates(
            vmec_file,
            wall_s,
            coords[i * batch_size : (i + 1) * batch_size],
            num_threads,
            conv_tol,
        )
        toroidal_angles += toroidal_angle_batch
        poloidal_angles += poloidal_angle_batch

    # Define minimum and maximum bin edges for each dimension
    toroidal_bin_min = 0.0 - toroidal_extent / num_toroidal_bins / 2
    toroidal_bin_max = (
        toroidal_extent + toroidal_extent / num_toroidal_bins / 2
    )

    poloidal_bin_min = 0.0 - poloidal_extent / num_poloidal_bins / 2
    poloidal_bin_max = (
        poloidal_extent + poloidal_extent / num_poloidal_bins / 2
    )

    # Bin particle crossings
    count_mat, toroidal_bin_edges, poloidal_bin_edges = np.histogram2d(
        toroidal_angles,
        poloidal_angles,
        bins=[num_toroidal_bins, num_poloidal_bins],
        range=[
            [toroidal_bin_min, toroidal_bin_max],
            [poloidal_bin_min, poloidal_bin_max],
        ],
    )

    # Adjust endpoints to reflect geometry
    toroidal_bin_edges[0] = 0.0
    toroidal_bin_edges[-1] = toroidal_extent
    poloidal_bin_edges[0] = 0.0
    poloidal_bin_edges[-1] = poloidal_extent

    # Compute centroids of bins
    toroidal_centroids = np.linspace(
        0.0, toroidal_extent, num=num_toroidal_bins
    )
    poloidal_centroids = np.linspace(
        0.0, poloidal_extent, num=num_poloidal_bins
    )

    num_particles = len(coords)
    nwl_mat = count_mat * neutron_power / num_particles

    # Construct matrix of bin boundaries
    vmec_obj = read_vmec.VMECData(vmec_file)
    bin_mat = np.zeros((num_toroidal_bins + 1, num_poloidal_bins + 1, 3))
    for toroidal_id, toroidal_edge in enumerate(toroidal_bin_edges):
        for poloidal_id, poloidal_edge in enumerate(poloidal_bin_edges):
            x, y, z = vmec_obj.vmec2xyz(wall_s, poloidal_edge, toroidal_edge)
            bin_mat[toroidal_id, poloidal_id, :] = [x, y, z]

    # Construct matrix of bin areas
    area_mat = np.zeros((num_toroidal_bins, num_poloidal_bins))
    for toroidal_id in range(num_toroidal_bins):
        for poloidal_id in range(num_poloidal_bins):
            # Each bin has 4 corners
            corner_1 = bin_mat[toroidal_id, poloidal_id]
            corner_2 = bin_mat[toroidal_id, poloidal_id + 1]
            corner_3 = bin_mat[toroidal_id + 1, poloidal_id + 1]
            corner_4 = bin_mat[toroidal_id + 1, poloidal_id]
            corners = np.array([corner_1, corner_2, corner_3, corner_4])
            area_mat[toroidal_id, poloidal_id] = compute_quadrilateral_area(
                corners
            )

    nwl_mat = nwl_mat / area_mat

    return nwl_mat, toroidal_centroids, poloidal_centroids, area_mat


def plot_nwl(
    nwl_mat,
    toroidal_centroids,
    poloidal_centroids,
    filename="nwl",
    num_levels=11,
):
    """Generates contour plot of NWL.

    Arguments:
        nwl_mat (np.array): matrix of NWL solutions for each bin [MW/m^2].
        toroidal_centroids (list): centroids of toroidal angle bins [rad].
        poloidal_centroids (list): centroids of poloidal angle bins [rad].
        filename (str): name of plot output file (defaults to 'nwl').
        num_levels (int): number of contours in plot (defaults to 11).
    """
    toroidal_centroids = np.rad2deg(toroidal_centroids)
    poloidal_centroids = np.rad2deg(poloidal_centroids)
    levels = np.linspace(np.min(nwl_mat), np.max(nwl_mat), num=num_levels)

    fig, ax = plt.subplots()
    CF = ax.contourf(
        toroidal_centroids, poloidal_centroids, nwl_mat.T, levels=levels
    )
    cbar = plt.colorbar(CF)

    cbar.ax.set_ylabel(r"Neutron wall loading (MW/m$^2$)")
    ax.set_xlabel("Toroidal Angle [deg]")
    ax.set_ylabel("Poloidal Angle [deg]")

    cbar.ax.set_yticks(levels[::2])
    ax.set_xticks(
        [int(i) for i in np.linspace(0.0, np.max(toroidal_centroids), num=11)]
    )
    ax.set_yticks([int(i) for i in np.linspace(0.0, 360.0, num=11)])

    export_path = Path(filename).with_suffix(".png")
    fig.savefig(export_path)
