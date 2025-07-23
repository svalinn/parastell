import yaml
import tempfile
from functools import cached_property
import tempfile
from pathlib import Path
from abc import ABC

import numpy as np
import math
from scipy.ndimage import gaussian_filter
from pymoab import core, types
import pydagmc
import cadquery as cq
import gmsh
from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing
from OCP.StlAPI import StlAPI_Reader
from OCP.TopoDS import TopoDS_Shape
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_SHELL

from . import log

m2cm = 100
m3tocm3 = m2cm * m2cm * m2cm


def downsample_loop(list, sample_mod):
    """Downsamples a list representing a closed loop.

    Arguments:
        list (iterable): closed loop.
        sample_mod (int): sampling modifier.

    Returns:
        (iterable): downsampled closed loop
    """
    return np.append(list[:-1:sample_mod], [list[0]], axis=0)


def enforce_helical_symmetry(matrix):
    """Ensures that a matrix is helically symmetric according to stellarator
    geometry by overwriting certain matrix elements.

    Assumes several qualities about the input matrix:
        - Regular spacing between angles defining matrix
        - Matrix represents a full stellarator period
        - The first row corresponds to a toroidal angle at the beginning of a
          period (i.e., poloidal symmetry is expected)

    Arguments:
        matrix (2-D iterable of float): matrix to be made helically symmetric.

    Returns:
        matrix (2-D iterable of float): helically symmetric matrix.
    """
    num_rows, num_columns = matrix.shape

    # Ensure rows represent closed loops
    matrix[:, -1] = matrix[:, 0]

    # Ensure poloidal symmetry at beginning of period
    matrix[0] = np.concatenate(
        [
            # Ceil and floor ensure middle element of odd sized array is
            # included only once
            matrix[0, : math.ceil(num_columns / 2)],
            np.flip(matrix[0, : math.floor(num_columns / 2)]),
        ]
    )

    # Ensure helical symmetry toroidally and poloidally by mirroring the period
    # about both matrix axes
    flattened_matrix = matrix.flatten()
    flattened_length = len(flattened_matrix)

    first_half = flattened_matrix[: math.ceil(flattened_length / 2)]
    last_half = np.flip(flattened_matrix[: math.floor(flattened_length / 2)])
    flattened_matrix = np.concatenate([first_half, last_half])

    matrix = flattened_matrix.reshape((num_rows, num_columns))

    return matrix


def expand_list(list_to_expand, num):
    """Expands a list of ordered floats to a total number of entries by
    linearly interpolating between entries, inserting a proportional number of
    new entries between original entries. If num <= len(list), list_to_expand
    is not modified. It is possible that the result will have slightly more
    or fewer elements than num, due to round off approximations.

    Arguments:
        list_to_expand (iterable of float): list to be expanded.
        num (int): desired number of entries in expanded list.

    Returns:
        list_exp (iterable of float): expanded list.
    """
    if len(list_to_expand) >= num:
        return list_to_expand

    list_exp = []

    init_entry = list_to_expand[0]
    final_entry = list_to_expand[-1]
    extent = final_entry - init_entry

    avg_diff = extent / (num - 1)

    for entry, next_entry in zip(list_to_expand[:-1], list_to_expand[1:]):
        # Only add entries to current block if difference between entry and
        # next_entry is greater than desired average
        num_new_entries = 0

        if next_entry - entry > avg_diff:
            # Goal is to create bins of approximately avg_diff width between
            # entry and next_entry
            num_new_entries = int(round((next_entry - entry) / avg_diff)) - 1

        # Manually append first entry
        list_exp = np.append(list_exp, entry)

        # If num_new_entries == 0, don't add new entries
        # First and last elements of new_entries are entry and next_entry,
        # respectively
        new_entries = np.linspace(
            entry,
            next_entry,
            num=num_new_entries + 2,
        )[1:-1]

        list_exp = np.append(list_exp, new_entries)

    list_exp = np.append(list_exp, final_entry)

    return list_exp


def check_ascending(iterable):
    """Checks if an iterable's elements are in ascending order.

    Arguments:
        iterable (iterable of int or float): iterable to check.

    Returns:
        (bool): flag to indicate whether the iterable's elements are in
            ascending order.
    """
    return np.all(
        elem < next_elem
        for elem, next_elem in zip(iterable[:-1], iterable[1:])
    )


def filter_kwargs(
    dict, allowed_kwargs, all_kwargs=False, fn_name=None, logger=None
):
    """Constructs a dictionary of keyword arguments with corresponding values
    for a class method from an input dictionary based on a list of allowable
    keyword argument argument names. Conditionally raises an exception if the
    user flags that the supplied list of allowable arguments should represent
    all keys present in the input dictionary of arguments.

    Arguments:
        dict (dict): dictionary of arguments and corresponding values.
        allowed_kwargs (list of str): list of allowed keyword argument names.
        all_kwargs (bool): flag to indicate whether 'allowed_kwargs' should
            represent all keys present in 'dict' (optional, defaults to False).
        fn_name (str): name of class method (optional, defaults to None). If
            'all_kwargs' is True, a method name should be supplied.
        logger (object): logger object (optional, defaults to None). If
            'all_kwargs' is True, a logger object should be supplied.

    Returns:
        kwarg_dict (dict): dictionary of keyword arguments and values.
    """
    allowed_keys = dict.keys() & allowed_kwargs
    extra_keys = dict.keys() - allowed_kwargs

    if all_kwargs and extra_keys:
        e = ValueError(
            f"{extra_keys} not supported keyword argument(s) of "
            f'"{fn_name}"'
        )
        logger.error(e.args[0])
        raise e

    return {name: dict[name] for name in allowed_keys}


def normalize(vec_list):
    """Normalizes a set of vectors.

    Arguments:
        vec_list (1 or 2D np array): single 1D vector or array of 1D vectors
            to be normalized
    Returns:
        vec_list (np array of same shape as input): single 1D normalized vector
            or array of normalized 1D vectors
    """
    if len(vec_list.shape) == 1:
        return vec_list / np.linalg.norm(vec_list)
    elif len(vec_list.shape) == 2:
        return vec_list / np.linalg.norm(vec_list, axis=1)[:, np.newaxis]
    else:
        print('Input "vec_list" must be 1-D or 2-D NumPy array')


def read_yaml_config(filename):
    """Read YAML file describing ParaStell configuration and extract all data."""
    with open(filename) as yaml_file:
        all_data = yaml.safe_load(yaml_file)

    return all_data


def reorder_loop(list, index):
    """Reorders a list representing a closed loop.

    Arguments:
        list (iterable): closed loop.
        index (int): list index about which to reorder loop.

    Returns:
        (iterable): reordered closed loop.
    """
    return np.concatenate([list[index:], list[1 : index + 1]])


def smooth_matrix(matrix, steps, sigma):
    """Smooths a matrix via Gaussian filtering, without allowing matrix
    elements to increase in value.

    Arguments:
        matrix (2-D iterable of float): matrix to be smoothed.
        steps (int): number of smoothing steps.
        sigma (float): standard deviation for Gaussian kernel.

    Returns:
        smoothed_matrix (2-D iterable of float): smoothed matrix.
    """
    previous_matrix = matrix

    for step in range(steps):
        smoothed_matrix = np.minimum(
            previous_matrix,
            gaussian_filter(
                previous_matrix,
                sigma=sigma,
                mode="wrap",
            ),
        )
        previous_matrix = smoothed_matrix

    return smoothed_matrix


def create_vol_mesh_from_surf_mesh(
    min_mesh_size, max_mesh_size, algorithm, filename
):
    """Creates a volumetric mesh from a surface mesh, according to specified
    minimum and maximum mesh element sizes, using Gmsh. The resultant mesh will
    maintain the given surface mesh at its boundary. Assumes Gmsh has already
    been initialized.

    Arguments:
        min_mesh_size (float): minimum size of mesh elements (defaults to 5.0).
        max_mesh_size (float): maximum size of mesh elements (defaults to
            20.0).
        algorithm (int): integer identifying the meshing algorithm to use
                for the surface boundary. Options are as follows, refer to Gmsh
                documentation for explanations of each.
                1: MeshAdapt, 2: automatic, 3: initial mesh only, 4: N/A,
                5: Delaunay, 6: Frontal-Delaunay, 7: BAMG, 8: Frontal-Delaunay
                for Quads, 9: Packing of Parallelograms, 11: Quasi-structured
                Quad.
        filename (str): path to input mesh file. Must be a Gmsh-compatible file
            type. Options include VTK and MSH file types.

    Returns:
        filename (str): path to remeshed mesh output file. The output file type
            and path will be the same as the input.
    """
    gmsh.open(filename)

    surfaces = gmsh.model.getEntities(dim=2)
    surface_tags = [s[1] for s in surfaces]
    surface_loop = gmsh.model.geo.addSurfaceLoop(surface_tags)
    gmsh.model.geo.addVolume([surface_loop])

    gmsh.model.geo.synchronize()

    gmsh.option.setNumber("Mesh.MeshSizeMin", min_mesh_size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", max_mesh_size)
    gmsh.option.setNumber("Mesh.Algorithm", algorithm)

    gmsh.model.mesh.generate(dim=3)

    gmsh.write(filename)

    gmsh.clear()

    return filename


class DAGMCRenumberizer(object):
    """Class to facilitate renumbering of entities to combine DAGMC models.

    Arguments:
        mb (PyMOAB Core): PyMOAB core with the DAGMC models to be renumbered
        loaded. Optional.
    """

    def __init__(self, mb=None):
        self.mb = mb
        if mb is None:
            self.mb = core.Core()

    @cached_property
    def global_id_tag(self):
        return self.mb.tag_get_handle(
            "GLOBAL_ID", 1, types.MB_TYPE_INTEGER, types.MB_TAG_DENSE
        )

    @cached_property
    def category_tag(self):
        """Returns the category tag used to intidate the use of meshset. Values
        include "Group", "Volume", "Surface", "Curve" and "Vertex".
        """
        return self.mb.tag_get_handle(
            types.CATEGORY_TAG_NAME,
            types.CATEGORY_TAG_SIZE,
            types.MB_TYPE_OPAQUE,
            types.MB_TAG_SPARSE,
            create_if_missing=True,
        )

    def load_file(self, filename):
        """Load DAGMC model from file.

        Arguments:
            filename (str): Path to DAGMC model to be loaded.
        """
        self.mb.load_file(filename)

    def renumber_ids(self):
        """Renumbers the ids from 1 to N where N is the total number of
        entities in that category.
        """
        categories = ["Vertex", "Curve", "Surface", "Volume", "Group"]
        root_set = self.mb.get_root_set()
        for category in categories:
            category_set = self.mb.get_entities_by_type_and_tag(
                root_set, types.MBENTITYSET, self.category_tag, [category]
            )
            num_ids = len(category_set)
            if num_ids != 0:
                set_ids = list(range(1, num_ids + 1))
                self.mb.tag_set_data(
                    self.global_id_tag,
                    category_set,
                    set_ids,
                )


def combine_dagmc_models(models_to_merge):
    """Takes a list of DAGMC models, and renumbers entity ids such that they
    will no longer clash, allowing the models to be loaded into the same
    PyMOAB core instance, and saved to a single model.

    Arguments:
        models_to_merge (list of PyMOAB core): List of DAGMC models to be
            merged.
    Returns:
        combined_model (pydagmc.Model): Single DAGMC model containing the
            combined individual models.
    """
    renumberizer = DAGMCRenumberizer()
    for model in models_to_merge:
        with tempfile.NamedTemporaryFile(
            delete=True, suffix=".h5m"
        ) as temp_file:
            temp_filename = temp_file.name
            model.write_file(temp_filename)
            renumberizer.load_file(temp_filename)
    renumberizer.renumber_ids()
    return pydagmc.Model(renumberizer.mb)


def stl_to_cq_solid(stl_path, tolerance=0.001):
    """Create a solid volume in CadQuery from an STL file containing the
    triangles defining a watertight DAGMC volume. The resulting volume should
    have surfaces that exactly match the STL files provided. Useful for
    obtaining CAD representations of volumes in DAGMC models.

    Arguments:
        stl_path (str): Path to the STL file defining the volume.
        tolerance (float): Distance (in whatever units the STL file is in)
            at which to consider vertices coincident or edges colinear.

    Returns:
        cq_solid (CadQuery Solid): Solid volume with surfaces defined by the
            the triangles in the provided STL files.
    """
    sewer = BRepBuilderAPI_Sewing(tolerance)

    reader = StlAPI_Reader()
    shape = TopoDS_Shape()
    reader.Read(shape, stl_path)
    sewer.Add(shape)

    sewer.Perform()
    sewn_shape = sewer.SewedShape()

    shell_explorer = TopExp_Explorer(sewn_shape, TopAbs_SHELL)
    shell_shape = shell_explorer.Current()

    cq_solid = cq.Solid.makeSolid(cq.Shape(shell_shape))

    return cq_solid


def dagmc_volume_to_step(
    dagmc_model, volume_id, step_file_path, tolerance=0.001
):
    """Create a STEP file with surfaces defined by the triangles making up the
    surface of a DAGMC volume and save it to file.

    Arguments:
        dagmc_model (PyDAGMC Model object): DAGMC model containing the
            volume to be converted to STEP.
        volume_id (int): ID of the volume to be converted to STEP.
        step_file_path (str): Path on which to save the resulting step file.
        tolerance (float): Distance (in whatever units the STL file is in)
            at which to consider vertices coincident or edges colinear.
            Defaults to 0.001.
    """
    volume = dagmc_model.volumes_by_id[volume_id]

    with tempfile.NamedTemporaryFile(delete=True, suffix=".stl") as temp_file:
        stl_path = temp_file.name
        dagmc_model.mb.write_file(
            stl_path, output_sets=[s.handle for s in volume.surfaces]
        )
        cq_solid = stl_to_cq_solid(stl_path, tolerance)

    num_faces = len(cq_solid.Faces())
    num_tris = len(volume.triangle_handles)

    if num_tris != num_faces:
        raise ValueError(
            f"Number of faces in the STEP file ({num_faces}) does"
            "not match the number of triangles in the DAGMC volume"
            f"({num_tris}). Please verify that your geometry is valid or use"
            "a tighter tolerance."
        )

    cq_solid.exportStep(str(Path(step_file_path).with_suffix(".step")))


def get_obmp_index(coords):
    """Finds the index of the outboard midplane (or nearest to) coordinate on a
    closed loop. Assumes the first and final points of the loop are equal.

    Arguments:
        coords (Nx2 numpy.array of float): list of R, Z coordinates for the
            closed loop.

    Returns:
        obmp_index (int): index of the outboard midplane point.
    """
    # Define small value
    eps = 1e-10
    # Shift some coordinates by eps to compute appropriate midplane flags
    # If z = 0, then midplane flag = 0, then incorrect maximum radius computed,
    # then incorrect OB midplane index
    # Replace z-coordinates at 0 with eps
    np.place(coords[:, 1], np.abs(coords[:, 1]) < eps, [eps])

    radii = coords[:, 0]
    # Determine whether adjacent points cross the midplane (if so, they will
    # have opposite signs)
    shifted_coords = np.append(coords[1:], [coords[1]], axis=0)
    midplane_flags = -np.sign(coords[:, 1] * shifted_coords[:, 1])

    # Count number of crossing points (will have positive magnitude)
    count = np.count_nonzero(midplane_flags > 0)

    # If no crossing points, loop does not cross midplane
    if count == 0:
        # Find index of point closest to midplane
        obmp_index = np.argmin(np.abs(coords[:, 1]))
    # If 1 crossing point, coordinates likely do not represent a closed loop
    elif count == 1:
        e = AssertionError(
            "Only one crossing point found. Coordinates given likely do not "
            "represent a closed loop."
        )
        raise e
    # If 2 or more crossing points, loop crosses midplane
    elif count >= 2:
        # Find index of outboard midplane point
        obmp_index = np.argmax(midplane_flags * radii)

        obmp_index = obmp_index + (
            np.argmin(
                np.abs(
                    [
                        coords[obmp_index - 1, 1],
                        coords[obmp_index, 1],
                        coords[obmp_index + 1, 1],
                    ]
                )
            )
            - 1
        )

        if obmp_index == len(coords) - 1:
            obmp_index = 0

    return obmp_index


def orient_coords(coords, positive=True):
    """Orients closed loop coordinates such that they initially
    progress positively or negatively in the z-direction.

    Arguments:
        coords (Nx2 numpy.array of float): list of R, Z coordinates for the
            closed loop.
        positive (bool): progress coordinates in positive z-direciton
            (defaults to True). If negative, coordinates will progress in
            negative direction.

    Returns:
        coords (Nx2 numpy.array of float): reordered list of R, Z coordinates
            for the closed loop.
    """
    if positive == (coords[0, 1] > coords[1, 1]):
        coords = np.flip(coords, axis=0)

    return coords


def format_surface_coords(surface_coords):
    """Reformats a list of closed loops such that each begins at the outboard
    midplane (or nearest to) coordinate and progresses counter-clockwise.

    Arguments:
        surface_coords (Nx2 numpy.array of float): list of R, Z coordinates for the
            closed loop.

    Returns:
        (Nx2 numpy.array of float): reformatted list of R, Z coordinates for
            the closed loop.
    """
    new_surface_coords = []

    for toroidal_slice in surface_coords:
        obmp_index = get_obmp_index(toroidal_slice)
        slice_coords = reorder_loop(toroidal_slice, obmp_index)
        slice_coords = orient_coords(slice_coords)
        new_surface_coords.append(slice_coords)

    return np.array(new_surface_coords)


def ribs_from_kisslinger_format(
    filename, start_line=2, scale=1.0, delimiter="\t", format=True
):
    """Reads a Kisslinger format file and extracts the R, Z data, the number of
    periods, and the toroidal angles at which the R, Z data is specified.

    The expected format is as follows:

    Comments up to start line.

    Start line: A single line containing:
        3 delimiter separated ints, representing:
            - the number of toroidal angles
            - the number of points per toroidal angle
            - the number of periods in the device
        Only the first 3 ints will be read from this line.

    Toroidal angle block: A set of lines of length points per toroidal angle
        plus 1, where:
            - The first line is the corresponding toroidal angle, in degrees
            - The subsequent lines are the delimiter separated R, Z values
                of each point on the toroidal angle

    There should be the same number of toroidal angle blocks as the number
    of toroidal angles specified in the start line.

    This file is expected not to have any blank lines.

    Arguments:
        filename (str): Path to the file to be read.
        start_line (int): Line at which the data should start being read. This
            should be the line that includes the number of toroidal angles,
            number of points per toroidal angle, and number of periods.
            Defaults to 2.
        scale (float): a scaling factor between input and output data
            (defaults to 1.0).
        delimiter (str): delimiter used to signify new coordiante values
            (defaults to " ").
        format (bool): flag to indicate whether the data should be formatted
            (defaults to True).

    Returns:
        toroidal_angles (numpy array): Toroidal angles in the
            angles in the input file in degrees.
        num_toroidal_angles (int): Number of toroidal angles as specified in
            the file header.
        num_poloidal_angles (int): Number of points at each toroidal angle as
            specified in the file header.
        periods (int): Number of periods as specified in the file header.
        surface_coords (numpy array): 3 dimensional numpy array where the first
            dimension corresponds to individual ribs, the second to the
            position on a rib, and the third to the R,Z coordinates of the
            point.
    """

    with open(file=filename) as file:
        data = file.readlines()[start_line - 1 :]

    surface_coords = []
    toroidal_angles = []
    num_toroidal_angles, num_poloidal_angles, periods = (
        int(x) for x in data[0].rstrip().split(delimiter)[0:3]
    )

    ribs = [
        data[x : x + num_poloidal_angles + 1]
        for x in range(1, len(data[1:]), num_poloidal_angles + 1)
    ]

    for rib in ribs:
        toroidal_angle = float(rib[0].rstrip())
        toroidal_angles.append(toroidal_angle)
        profile = []
        for loci in rib[1:]:
            loci = loci.rstrip()
            r_z_coords = [
                float(coord) * scale for coord in loci.split(delimiter)
            ]
            profile.append(r_z_coords)
        surface_coords.append(profile)

    surface_coords = np.array(surface_coords)

    if format:
        surface_coords = format_surface_coords(surface_coords)

    return (
        np.array(toroidal_angles),
        num_toroidal_angles,
        num_poloidal_angles,
        periods,
        surface_coords,
    )


class ToroidalMesh(ABC):
    """An abstract class to facilitate generation of structured toroidal meshes
    in MOAB. The inheriting class must have a "_get_vertex_id" method that maps
    vertex IDs in (surface ID, poloidal ID, toroidal ID) space to row-major
    order, as given in the "add_vertices" method. It is also expected that the
    inheriting class has some method that iteratively calls
    "_create_tets_from_hex" and/or "_create_tets_from_wedge" to generate the
    mesh.

    Arguments:
        logger (object): logger object (defaults to None). If no logger is
            supplied, a default logger will be instantiated.
    """

    def __init__(self, logger=None):
        self.logger = logger

        self.mbc = core.Core()

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, logger_object):
        self._logger = log.check_init(logger_object)

    def add_vertices(self, coords):
        """Creates vertices and adds to PyMOAB core.

        Arguments:
            coords (Nx3 array of float): Cartesian coordinates of mesh
                vertices.
        """
        self.verts = self.mbc.create_vertices(coords)
        self.mesh_set = self.mbc.create_meshset()
        self.mbc.add_entity(self.mesh_set, self.verts)

    def _create_tet(self, tet_ids):
        """Creates tetrahedron and adds to PyMOAB core.
        (Internal function not intended to be called externally)

        Arguments:
            tet_ids (list of int): tetrahedron vertex indices.

        Returns:
            tet (object): pymoab.EntityHandle of tetrahedron.
        """
        tet_verts = [self.verts[int(id)] for id in tet_ids]
        tet = self.mbc.create_element(types.MBTET, tet_verts)
        self.mbc.add_entity(self.mesh_set, tet)

        return tet

    def _create_tets_from_hex(self, surface_idx, poloidal_idx, toroidal_idx):
        """Creates five tetrahedra from defined hexahedron.
        (Internal function not intended to be called externally)

        Arguments:
            surface_idx (int): index defining location along surface axis.
            poloidal_idx (int): index defining location along poloidal angle
                axis.
            toroidal_idx (int): index defining location along toroidal angle
                axis.
        """
        # relative offsets of vertices in a 3-D index space
        hex_vertex_stencil = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ]
        )

        # IDs of hex vertices applying offset stencil to current point
        hex_idx_data = (
            np.array([surface_idx, poloidal_idx, toroidal_idx])
            + hex_vertex_stencil
        )

        idx_list = [
            self._get_vertex_id(vertex_idx) for vertex_idx in hex_idx_data
        ]

        # Define MOAB canonical ordering of hexahedron vertex indices
        # Ordering follows right hand rule such that the fingers curl around
        # one side of the tetrahedron and the thumb points to the remaining
        # vertex. The vertices are ordered such that those on the side are
        # first, ordered clockwise relative to the thumb, followed by the
        # remaining vertex at the end of the thumb.
        # See Moreno, Bader, Wilson 2024 for hexahedron splitting
        canonical_ordering_schemes = [
            [
                [idx_list[0], idx_list[3], idx_list[1], idx_list[4]],
                [idx_list[1], idx_list[3], idx_list[2], idx_list[6]],
                [idx_list[1], idx_list[4], idx_list[6], idx_list[5]],
                [idx_list[3], idx_list[6], idx_list[4], idx_list[7]],
                [idx_list[1], idx_list[3], idx_list[6], idx_list[4]],
            ],
            [
                [idx_list[0], idx_list[2], idx_list[1], idx_list[5]],
                [idx_list[0], idx_list[3], idx_list[2], idx_list[7]],
                [idx_list[0], idx_list[7], idx_list[5], idx_list[4]],
                [idx_list[7], idx_list[2], idx_list[5], idx_list[6]],
                [idx_list[0], idx_list[2], idx_list[5], idx_list[7]],
            ],
        ]

        # Alternate canonical ordering schemes defining hexahedron splitting to
        # avoid gaps and overlaps between non-planar hexahedron faces
        scheme_idx = (surface_idx + poloidal_idx + toroidal_idx) % 2

        vertex_id_list = canonical_ordering_schemes[scheme_idx]

        tets = [self._create_tet(vertex_ids) for vertex_ids in vertex_id_list]

        return tets, vertex_id_list

    def _create_tets_from_wedge(self, poloidal_idx, toroidal_idx):
        """Creates three tetrahedra from defined wedge.
        (Internal function not intended to be called externally)

        Arguments:
            poloidal_idx (int): index defining location along poloidal angle
                axis.
            toroidal_idx (int): index defining location along toroidal angle
                axis.
        """
        # relative offsets of wedge vertices in a 3-D index space
        wedge_vertex_stencil = np.array(
            [
                [0, 0, 0],
                [1, poloidal_idx, 0],
                [1, poloidal_idx + 1, 0],
                [0, 0, 1],
                [1, poloidal_idx, 1],
                [1, poloidal_idx + 1, 1],
            ]
        )

        # Ids of wedge vertices applying offset stencil to current point
        wedge_idx_data = np.array([0, 0, toroidal_idx]) + wedge_vertex_stencil

        idx_list = [
            self._get_vertex_id(vertex_idx) for vertex_idx in wedge_idx_data
        ]

        # Define MOAB canonical ordering of wedge vertex indices
        # Ordering follows right hand rule such that the fingers curl around
        # one side of the tetrahedron and the thumb points to the remaining
        # vertex. The vertices are ordered such that those on the side are
        # first, ordered clockwise relative to the thumb, followed by the
        # remaining vertex at the end of the thumb.
        # See Moreno, Bader, Wilson 2024 for wedge splitting
        canonical_ordering_schemes = [
            [
                [idx_list[0], idx_list[2], idx_list[1], idx_list[3]],
                [idx_list[1], idx_list[3], idx_list[5], idx_list[4]],
                [idx_list[1], idx_list[3], idx_list[2], idx_list[5]],
            ],
            [
                [idx_list[0], idx_list[2], idx_list[1], idx_list[3]],
                [idx_list[3], idx_list[2], idx_list[4], idx_list[5]],
                [idx_list[3], idx_list[2], idx_list[1], idx_list[4]],
            ],
        ]

        # Alternate canonical ordering schemes defining wedge splitting to
        # avoid gaps and overlaps between non-planar wedge faces
        scheme_idx = (poloidal_idx + toroidal_idx) % 2

        vertex_id_list = canonical_ordering_schemes[scheme_idx]

        tets = [self._create_tet(vertex_ids) for vertex_ids in vertex_id_list]

        return tets, vertex_id_list

    def export_mesh(self, filename, export_dir=""):
        """Exports a tetrahedral mesh in H5M format via MOAB.

        Arguments:
            filename (str): name of H5M output file.
            export_dir (str): directory to which to export the h5m output file
                (defaults to empty string).
        """
        self._logger.info("Exporting mesh H5M file...")

        export_path = Path(export_dir) / Path(filename).with_suffix(".h5m")
        self.mbc.write_file(str(export_path))
