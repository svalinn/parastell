import argparse
from pathlib import Path

import numpy as np
from pymoab import core, types
import pystell.read_vmec as read_vmec

from . import log as log
from .utils import read_yaml_config, filter_kwargs, m2cm, m3tocm3

export_allowed_kwargs = ["filename"]


def default_reaction_rate(n_i, T_i):
    """Default reaction rate formula for DT fusion assumes an equal mixture of
    D and T in a hot plasma. From A. Bader et al 2021 Nucl. Fusion 61 116060
    DOI 10.1088/1741-4326/ac2991


    Arguments:
        n_i (float) : ion density (ions per m3)
        T_i (float) : ion temperature (KeV)

    Returns:
        rr (float) : reaction rate in reactions/cm3/s. Equates to neutron source
            density.
    """
    if T_i == 0 or n_i == 0:
        return 0

    rr = (
        3.68e-18
        * (n_i**2)
        / 4
        * T_i ** (-2 / 3)
        * np.exp(-19.94 * T_i ** (-1 / 3))
    )

    return rr / m3tocm3


def default_plasma_conditions(s):
    """Calculates ion density and temperature as a function of the
    plasma paramter s using profiles found in A. Bader et al 2021 Nucl. Fusion
    61 116060 DOI 10.1088/1741-4326/ac2991

    Arguments:
        s (float): closed magnetic flux surface index in range of 0 (magnetic
            axis) to 1 (plasma edge).

    Returns:
        n_i (float) : ion density in ions/m3
        T_i (float) : ion temperature in KeV
    """

    # Temperature
    T_i = 11.5 * (1 - s)
    # Ion density
    n_i = 4.8e20 * (1 - s**5)

    return n_i, T_i


class SourceMesh(object):
    """Generates a source mesh that describes the relative source intensity of
    neutrons in a magnetically confined plasma described by a VMEC plasma
    equilibrium.

    The mesh will be defined on a regular grid in the flux coordinates of
    (CFS value, poloidal angle, toroidal angle).  Mesh vertices will be defined
    on circular grid at each toroidal plane, and connected between toroidal
    planes. This results in wedge elements along the magnetic axis and
    hexagonal elements throughout the remainder of the mesh. Each of these
    elements will be subdivided into tetrahedra (3 for the wedges and 5 for the
    hexahedra) to result in a mesh that is simpler to use.

    Each tetrahedron will be tagged with the volumetric neutron source
    intensity in n/cm3/s, using on a finite-element based quadrature of the
    source intensity evaluated at each vertex.

    Arguments:
        vmec_obj (object): plasma equilibrium VMEC object as defined by the
            PyStell-UW VMEC reader. Must have a method
            'vmec2xyz(cfs, poloidal_ang, toroidal_ang)' that returns
            a (x,y,z) coordinate for any closed flux surface value, cfs,
            poloidal angle, poloidal_ang, and toroidal angle, toroidal_ang.
        mesh_size (tuple of int): number of grid points along each axis of
            flux-coordinate space, in the order (num_cfs_pts, num_poloidal_pts,
            num_toroidal_pts). 'num_cfs_pts' is the number of closed flux
            surfaces for vertex locations in each toroidal plane.
            'num_poloidal_pts' is the number of poloidal angles for vertex
            locations in each toroidal plane. 'num_toroidal_pts' is the number
            of toroidal angles for planes of vertices.
        toroidal_extent (float): extent of source mesh in toroidal direction
            [deg].
        logger (object): logger object (optional, defaults to None). If no
            logger is supplied, a default logger will be instantiated.

    Optional attributes:
        scale (float): a scaling factor between the units of VMEC and [cm]
            (defaults to m2cm = 100).
        plasma_conditions (function): function that takes the plasma parameter
            s, and returns temperature and ion density with suitable units for
            the reaction_rate() function. Defaults to
            default_plasma_conditions()
        reaction_rate (function): function that takes the values returned by
            plasma_conditions() and returns a reaction rate in reactions/cm3/s
    """

    def __init__(
        self, vmec_obj, mesh_size, toroidal_extent, logger=None, **kwargs
    ):

        self.logger = logger
        self.vmec_obj = vmec_obj
        self.num_cfs_pts = mesh_size[0]
        self.num_poloidal_pts = mesh_size[1]
        self.num_toroidal_pts = mesh_size[2]
        self.toroidal_extent = toroidal_extent

        self.scale = m2cm
        self.plasma_conditions = default_plasma_conditions
        self.reaction_rate = default_reaction_rate

        for name in kwargs.keys() & (
            "scale",
            "plasma_conditions",
            "reaction_rate",
        ):
            self.__setattr__(name, kwargs[name])

        self.strengths = []
        self.volumes = []

        self._create_mbc()

    @property
    def num_poloidal_pts(self):
        return self._num_poloidal_pts

    @num_poloidal_pts.setter
    def num_poloidal_pts(self, value):
        if value % 2 != 1:
            e = AttributeError(
                "To ensure that tetrahedral faces are coincident at the end of "
                "the closed poloidal loop, the number of poloidal intervals "
                "must be even. To ensure this, the number of poloidal grid "
                "points must be odd."
            )
            self._logger.error(e.args[0])
            raise e
        self._num_poloidal_pts = value

    @property
    def num_toroidal_pts(self):
        return self._num_toroidal_pts

    @num_toroidal_pts.setter
    def num_toroidal_pts(self, value):
        self._num_toroidal_pts = value

    @property
    def toroidal_extent(self):
        return self._toroidal_extent

    @toroidal_extent.setter
    def toroidal_extent(self, angle):
        if angle > 360.0:
            e = AttributeError("Toroidal extent cannot exceed 360.0 degrees.")
            self._logger.error(e.args[0])
            raise e

        if angle == 360.0 and self._num_toroidal_pts % 2 != 1:
            e = AttributeError(
                "To ensure that tetrahedral faces are coincident at the end of "
                "the closed toroidal loop, the number of toroidal intervals "
                "must be even. To ensure this, the number of toroidal grid "
                "points must be odd."
            )
            self._logger.error(e.args[0])
            raise e

        self._toroidal_extent = np.deg2rad(angle)

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, logger_object):
        self._logger = log.check_init(logger_object)

    def _create_mbc(self):
        """Creates PyMOAB core instance with source strength tag.
        (Internal function not intended to be called externally)
        """
        self.mbc = core.Core()

        tag_type = types.MB_TYPE_DOUBLE
        tag_size = 1
        storage_type = types.MB_TAG_DENSE

        ss_tag_name = "Source Strength"
        self.source_strength_tag = self.mbc.tag_get_handle(
            ss_tag_name,
            tag_size,
            tag_type,
            storage_type,
            create_if_missing=True,
        )

        vol_tag_name = "Volume"
        self.volume_tag = self.mbc.tag_get_handle(
            vol_tag_name,
            tag_size,
            tag_type,
            storage_type,
            create_if_missing=True,
        )

    def create_vertices(self):
        """Creates mesh vertices and adds them to PyMOAB core.

        The grid of mesh vertices is generated from the user input
        defining the number of meshes in each of the plasma
        coordinate directions. Care is taken to manage the
        mesh at the 0 == 2 * pi wrap so that everything
        is closed and consistent.
        """
        self._logger.info("Computing source mesh point cloud...")

        toroidal_grid_pts = np.linspace(
            0, self._toroidal_extent, num=self._num_toroidal_pts
        )
        # don't include magnetic axis in list of s values
        cfs_grid_pts = np.linspace(0.0, 1.0, num=self.num_cfs_pts)[1:]
        # don't include repeated entry at 0 == 2*pi
        poloidal_grid_pts = np.linspace(
            0, 2 * np.pi, num=self._num_poloidal_pts
        )[:-1]

        # don't include repeated entry at 0 == 2*pi
        if self._toroidal_extent == 2 * np.pi:
            toroidal_grid_pts = toroidal_grid_pts[:-1]

        self.verts_per_ring = poloidal_grid_pts.shape[0]
        # add one vertex per plane for magenetic axis
        self.verts_per_plane = cfs_grid_pts.shape[0] * self.verts_per_ring + 1

        num_verts = toroidal_grid_pts.shape[0] * self.verts_per_plane
        self.coords = np.zeros((num_verts, 3))
        self.coords_cfs = np.zeros(num_verts)

        # Initialize vertex index
        vert_idx = 0

        for toroidal_ang in toroidal_grid_pts:
            # vertex coordinates on magnetic axis
            self.coords[vert_idx, :] = (
                np.array(self.vmec_obj.vmec2xyz(0, 0, toroidal_ang))
                * self.scale
            )
            self.coords_cfs[vert_idx] = 0

            vert_idx += 1

            # vertex coordinate away from magnetic axis
            for cfs in cfs_grid_pts:
                for poloidal_ang in poloidal_grid_pts:
                    self.coords[vert_idx, :] = (
                        np.array(
                            self.vmec_obj.vmec2xyz(
                                cfs, poloidal_ang, toroidal_ang
                            )
                        )
                        * self.scale
                    )
                    self.coords_cfs[vert_idx] = cfs

                    vert_idx += 1

        self.verts = self.mbc.create_vertices(self.coords)

    def _source_strength(self, tet_ids):
        """Computes neutron source strength for a tetrahedron using five-node
        Gaussian quadrature.
        (Internal function not intended to be called externally)

        Arguments:
            ids (list of int): tetrahedron vertex indices.

        Returns:
            ss (float): integrated source strength for tetrahedron.
        """

        # Initialize list of vertex coordinates for each tetrahedron vertex
        tet_coords = [self.coords[id] for id in tet_ids]

        # Initialize list of source strengths for each tetrahedron vertex
        vertex_strengths = [
            self.reaction_rate(*self.plasma_conditions(self.coords_cfs[id]))
            for id in tet_ids
        ]

        # Define barycentric coordinates for integration points
        bary_coords = np.array(
            [
                [0.25, 0.25, 0.25, 0.25],
                [0.5, 1 / 6, 1 / 6, 1 / 6],
                [1 / 6, 0.5, 1 / 6, 1 / 6],
                [1 / 6, 1 / 6, 0.5, 1 / 6],
                [1 / 6, 1 / 6, 1 / 6, 0.5],
            ]
        )

        # Define weights for integration points
        int_w = np.array([-0.8, 0.45, 0.45, 0.45, 0.45])

        # Interpolate source strength at integration points
        ss_int_pts = np.dot(bary_coords, vertex_strengths)

        # Compute edge vectors between tetrahedron vertices
        edge_vectors = np.subtract(tet_coords[:3], tet_coords[3]).T

        tet_vol = -np.linalg.det(edge_vectors) / 6

        ss = np.abs(tet_vol) * np.dot(int_w, ss_int_pts)

        return ss, tet_vol

    def _create_tet(self, tet_ids):
        """Creates tetrahedron and adds to pyMOAB core.
        (Internal function not intended to be called externally)

        Arguments:
            tet_ids (list of int): tetrahedron vertex indices.
        """

        tet_verts = [self.verts[int(id)] for id in tet_ids]
        tet = self.mbc.create_element(types.MBTET, tet_verts)
        self.mbc.add_entity(self.mesh_set, tet)

        # Compute source strength for tetrahedron
        ss, vol = self._source_strength(tet_ids)
        self.strengths.append(ss)
        self.volumes.append(vol)

        # Tag tetrahedra with data
        self.mbc.tag_set_data(self.source_strength_tag, tet, [ss])
        self.mbc.tag_set_data(self.volume_tag, tet, [vol])

    def _get_vertex_id(self, vertex_idx):
        """Computes vertex index in row-major order as stored by MOAB from
        three-dimensional n x 3 matrix indices.
        (Internal function not intended to be called externally)

        Arguments:
            vert_idx (list of int): list of vertex
                [flux surface index, poloidal angle index, toroidal angle index]

        Returns:
            id (int): vertex index in row-major order as stored by MOAB
        """

        cfs_idx, poloidal_idx, toroidal_idx = vertex_idx

        ma_offset = toroidal_idx * self.verts_per_plane

        # Wrap around if final plane and it is 2*pi
        if (
            self._toroidal_extent == 2 * np.pi
            and toroidal_idx == self._num_toroidal_pts - 1
        ):
            ma_offset = 0

        # Compute index offset from closed flux surface
        cfs_offset = cfs_idx * self.verts_per_ring

        poloidal_offset = poloidal_idx

        # Wrap around if poloidal angle is 2*pi
        if poloidal_idx == self._num_poloidal_pts:
            poloidal_offset = 1

        id = ma_offset + cfs_offset + poloidal_offset

        return id

    def _create_tets_from_hex(self, cfs_idx, poloidal_idx, toroidal_idx):
        """Creates five tetrahedra from defined hexahedron.
        (Internal function not intended to be called externally)

        Arguments:
            cfs_idx (int): index defining location along CFS axis.
            poloidal_idx (int): index defining location along poloidal angle axis.
            toroidal_idx (int): index defining location along toroidal angle axis.
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

        # Ids of hex vertices applying offset stencil to current point
        hex_idx_data = (
            np.array([cfs_idx, poloidal_idx, toroidal_idx])
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
        # cfs_idx begins at 0 and poloidal_idx begins at 1
        scheme_idx = ((cfs_idx + 1) + (poloidal_idx - 1) + toroidal_idx) % 2

        for vertex_ids in canonical_ordering_schemes[scheme_idx]:
            self._create_tet(vertex_ids)

    def _create_tets_from_wedge(self, poloidal_idx, toroidal_idx):
        """Creates three tetrahedra from defined wedge.
        (Internal function not intended to be called externally)

        Arguments:
            poloidal_idx (int): index defining location along poloidal angle axis.
            toroidal_idx (int): index defining location along toroidal angle axis.
        """

        # relative offsets of wedge vertices in a 3-D index space
        wedge_vertex_stencil = np.array(
            [
                [0, 0, 0],
                [0, poloidal_idx, 0],
                [0, poloidal_idx + 1, 0],
                [0, 0, 1],
                [0, poloidal_idx, 1],
                [0, poloidal_idx + 1, 1],
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
        # cfs_idx begins at 0 and poloidal_idx begins at 1
        scheme_idx = ((poloidal_idx - 1) + toroidal_idx) % 2

        for vertex_ids in canonical_ordering_schemes[scheme_idx]:
            self._create_tet(vertex_ids)

    def create_mesh(self):
        """Creates volumetric source mesh in real space."""
        self._logger.info("Constructing source mesh...")

        self.mesh_set = self.mbc.create_meshset()
        self.mbc.add_entity(self.mesh_set, self.verts)

        for toroidal_idx in range(self._num_toroidal_pts - 1):
            # Create tetrahedra for wedges at center of plasma
            for poloidal_idx in range(1, self._num_poloidal_pts):
                self._create_tets_from_wedge(poloidal_idx, toroidal_idx)

            # Create tetrahedra for hexahedra beyond center of plasma
            for cfs_idx in range(self.num_cfs_pts - 2):
                for poloidal_idx in range(1, self._num_poloidal_pts):
                    self._create_tets_from_hex(
                        cfs_idx, poloidal_idx, toroidal_idx
                    )

    def export_mesh(self, filename="source_mesh", export_dir=""):
        """Use PyMOAB interface to write source mesh with source strengths
        tagged.

        Arguments:
            filename: name of H5M output file, excluding '.h5m' extension
                (optional, defaults to 'source_mesh').
            export_dir (str): directory to which to export the H5M output file
                (optional, defaults to empty string).
        """
        self._logger.info("Exporting source mesh H5M file...")

        export_path = Path(export_dir) / Path(filename).with_suffix(".h5m")
        self.mbc.write_file(str(export_path))


def parse_args():
    """Parser for running as a script"""
    parser = argparse.ArgumentParser(prog="source_mesh")

    parser.add_argument(
        "filename",
        help="YAML file defining ParaStell source mesh configuration",
    )
    parser.add_argument(
        "-e",
        "--export_dir",
        default="",
        help=(
            "Directory to which output files are exported (default: working "
            "directory)"
        ),
        metavar="",
    )
    parser.add_argument(
        "-l",
        "--logger",
        default=False,
        help=(
            "Flag to indicate whether to instantiate a logger object (default: "
            "False)"
        ),
        metavar="",
    )

    return parser.parse_args()


def generate_source_mesh():
    """Main method when run as a command line script."""
    args = parse_args()

    all_data = read_yaml_config(args.filename)

    if args.logger == True:
        logger = log.init()
    else:
        logger = log.NullLogger()

    vmec_file = all_data["vmec_file"]
    vmec_obj = read_vmec.VMECData(vmec_file)

    source_mesh_dict = all_data["source_mesh"]

    source_mesh = SourceMesh(
        vmec_obj,
        source_mesh_dict["mesh_size"],
        source_mesh_dict["toroidal_extent"],
        logger=logger**source_mesh_dict,
    )

    source_mesh.create_vertices()
    source_mesh.create_mesh()

    source_mesh.export_mesh(
        export_dir=args.export_dir,
        **(filter_kwargs(source_mesh_dict, ["filename"]))
    )


if __name__ == "__main__":
    generate_source_mesh()
