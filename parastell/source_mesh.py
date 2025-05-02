import argparse
from pathlib import Path

import numpy as np
from pymoab import core, types
import pystell.read_vmec as read_vmec

from . import log
from .utils import ToroidalMesh, read_yaml_config, filter_kwargs, m2cm, m3tocm3

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


class SourceMesh(ToroidalMesh):
    """Generates a source mesh that describes the relative source intensity of
    neutrons in a magnetically confined plasma described by a VMEC plasma
    equilibrium. Inherits from ToroidalMesh.

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
        logger (object): logger object (defaults to None). If no logger is
            supplied, a default logger will be instantiated.

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
        super().__init__(logger=logger)

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

        self._add_tags_to_core()

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
        angle = np.deg2rad(angle)

        if angle > 2 * np.pi:
            e = AttributeError("Toroidal extent cannot exceed 360.0 degrees.")
            self._logger.error(e.args[0])
            raise e

        if angle == 2 * np.pi and self._num_toroidal_pts % 2 != 1:
            e = AttributeError(
                "To ensure that tetrahedral faces are coincident at the end of "
                "the closed toroidal loop, the number of toroidal intervals "
                "must be even. To ensure this, the number of toroidal grid "
                "points must be odd."
            )
            self._logger.error(e.args[0])
            raise e

        self._toroidal_extent = angle

    def _add_tags_to_core(self):
        """Creates PyMOAB core instance with source strength tag.
        (Internal function not intended to be called externally)
        """
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

        # Exclude entry at magnetic axis
        cfs_grid_pts = np.linspace(0.0, 1.0, num=self.num_cfs_pts)[1:]

        # Exclude repeated entry at 0 == 2*pi
        poloidal_grid_pts = np.linspace(
            0, 2 * np.pi, num=self._num_poloidal_pts
        )[:-1]

        toroidal_grid_pts = np.linspace(
            0, self._toroidal_extent, num=self._num_toroidal_pts
        )
        # Conditionally exclude repeated entry at 0 == 2*pi
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

        self.add_vertices(self.coords)

    def _compute_tet_data(self, tet_ids, tet):
        """Computes tetrahedron neutron source strength, using five-node
        Gaussian quadrature, and volume, and sets the corresponding values of
        the respective tags for that tetrahedron.
        (Internal function not intended to be called externally)

        Arguments:
            tet_ids (list of int): tetrahedron vertex indices.
            tet (object): pymoab.EntityHandle of tetrahedron.

        Returns:
            ss (float): integrated source strength for tetrahedron.
            tet_vol (float): volume of tetrahedron
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

        self.strengths.append(ss)
        self.volumes.append(tet_vol)

        # Tag tetrahedra with data
        self.mbc.tag_set_data(self.source_strength_tag, tet, [ss])
        self.mbc.tag_set_data(self.volume_tag, tet, [tet_vol])

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
        surface_idx, poloidal_idx, toroidal_idx = vertex_idx

        ma_offset = toroidal_idx * self.verts_per_plane

        # Wrap around if final plane and it is 2*pi
        if (
            self._toroidal_extent == 2 * np.pi
            and toroidal_idx == self._num_toroidal_pts - 1
        ):
            ma_offset = 0

        # Compute index offset from closed flux surface, taking single vertex
        # at magnetic axis into account
        if surface_idx == 0:
            surface_offset = surface_idx
        else:
            surface_offset = (surface_idx - 1) * self.verts_per_ring + 1

        poloidal_offset = poloidal_idx
        # Wrap around if poloidal angle is 2*pi
        if poloidal_idx == self._num_poloidal_pts - 1:
            poloidal_offset = 0

        id = ma_offset + surface_offset + poloidal_offset

        return id

    def create_mesh(self):
        """Creates volumetric source mesh in real space."""
        self._logger.info("Constructing source mesh...")

        for toroidal_idx in range(self._num_toroidal_pts - 1):
            # Create tetrahedra for wedges at center of plasma
            for poloidal_idx in range(self._num_poloidal_pts - 1):
                tets, vertex_id_list = self._create_tets_from_wedge(
                    poloidal_idx, toroidal_idx
                )
                [
                    self._compute_tet_data(tet_ids, tet)
                    for tet_ids, tet in zip(vertex_id_list, tets)
                ]

            # Create tetrahedra for hexahedra beyond center of plasma
            for cfs_idx in range(1, self.num_cfs_pts - 1):
                for poloidal_idx in range(self._num_poloidal_pts - 1):
                    tets, vertex_id_list = self._create_tets_from_hex(
                        cfs_idx, poloidal_idx, toroidal_idx
                    )
                    [
                        self._compute_tet_data(tet_ids, tet)
                        for tet_ids, tet in zip(vertex_id_list, tets)
                    ]


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
        **(filter_kwargs(source_mesh_dict, ["filename"])),
    )


if __name__ == "__main__":
    generate_source_mesh()
