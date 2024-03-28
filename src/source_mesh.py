import argparse
import yaml
import log
from pathlib import Path

import numpy as np
from pymoab import core, types
import read_vmec

from src.utils import m2cm, source_def

def rxn_rate(s):
    """Calculates fusion reaction rate in plasma.

    Arguments:
        s (float): closed magnetic flux surface index in range of 0 (magnetic
            axis) to 1 (plasma edge).

    Returns:
        rr (float): fusion reaction rate (1/cm^3/s). Equates to neutron source
            density.
    """
    # Define m^3 to cm^3 constant
    m3tocm3 = 1e6

    if s == 1:
        rr = 0
    else:
        # Temperature
        T = 11.5 * (1 - s)
        # Ion density
        n = 4.8e20 * (1 - s**5)
        # Reaction rate in 1/m^3/s
        rr = 3.68e-18 * (n**2) / 4 * T**(-2/3) * np.exp(-19.94 * T**(-1/3))

    return rr/m3tocm3


class SourceMesh(object):
    """Generates a source mesh that describes the relative source intensity of
    neutrons in a magnetically confined plasma described by a VMEC plasma
    equilibrium.  

    The mesh will be defined on a regular grid in the plasma coordinates of s,
    theta, phi.  Mesh vertices will be defined on circular grid at each toroidal
    plane, and connected between toroidal planes. This results in wedge elements
    along the magnetic axis and hexagonal elements throughout the remainder of
    the mesh.  Each of these elements will be subdivided into tetrahedra (4 for
    the wedges and 5 for the hexahedra) to result in a mesh that is simpler to
    use.

    Each tetrahedron will be tagged with the volumetric neutron source intensity
    in n/cm3/s, using on a finite-element based quadrature of the source
    intensity evaluated at each vertex.

    Parameters:
        vmec (object): plasma equilibrium VMEC object as defined by the
            PyStell-UW VMEC reader. Must have a method
            'vmec2xyz(s, theta, phi)' that returns an (x,y,z) coordinate for
            any closed flux surface label, s, poloidal angle, theta, and
            toroidal angle, phi.
        num_s (int) : number of closed flux surfaces for vertex locations in
            each toroidal plane.
        num_theta (int) : number of poloidal angles for vertex locations in
            each toroidal plane.
        num_phi (int) : number of toroidal angles for planes of vertices.
        toroidal_extent (float) : extent of source mesh in toroidal direction
            [deg].
        scale (double): a scaling factor between the units of VMEC and [cm]
            (defaults to m2cm = 100).
        logger (object): logger object (defaults to None). If no logger is
            supplied, a default logger will be instantiated.
    """

    def __init__(
            self,
            vmec,
            num_s,
            num_theta,
            num_phi,
            toroidal_extent,
            scale=m2cm,
            logger=None
    ):

        self.vmec = vmec
        self.num_s = num_s
        self.num_theta = num_theta
        self.num_phi = num_phi
        self.toroidal_extent = np.deg2rad(toroidal_extent)
        self.scale = scale
        
        if logger == None or not logger.hasHandlers():
            self.logger = log.init()
        else:
            self.logger = logger
        
        self.strengths = []

        self._create_mbc()

    def _create_mbc(self):
        """Creates PyMOAB core instance with source strength tag.
        (Internal function not intended to be called externally)

        Returns:
            mbc (object): PyMOAB core instance.
            tag_handle (TagHandle): PyMOAB source strength tag.
        """
        self.mbc = core.Core()

        tag_type = types.MB_TYPE_DOUBLE
        tag_size = 1
        storage_type = types.MB_TAG_DENSE
        tag_name = "SourceStrength"
        self.tag_handle = self.mbc.tag_get_handle(
            tag_name, tag_size, tag_type, storage_type,
            create_if_missing=True
        )

    def create_vertices(self):
        """Creates mesh vertices and adds them to PyMOAB core.

        The grid of mesh vertices is generated from the user input
        defining the number of meshes in each of the plasma
        coordinate directions. Care is taken to manage the
        mesh at the 0 == 2 * pi wrap so that everything
        is closed and consistent.
        """
        self.logger.info('Computing source mesh point cloud...')
        
        phi_list = np.linspace(0, self.toroidal_extent, num=self.num_phi)
        # don't include magnetic axis in list of s values
        s_list = np.linspace(0.0, 1.0, num=self.num_s)[1:]
        # don't include repeated entry at 0 == 2*pi
        theta_list = np.linspace(0, 2*np.pi, num=self.num_theta)[:-1]

        # don't include repeated entry at 0 == 2*pi
        if self.toroidal_extent == 2*np.pi:
            phi_list = phi_list[:-1]

        self.verts_per_ring = theta_list.shape[0]
        # add one vertex per plane for magenetic axis
        self.verts_per_plane = s_list.shape[0] * self.verts_per_ring + 1

        num_verts = phi_list.shape[0] * self.verts_per_plane
        self.coords = np.zeros((num_verts, 3))
        self.coords_s = np.zeros(num_verts)

        # Initialize vertex index
        vert_idx = 0

        for phi in phi_list:
            # vertex coordinates on magnetic axis
            self.coords[vert_idx, :] = np.array(
                self.vmec.vmec2xyz(0, 0, phi)) * self.scale
            self.coords_s[vert_idx] = 0

            vert_idx += 1

            # vertex coordinate away from magnetic axis
            for s in s_list:
                for theta in theta_list:
                    self.coords[vert_idx, :] = np.array(
                        self.vmec.vmec2xyz(s, theta, phi)) * self.scale
                    self.coords_s[vert_idx] = s

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
        vertex_strengths = [rxn_rate(self.coords_s[id]) for id in tet_ids]

        # Define barycentric coordinates for integration points
        bary_coords = np.array([
            [0.25, 0.25, 0.25, 0.25],
            [0.5, 1/6, 1/6, 1/6],
            [1/6, 0.5, 1/6, 1/6],
            [1/6, 1/6, 0.5, 1/6],
            [1/6, 1/6, 1/6, 0.5]
        ])

        # Define weights for integration points
        int_w = np.array([-0.8, 0.45, 0.45, 0.45, 0.45])

        # Interpolate source strength at integration points
        ss_int_pts = np.dot(bary_coords, vertex_strengths)

        # Compute edge vectors between tetrahedron vertices
        edge_vectors = np.subtract(tet_coords[:3], tet_coords[3]).T

        tet_vol = np.abs(np.linalg.det(edge_vectors))/6

        ss = tet_vol * np.dot(int_w, ss_int_pts)

        return ss

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
        ss = self._source_strength(tet_ids)
        self.strengths.append(ss)

        # Tag tetrahedra with source strength data
        self.mbc.tag_set_data(self.tag_handle, tet, [ss])

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

        s_idx, theta_idx, phi_idx = vertex_idx

        ma_offset = phi_idx * self.verts_per_plane

        # Wrap around if final plane and it is 2*pi
        if self.toroidal_extent == 2*np.pi and phi_idx == self.num_phi - 1:
            ma_offset = 0

        # Compute index offset from closed flux surface
        s_offset = s_idx * self.verts_per_ring

        theta_offset = theta_idx

        # Wrap around if theta is 2*pi
        if theta_idx == self.num_theta:
            theta_offset = 1

        id = ma_offset + s_offset + theta_offset

        return id

    def _create_tets_from_hex(self, s_idx, theta_idx, phi_idx):
        """Creates five tetrahedra from defined hexahedron.
        (Internal function not intended to be called externally)

        Arguments:
            idx_list (list of int): list of hexahedron vertex indices.
        """

        # relative offsets of vertices in a 3-D index space
        hex_vertex_stencil = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1]
        ])

        # Ids of hex vertices applying offset stencil to current point
        hex_idx_data = np.array(
            [s_idx, theta_idx, phi_idx]) + hex_vertex_stencil

        idx_list = [self._get_vertex_id(vertex_idx)
                    for vertex_idx in hex_idx_data]

        # Define MOAB canonical ordering of hexahedron vertex indices
        hex_canon_ids = [
            [idx_list[0], idx_list[3], idx_list[1], idx_list[4]],
            [idx_list[7], idx_list[4], idx_list[6], idx_list[3]],
            [idx_list[2], idx_list[1], idx_list[3], idx_list[6]],
            [idx_list[5], idx_list[6], idx_list[4], idx_list[1]],
            [idx_list[3], idx_list[1], idx_list[4], idx_list[6]]
        ]

        for vertex_ids in hex_canon_ids:
            self._create_tet(vertex_ids)

    def _create_tets_from_wedge(self, theta_idx, phi_idx):
        """Creates three tetrahedra from defined wedge.
        (Internal function not intended to be called externally)

        Arguments:
            idx_list (list of int): list of wedge vertex indices.
        """

        # relative offsets of wedge vertices in a 3-D index space
        wedge_vertex_stencil = np.array([
            [0, 0,              0],
            [0, theta_idx,      0],
            [0, theta_idx + 1,  0],
            [0, 0,              1],
            [0, theta_idx,      1],
            [0, theta_idx + 1,  1]
        ])

        # Ids of wedge vertices applying offset stencil to current point
        wedge_idx_data = np.array([0, 0, phi_idx]) + wedge_vertex_stencil

        idx_list = [self._get_vertex_id(vertex_idx)
                    for vertex_idx in wedge_idx_data]

        # Define MOAB canonical ordering of wedge vertex indices
        wedge_canon_ids = [
            [idx_list[1], idx_list[2], idx_list[4], idx_list[0]],
            [idx_list[5], idx_list[4], idx_list[2], idx_list[3]],
            [idx_list[0], idx_list[2], idx_list[4], idx_list[3]]
        ]

        for vertex_ids in wedge_canon_ids:
            self._create_tet(vertex_ids)

    def create_mesh(self):
        """Creates volumetric source mesh in real space.
        """
        self.logger.info('Constructing source mesh...')

        self.mesh_set = self.mbc.create_meshset()
        self.mbc.add_entity(self.mesh_set, self.verts)

        for phi_idx in range(self.num_phi - 1):
            # Create tetrahedra for wedges at center of plasma
            for theta_idx in range(1, self.num_theta):
                self._create_tets_from_wedge(theta_idx, phi_idx)

            # Create tetrahedra for hexahedra beyond center of plasma
            for s_idx in range(self.num_s - 2):
                for theta_idx in range(1, self.num_theta):
                    self._create_tets_from_hex(s_idx, theta_idx, phi_idx)

    def export_mesh(self, filename='source_mesh', export_dir=''):
        """Use PyMOAB interface to write source mesh with source strengths
        tagged.
        """
        self.logger.info('Exporting source mesh H5M file...')
        
        export_path = Path(export_dir) / Path(filename).with_suffix('.h5m')
        self.mbc.write_file(str(export_path))


def parse_args():
    """Parser for running as a script
    """
    parser = argparse.ArgumentParser(prog='source_mesh')

    parser.add_argument(
        'filename',
        help='YAML file defining ParaStell source mesh configuration'
    )

    return parser.parse_args()


def read_yaml_src(filename):
    """Read YAML file describing the stellarator source mesh configuration and
    extract all data.
    """
    with open(filename) as yaml_file:
        all_data = yaml.safe_load(yaml_file)

    return all_data['vmec_file'], all_data['source']


def generate_source_mesh():
    """Main method when run as a command line script.
    """
    args = parse_args()

    vmec_file, source = read_yaml_src(args.filename)

    vmec = read_vmec.vmec_data(vmec_file)

    source_dict = source_def.copy()
    source_dict.update(source)

    source_mesh = SourceMesh(
        vmec, source_dict['num_s'], source_dict['num_theta'],
        source_dict['num_phi'], source_dict['toroidal_extent'],
        scale=source_dict['scale']
    )

    source_mesh.create_vertices()
    source_mesh.create_mesh()
    source_mesh.export_mesh(
        filename=source_dict['filename'],
        export_dir=source_dict['export_dir']
    )


if __name__ == "__main__":
    generate_source_mesh()
