import argparse
import yaml
import numpy as np
from pymoab import core, types

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
    def __init__(
            self,
            vmec_, 
            num_s_,
            num_theta_,
            num_phi_,
            tor_ext_
    ):
        
        self.vmec = vmec_
        self.num_s = num_s_
        self.num_theta = num_theta_
        self.num_phi = num_phi_
        self.tor_ext = np.deg2rad(tor_ext_)

        self.create_mbc()
        self.create_vertices()
        self.create_mesh()


    def create_mbc(self):
        """Creates PyMOAB core instance with source strength tag.

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
            tag_name, tag_size,tag_type, storage_type,
            create_if_missing = True
        )

        return

    def create_vertices(self):
        """Creates mesh vertices and adds them to PyMOAB core.
        """
        phi_list = np.linspace(0, tor_ext, num = num_phi)
        # don't include magnetic axis in list of s values
        s_list = np.linspace(0.0, 1.0, num = num_s)[1:]
        # don't include repeated entry at 0 == 2*pi
        theta_list = np.linspace(0, 2*np.pi, num = num_theta)[:-1]

        # don't include repeated entry at 0 == 2*pi
        if self.tor_ext == 2*np.pi:
            phi_list = phi_list[:-1]

        self.verts_per_ring = theta_list.shape[0]
        # add one vertex per plane for magenetic axis  
        self.verts_per_plane = s_list.shape[0] self.verts_per_ring + 1  

        num_verts = phi_list.shape[0] * self.verts_per_plane
        self.coords = np.zeros((num_verts, 3))
        self.coords_s = np.zeros(num_verts)

        # Initialize vertex index
        vert_idx = 0
        
        for phi in phi_list:
            # vertex coordinates on magnetic axis
            self.coords[vert_idx,:] = np.array(self.vmec.vmec2xyz(0, 0, phi)) * m2cm
            self.coords_s[vert_idx] = 0
            
            vert_idx += 1

            # vertex coordinate away from magnetic axis
            for s in s_list:
                for theta in theta_list:
                    self.coords[vert_idx,:] = np.array(vmec.vmec2xyz(s, theta, phi)) * m2cm
                    self.coords_s[vert_idx] = s
                    
                    vert_idx += 1

        self.verts = self.mbc.create_vertices(self.coords)


    def source_strength(ids):
        """Computes neutron source strength for a tetrahedron using five-node
        Gaussian quadrature.

        Arguments:
            ids (list of int): tetrahedron vertex indices.

        Returns:
            ss (float): integrated source strength for tetrahedron.
        """

        # Initialize list of coordinates for each tetrahedron vertex
        tet_verts = [ self.verts[id] for id in ids ]
        # Initialize list of source strengths for each tetrahedron vertex
        ss_verts = [ rxn_rate(self.verts_s[id]) for id in ids ]

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
        ss_int_pts = np.dot(bary_coords, ss_verts)
        
        # Compute graph of tetrahedral vertices
        T = np.subtract(tet_verts[:3], tet_verts[3]).T
        
        tet_vol = np.abs(np.linalg.det(T))/6
        
        ss = tet_vol * np.dot(int_w, ss_int_pts)

        return ss



    def create_tet(self, ids):
        """Creates tetrahedron and adds to moab core.

        Arguments:
            ids (list of int): tetrahedron vertex indices.
        """

        tet_verts = [ self.verts[id] for id in ids ]

        # Create tetrahedron in PyMOAB
        tet = self.mbc.create_element(types.MBTET, tet_verts)
        self.mbc.add_entity(mesh_set, tet)
        
        # Compute source strength for tetrahedron
        ss = self.source_strength(ids)
        self.strengths.append(ss)

        # Tag tetrahedra with source strength data
        self.mbc.tag_set_data(tag_handle, tet, [ss])

        return        

    def create_tets_from_hex(s_idx, theta_idx, phi_idx):
        """Creates five tetrahedra from defined hexahedron.

        Arguments:
            idx_list (list of int): list of hexahedron vertex indices.
        """
        # Define eight hexahedron vertex indices in matrix format as
        # well as index offset accounting for magnetic axis vertex in
        # the form
        # [flux surface index, poloidal index, toroidal index]
        # This is a canonical ordering expected by MOAB for elements
        # of this shape
        hex_idx_data = np.array([s_idx, theta_idx, phi_idx]) + np.array([
            [0, 1, 0],
            [0, 0, 0],
            [0, 1, 1],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 0],
            [1, 1, 1],
            [1, 0, 1]
        ]) 

        idx_list = [ self.get_vertex_id(vertex_idx) for vertex_idx in hex_idx_data ]

        # Define MOAB canonical ordering of hexahedron vertex indices
        hex_canon_ids = [
            [idx_list[0], idx_list[3], idx_list[1], idx_list[4]],
            [idx_list[7], idx_list[4], idx_list[6], idx_list[3]],
            [idx_list[2], idx_list[1], idx_list[3], idx_list[6]],
            [idx_list[5], idx_list[6], idx_list[4], idx_list[1]],
            [idx_list[3], idx_list[1], idx_list[4], idx_list[6]]
        ]

        for vertex_ids in hex_canon_ids:
            self.create_tet(vertex_ids)
        
        return

    def create_tets_from_wedge(self, theta_idx, phi_idx):
        """Creates three tetrahedra from defined wedge.

        Arguments:
            idx_list (list of int): list of wedge vertex indices.
        """
        # Define six wedge vertex indices in matrix format as well as index
        # offset accounting for magnetic axis vertex in the form
        # [flux surface index, poloidal index, toroidal index]
        # This is a canonical ordering expected by MOAB for elements
        # of this shape
        wedge_idx_data = np.array([0, 0, phi_idx]) + np.array([
            [0, 0,           ,  0],
            [0, theta_idx,   ,  0],
            [0, theta_idx + 1,  0],
            [0, 0,           ,  1],
            [0, theta_idx,   ,  1],
            [0, theta_idx + 1,  1]
        ])
                        
        idx_list = [ self.get_vertex_id(vertex_idx) for vertex_idx in wedge_idx_data ]

        # Define MOAB canonical ordering of wedge vertex indices
        wedge_canon_ids = [
            [idx_list[1], idx_list[2], idx_list[4], idx_list[0]],
            [idx_list[5], idx_list[4], idx_list[2], idx_list[3]],
            [idx_list[0], idx_list[2], idx_list[4], idx_list[3]]
        ]
        
        for vertex_ids in wedge_canon_ids:
            self.create_tet(vertex_ids)

        return

    def get_vertex_id(self, vertex_idx):
        """Computes vertex index in row-major order as stored by MOAB from
        three-dimensional n x 3 matrix indices.

        Arguments:
            vert_idx (list of int): list of vertex 
                [toroidal angle index, flux surface index, poloidal angle index]

        Returns:
            id (int): vertex index in row-major order as stored by MOAB
        """

        s_idx, theta_idx, phi_idx = vert_idx

        ma_offset = phi_idx * self.verts_per_plane

        # Wrap around if final plane and it is 2*pi
        if tor_ext == 2*np.pi and phi_idx == num_phi - 1:
            ma_offset = 0
        
        # Compute index offset from closed flux surface
        s_offset = s_idx * self.verts_per_ring
        
        theta_offset = theta_idx

        # Wrap around if theta is 2*pi
        if theta_idx == self.num_theta:
            theta_offset = 1
        
        id = ma_offset + s_offset + theta_offset

        return id


    def create_mesh(self):
        """Creates volumetric source mesh in real space.
        """

        tet_set = self.mbc.create_meshset()
        self.mbc.add_entity(tet_set, self.verts)
        
        for phi_idx in range(self.num_phi - 1):
            # Create tetrahedra for wedges at center of plasma
            for theta_idx in range(1, self.num_theta):                
                self.create_tets_from_wedge(theta_idx, phi_idx)

            # Create tetrahedra for hexahedra beyond center of plasma
            for s_idx in range(self.num_s - 2):
                for theta_idx in range(1, self.num_theta):
                    self.create_tets_from_hex(idx_list)

    def write(self, filename):

        self.mbc.write(str(filename))

def parse_args():
    parser = argparse.ArgumentParser(prog='generate_stellarator_source_mesh')

    parser.add_argument('filename', desc='YAML file defining this case')

    return parser.parse_args()

def read_yaml_src():
    
    with open(filename) as stream:
        data = yaml.safe_load(stream)

    # extract data to define source mesh
    src_data = data

    return src_data            

def generate_source_mesh():
    args = parse_args()

    src_data = read_yaml_src(args.filename)

    source_mesh = SourceMesh(src_data['vmec'], src_data['num_s'], src_data['num_theta'], src_data['num_phi'], src_data['tor_ext'])

    source_mesh.write(src_data['export_dir'] / src_data['mesh_file'])

if __name__ == "__main__":
    generate_source_mesh()