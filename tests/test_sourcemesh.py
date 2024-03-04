import src.sourcemesh
import read_vmec
import numpy as np

vmec_file = 'files_for_tests/wout_vmec.nc'

def test_meshbasics():

    vmec = read_vmec.read_vmec_data(vmec_file)

    num_s_exp = 4
    num_theta_exp = 8
    num_phi_exp = 4
    tor_ext = 90.0

    source_mesh = SourceMesh(vmec, num_s_exp, num_theta_exp, num_phi_exp, 
                             tor_ext_exp)

    assert source_mesh.num_s == num_s_exp
    assert source_mesh.num_theta == num_theta_exp
    assert source_mesh.num_phi == num_phi_exp
    assert source_mesh.tor_ext == np.deg2rad(tor_ext_exp)

def test_vertices():

    vmec = read_vmec.read_vmec_data(vmec_file)

    num_s_exp = 4
    num_theta_exp = 8
    num_phi_exp = 4
    tor_ext = 90.0
    
    num_verts_exp = num_phi_exp * ( (num_s_exp - 1) * (num_theta_exp - 1) + 1)

    source_mesh = SourceMesh(vmec, num_s_exp, num_theta_exp, num_phi_exp, 
                             tor_ext_exp)
    
    source_mesh.create_vertices()

    assert source_mesh.coord.shape == (num_verts_exp, 3)
    assert source_mesh.coord_s.shape == (num_verts_exp,)

    assert source_mesh.verts.shape == (num_verts_exp,)
