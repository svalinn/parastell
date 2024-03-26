import src.sourcemesh as sm
import read_vmec as rv
import numpy as np
from pathlib import Path

vmec_file = Path('files_for_tests') / 'wout_vmec.nc'


def test_meshbasics():

    vmec = rv.vmec_data(vmec_file)

    num_s_exp = 4
    num_theta_exp = 8
    num_phi_exp = 4
    tor_ext_exp = 90.0
    scale_exp = 100

    source_mesh = sm.SourceMesh(
        vmec, num_s_exp, num_theta_exp, num_phi_exp, tor_ext_exp,
        scale=scale_exp
    )

    assert source_mesh.num_s == num_s_exp
    assert source_mesh.num_theta == num_theta_exp
    assert source_mesh.num_phi == num_phi_exp
    assert source_mesh.toroidal_extent == np.deg2rad(tor_ext_exp)
    assert source_mesh.scale == scale_exp


def test_vertices():

    vmec = rv.vmec_data(vmec_file)

    num_s_exp = 4
    num_theta_exp = 8
    num_phi_exp = 4
    tor_ext_exp = 90.0
    scale_exp = 100

    num_verts_exp = num_phi_exp * ((num_s_exp - 1) * (num_theta_exp - 1) + 1)
    
    source_mesh = sm.SourceMesh(
        vmec, num_s_exp, num_theta_exp, num_phi_exp, tor_ext_exp,
        scale=scale_exp
    )

    source_mesh.create_vertices()

    assert source_mesh.coords.shape == (num_verts_exp, 3)
    assert source_mesh.coords_s.shape == (num_verts_exp,)
    assert len(source_mesh.verts) == num_verts_exp

def test_export():

    vmec = rv.vmec_data(vmec_file)

    num_s_exp = 4
    num_theta_exp = 8
    num_phi_exp = 4
    tor_ext_exp = 90.0
    scale_exp = 100

    source_mesh = sm.SourceMesh(
        vmec, num_s_exp, num_theta_exp, num_phi_exp, tor_ext_exp,
        scale=scale_exp
    )

    source_mesh.create_vertices()
    source_mesh.create_mesh()
    source_mesh.export_mesh()

    assert Path('source_mesh.h5m').exists() == True

    Path.unlink('source_mesh.h5m')
    Path.unlink('stellarator.log')
