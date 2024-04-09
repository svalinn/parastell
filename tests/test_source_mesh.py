from pathlib import Path

import numpy as np
import pytest
import read_vmec as rv

import parastell.source_mesh as sm


def remove_files():

    if Path('source_mesh.h5m').exists():
        Path.unlink('source_mesh.h5m')
    if Path('stellarator.log').exists():
        Path.unlink('stellarator.log')


@pytest.fixture
def source_mesh():

    vmec_file = Path('files_for_tests') / 'wout_vmec.nc'

    vmec = rv.vmec_data(vmec_file)

    num_s_exp = 4
    num_theta_exp = 8
    num_phi_exp = 4
    tor_ext_exp = 90.0
    scale_exp = 100

    source_mesh_obj = sm.SourceMesh(
        vmec, num_s_exp, num_theta_exp, num_phi_exp, tor_ext_exp,
        scale=scale_exp
    )

    return source_mesh_obj


def test_mesh_basics(source_mesh):

    num_s_exp = 4
    num_theta_exp = 8
    num_phi_exp = 4
    tor_ext_exp = 90.0
    scale_exp = 100

    remove_files()

    assert source_mesh.num_s == num_s_exp
    assert source_mesh.num_theta == num_theta_exp
    assert source_mesh.num_phi == num_phi_exp
    assert source_mesh.toroidal_extent == np.deg2rad(tor_ext_exp)
    assert source_mesh.scale == scale_exp

    remove_files()


def test_vertices(source_mesh):

    num_s = 4
    num_theta = 8
    num_phi = 4

    num_verts_exp = num_phi * ((num_s - 1) * (num_theta - 1) + 1)

    remove_files()
    
    source_mesh.create_vertices()

    assert source_mesh.coords.shape == (num_verts_exp, 3)
    assert source_mesh.coords_s.shape == (num_verts_exp,)
    assert len(source_mesh.verts) == num_verts_exp

    remove_files()


def test_export(source_mesh):

    remove_files()

    source_mesh.create_vertices()
    source_mesh.create_mesh()
    source_mesh.export_mesh()

    assert Path('source_mesh.h5m').exists()

    remove_files()
    