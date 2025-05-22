from pathlib import Path

import numpy as np
import pytest
import pystell.read_vmec as read_vmec

import parastell.source_mesh as sm


files_to_remove = [
    "source_mesh.h5m",
    "stellarator.log",
]


def remove_files():
    for file in files_to_remove:
        if Path(file).exists():
            Path.unlink(file)


@pytest.fixture
def source_mesh():
    vmec_file = Path("files_for_tests") / "wout_vmec.nc"

    vmec_obj = read_vmec.VMECData(vmec_file)

    # Set mesh grids to minimum that maintains element aspect ratios that do not
    # result in negative volumes
    cfs_values = np.linspace(0.0, 1.0, num=6)
    poloidal_angles = np.linspace(0.0, 360.0, num=41)
    toroidal_angles = np.linspace(0.0, 15.0, num=9)

    source_mesh_obj = sm.SourceMesh(
        vmec_obj, cfs_values, poloidal_angles, toroidal_angles
    )

    return source_mesh_obj


def test_mesh_basics(source_mesh):
    """Tests whether SourceMesh arguments are instantiated as expected, by
    testing if:
        * after being set, member variables match inputs
    """
    remove_files()

    num_cfs_exp = 6
    num_poloidal_pts_exp = 41
    num_toroidal_pts_exp = 9
    tor_ext_exp = 15.0
    scale_exp = 100

    # Subtract 1 because magnetic axis gets excluded from stored iterable
    assert source_mesh.cfs_values.shape[0] == num_cfs_exp - 1
    # Subtract 1 because repeated point gets excluded from stored iterable
    assert source_mesh.poloidal_angles.shape[0] == num_poloidal_pts_exp - 1
    assert source_mesh.toroidal_angles.shape[0] == num_toroidal_pts_exp
    assert source_mesh._toroidal_extent == np.deg2rad(tor_ext_exp)
    assert source_mesh.scale == scale_exp

    remove_files()


def test_vertices(source_mesh):
    """Tests whether SourceMesh vertices are generated as expected, by testing
    if:
        * the correct number of vertices are produced, and that they have the
          expected dimension
        * the correct number CFS values are stored
    """
    remove_files()

    num_cfs_exp = 6
    num_poloidal_pts_exp = 41
    num_toroidal_pts_exp = 9

    num_verts_exp = num_toroidal_pts_exp * (
        (num_cfs_exp - 1) * (num_poloidal_pts_exp - 1) + 1
    )

    source_mesh.create_vertices()

    assert source_mesh.coords.shape == (num_verts_exp, 3)
    assert source_mesh.coords_cfs.shape == (num_verts_exp,)
    assert len(source_mesh.verts) == num_verts_exp

    remove_files()


def test_mesh_generation(source_mesh):
    """Tests whether SourceMesh construction functions as expected, by testing
    if:
        * the correct number of mesh elements are produced
        * no elements with negative volume are created
    """
    remove_files()

    num_s = 6
    num_theta = 41
    num_phi = 9

    tets_per_wedge = 3
    tets_per_hex = 5

    num_elements_exp = tets_per_wedge * (num_theta - 1) * (
        num_phi - 1
    ) + tets_per_hex * (num_s - 2) * (num_theta - 1) * (num_phi - 1)
    num_neg_vols_exp = 0

    source_mesh.create_vertices()
    source_mesh.create_mesh()

    assert len(source_mesh.volumes) == num_elements_exp
    assert len([i for i in source_mesh.volumes if i < 0]) == num_neg_vols_exp

    remove_files()


def test_export(source_mesh):
    """Tests whether SourceMesh's export functionality behaves as expected, by
    testing if:
        * the expected H5M file is produced
    """
    filename_exp = "source_mesh.h5m"

    remove_files()

    source_mesh.create_vertices()
    source_mesh.create_mesh()
    source_mesh.export_mesh(filename_exp)

    assert Path(filename_exp).exists()

    remove_files()
