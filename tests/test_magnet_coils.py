from pathlib import Path

import pytest
import numpy as np

import parastell.magnet_coils as magnet_coils


def remove_files():

    if Path("magnet_set.step").exists():
        Path.unlink("magnet_set.step")
    if Path("magnet_mesh.exo").exists():
        Path.unlink("magnet_mesh.exo")
    if Path("magnet_mesh.h5m").exists():
        Path.unlink("magnet_mesh.h5m")
    if Path("stellarator.log").exists():
        Path.unlink("stellarator.log")
    if Path("step_import.log").exists():
        Path.unlink("step_import.log")


@pytest.fixture
def coil_set():

    coils_file = Path("files_for_tests") / "coils.example"
    width = 40.0
    thickness = 50.0
    toroidal_extent = 90.0
    sample_mod = 10

    coil_set_obj = magnet_coils.MagnetSet(
        coils_file, width, thickness, toroidal_extent, sample_mod=sample_mod
    )

    return coil_set_obj


def test_magnet_construction(coil_set):

    width_exp = 40.0
    thickness_exp = 50.0
    toroidal_extent_exp = np.deg2rad(90.0)
    max_cs_len_exp = 50.0
    average_radial_distance_exp = 1023.7170384211436
    max_radial_distance_exp = 1646.3258131460148
    len_coils_exp = 1
    len_coords_exp = 129

    remove_files()

    coil_set.build_magnet_coils()

    assert len(coil_set.magnet_coils) == len_coils_exp
    assert coil_set.width == width_exp
    assert coil_set.thickness == thickness_exp
    assert coil_set.toroidal_extent == toroidal_extent_exp
    assert coil_set.max_cs_len == max_cs_len_exp
    assert coil_set.average_radial_distance == average_radial_distance_exp
    assert coil_set.max_radial_distance == max_radial_distance_exp

    test_coil = coil_set.magnet_coils[0]
    assert len(test_coil.coords) == len_coords_exp

    remove_files()


def test_magnet_exports(coil_set):

    remove_files()

    coil_set.build_magnet_coils()
    coil_set.export_step()
    assert Path("magnet_set.step").exists()

    coil_set.mesh_magnets()
    coil_set.export_mesh()
    assert Path("magnet_mesh.h5m").exists()

    remove_files()
