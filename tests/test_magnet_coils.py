from pathlib import Path

import pytest

import parastell.magnet_coils as magnet_coils


def remove_files():

    if Path("magnets.step").exists():
        Path.unlink("magnets.step")
    if Path("magnet_mesh.exo").exists():
        Path.unlink("magnet_mesh.exo")
    if Path("magnet_mesh.h5m").exists():
        Path.unlink("magnet_mesh.h5m")
    if Path("stellarator.log").exists():
        Path.unlink("stellarator.log")
    if Path("step_export.log").exists():
        Path.unlink("step_export.log")


@pytest.fixture
def rect_coil_set():

    coils_file = Path("files_for_tests") / "coils.example"
    rect_cross_section = ["rectangle", 20, 60]
    toroidal_extent = 90.0
    sample_mod = 6

    rect_coil_obj = magnet_coils.MagnetSet(
        coils_file, rect_cross_section, toroidal_extent, sample_mod=sample_mod
    )

    return rect_coil_obj


@pytest.fixture
def circ_coil_set():

    coils_file = Path("files_for_tests") / "coils.example"
    circ_cross_section = ["circle", 25]
    toroidal_extent = 90.0
    sample_mod = 6

    circ_coil_obj = magnet_coils.MagnetSet(
        coils_file, circ_cross_section, toroidal_extent, sample_mod=sample_mod
    )

    return circ_coil_obj


def test_rectangular_magnets(rect_coil_set):

    shape_exp = "rectangle"
    shape_str_exp = "rectangle width 60 height 20"
    mag_len_exp = 60

    remove_files()

    assert rect_coil_set.shape == shape_exp
    assert rect_coil_set.shape_str == shape_str_exp
    assert rect_coil_set.mag_len == mag_len_exp

    remove_files()


def test_circular_magnets(circ_coil_set):

    len_filaments_exp = 2
    average_radial_distance_exp = 1068.3010006669892
    len_filtered_filaments_exp = 1
    shape_exp = "circle"
    shape_str_exp = "circle radius 25"
    mag_len_exp = 25
    len_test_coil_filament_exp = 23

    remove_files()

    circ_coil_set.build_magnet_coils()

    assert len(circ_coil_set.filaments) == len_filaments_exp
    assert circ_coil_set.average_radial_distance == average_radial_distance_exp
    assert len(circ_coil_set.filtered_filaments) == len_filtered_filaments_exp
    assert circ_coil_set.shape == shape_exp
    assert circ_coil_set.shape_str == shape_str_exp
    assert circ_coil_set.mag_len == mag_len_exp

    test_coil = circ_coil_set.magnet_coils[0]
    test_coil_filament = test_coil.filament
    assert len(test_coil_filament) == len_test_coil_filament_exp

    remove_files()


def test_magnet_exports(circ_coil_set):

    remove_files()

    circ_coil_set.build_magnet_coils()
    circ_coil_set.export_step()
    assert Path("magnets.step").exists()

    circ_coil_set.mesh_magnets()
    circ_coil_set.export_mesh()
    assert Path("magnet_mesh.h5m").exists()

    remove_files()
