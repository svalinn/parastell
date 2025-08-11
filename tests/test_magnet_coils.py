from pathlib import Path

import pytest
import numpy as np
import cadquery as cq

import parastell.magnet_coils as magnet_coils
from parastell.cubit_utils import (
    check_cubit_installation,
    create_new_cubit_instance,
)

files_to_remove = [
    "magnet_set.step",
    "magnet_mesh.exo",
    "magnet_mesh.h5m",
    "stellarator.log",
    "step_import.log",
    "single_coil.step",
]


def remove_files():
    for file in files_to_remove:
        if Path(file).exists():
            Path.unlink(file)


@pytest.fixture
def coil_set_from_filaments():
    coils_file = Path("files_for_tests") / "coils.example"
    width = 40.0
    thickness = 50.0
    toroidal_extent = 90.0
    sample_mod = 10

    coil_set_obj = magnet_coils.MagnetSetFromFilaments(
        coils_file, width, thickness, toroidal_extent, sample_mod=sample_mod
    )

    return coil_set_obj


@pytest.fixture
def coil_set_from_geometry(geometry_file):
    geom_file = Path("files_for_tests") / Path(geometry_file).with_suffix(
        ".step"
    )

    if "with_casing" in str(geometry_file):
        coil_set_obj = magnet_coils.MagnetSetFromGeometry(
            geom_file, mat_tag=["mat1", "mat2"], volume_ids=[(0, 1), (2, 3)]
        )
    else:
        coil_set_obj = magnet_coils.MagnetSetFromGeometry(
            geom_file, mat_tag="mat1"
        )

    return coil_set_obj


@pytest.fixture
def filament_crossing_mp():
    coords = np.array(
        [
            [100, 0, 0],
            [75, 0, 25],
            [50, 0, 0],
            [75, 0, -25],
            [100, 0, 0],
        ]
    )
    return magnet_coils.Filament(coords)


@pytest.fixture
def filament_not_crossing_mp():
    coords = np.array(
        [
            [0, 0, 0],
            [200, 300, 0],
            [0, 0, 100],
            [0, 0, 0],
        ]
    )
    return magnet_coils.Filament(coords)


@pytest.fixture
def single_coil(filament_crossing_mp):
    return magnet_coils.MagnetCoil(filament_crossing_mp, 10, 20, 0, 1)


def test_filament_crossing_mp(filament_crossing_mp):
    """Tests whether the data for a Filament object that crosses the midplane
    is generated as expected, by testing if:
        * the expected tangent vectors are computed
        * the expected center of mass is computed
        * the expected center of mass toroidal angle is computed
        * the expected outboard midplane index is computed
    """
    tangents_exp = np.array(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    com_exp = np.array([75.0, 0.0, 0.0])
    com_toroidal_angle_exp = 0.0
    ob_mp_idx_exp = 0

    assert np.allclose(tangents_exp, filament_crossing_mp.tangents)
    assert np.allclose(com_exp, filament_crossing_mp.com)
    assert np.isclose(
        com_toroidal_angle_exp, filament_crossing_mp.com_toroidal_angle
    )
    assert filament_crossing_mp.get_ob_mp_index() == ob_mp_idx_exp


def test_filament_not_crossing_mp(filament_not_crossing_mp):
    """Tests whether the data for a Filament object that does not cross the
    midplane is generated as expected, by testing if:
        * the expected tangent vectors are computed
        * the expected center of mass is computed
        * the expected center of mass toroidal angle is computed
        * the expected outboard midplane index is computed
    """
    tangents_exp = np.array(
        [
            [0.53452248, 0.80178373, -0.26726124],
            [0.0, 0.0, 1.0],
            [-0.5547002, -0.83205029, 0.0],
            [0.53452248, 0.80178373, -0.26726124],
        ]
    )

    com_exp = np.array([66.66666667, 100.0, 33.33333333])
    com_toroidal_angle_exp = 0.982793723247329
    ob_mp_idx_exp = 0

    assert np.allclose(tangents_exp, filament_not_crossing_mp.tangents)
    assert np.allclose(com_exp, filament_not_crossing_mp.com)
    assert np.isclose(
        com_toroidal_angle_exp, filament_not_crossing_mp.com_toroidal_angle
    )
    assert filament_not_crossing_mp.get_ob_mp_index() == ob_mp_idx_exp


def test_single_coil(single_coil):
    """Tests whether a MagnetCoil object can be generated with valid CAD, by
    testing if:
        * the expected STEP file is produced
    """
    remove_files()

    single_coil.create_magnet()
    cq.exporters.export(single_coil.solids[0], "single_coil.step")
    assert Path("single_coil.step").exists()

    remove_files()


@pytest.mark.parametrize("case_thickness", [0.0, 5.0])
def test_magnet_construction(coil_set_from_filaments, case_thickness):
    """Tests whether the MagnetSetFromFilaments object is instantiated and
    constructed as expected, along with relevant data, by testing if:
        * after being set, member variables match inputs
        * the expected coil properties are computed
        * the expected number of coils are built
        * the built coil has the correct number of points
    """
    remove_files()

    width_exp = 40.0
    thickness_exp = 50.0
    toroidal_extent_exp = np.deg2rad(90.0)
    max_cs_len_exp = 50.0
    average_radial_distance_exp = 1023.7170384211436
    max_radial_distance_exp = 1646.3258131460148
    len_coils_exp = 1
    len_coords_exp = 129

    if case_thickness == 0.0:
        num_solids_exp = 1
    else:
        num_solids_exp = 2

    case_thickness_exp = case_thickness

    coil_set_from_filaments.case_thickness = case_thickness

    coil_set_from_filaments.populate_magnet_coils()
    coil_set_from_filaments.build_magnet_coils()

    assert coil_set_from_filaments.width == width_exp
    assert coil_set_from_filaments.thickness == thickness_exp
    assert coil_set_from_filaments.toroidal_extent == toroidal_extent_exp
    assert coil_set_from_filaments.case_thickness == case_thickness_exp
    assert coil_set_from_filaments.max_cs_len == max_cs_len_exp
    assert (
        coil_set_from_filaments.average_radial_distance
        == average_radial_distance_exp
    )
    assert (
        coil_set_from_filaments.max_radial_distance == max_radial_distance_exp
    )
    assert len(coil_set_from_filaments.magnet_coils) == len_coils_exp

    test_coil = coil_set_from_filaments.magnet_coils[0]
    assert len(test_coil.coords) == len_coords_exp

    assert (
        len(
            [
                solid
                for solids in coil_set_from_filaments.coil_solids
                for solid in solids
            ]
        )
        == num_solids_exp
    )

    remove_files()


@pytest.mark.parametrize("case_thickness", [0.0, 5.0])
def test_magnet_exports_from_filaments(
    coil_set_from_filaments, case_thickness
):
    """Tests whether the MagnetSetFromFilaments' export functionality behaves
    as expected, by testing if:
        * the expected STEP file is produced
        * if Cubit is correctly installed, the correct volume IDs are stored
        * if Cubit is correctly installed, the expected H5M file is produced

    The Cubit-enabled portion of this test is skipped if Cubit cannot be
    imported.
    """
    remove_files()

    if case_thickness == 0.0:
        volume_ids_exp = [[1]]
    else:
        volume_ids_exp = [[1, 2]]

    coil_set_from_filaments.case_thickness = case_thickness

    coil_set_from_filaments.populate_magnet_coils()
    coil_set_from_filaments.build_magnet_coils()
    coil_set_from_filaments.export_step()
    assert Path("magnet_set.step").exists()

    if check_cubit_installation():
        create_new_cubit_instance()

        coil_set_from_filaments.mesh_magnets_cubit()
        assert np.allclose(
            coil_set_from_filaments.cubit_volume_ids, volume_ids_exp
        )

        coil_set_from_filaments.export_mesh_cubit()
        assert Path("magnet_mesh.h5m").exists()

        remove_files()

    coil_set_from_filaments.mesh_magnets_gmsh()
    coil_set_from_filaments.export_mesh_gmsh()
    assert Path("magnet_mesh.h5m").exists()

    remove_files()


@pytest.mark.parametrize(
    "geometry_file", ["magnet_geom", "magnet_geom_with_casing"]
)
def test_magnet_exports_from_geometry(coil_set_from_geometry):
    """Tests whether the MagnetSetFromGeometry's export functionality behaves
    as expected, by testing if:
        * the expected number of solids are present in coil_set_from_geometry
        * the correct volume IDs are stored
        * the expected H5M file is produced

    This test is skipped if Cubit cannot be imported.
    """
    num_coil_solids_exp = 2

    if "with_casing" in str(coil_set_from_geometry.geometry_file):
        num_total_solids_exp = 4
        volume_ids_exp = [[0, 1], [2, 3]]
        cubit_volume_ids_exp = [[1, 2], [3, 4]]
    else:
        num_total_solids_exp = 2
        volume_ids_exp = [[0], [1]]
        cubit_volume_ids_exp = [[1], [2]]

    remove_files()

    assert len(coil_set_from_geometry.coil_solids) == num_coil_solids_exp
    assert len(coil_set_from_geometry.all_coil_solids) == num_total_solids_exp
    assert np.allclose(coil_set_from_geometry.volume_ids, volume_ids_exp)

    if check_cubit_installation():
        create_new_cubit_instance()

        coil_set_from_geometry.mesh_magnets_cubit()
        assert np.allclose(
            coil_set_from_geometry.cubit_volume_ids, cubit_volume_ids_exp
        )

        coil_set_from_geometry.export_mesh_cubit()
        assert Path("magnet_mesh.h5m").exists()

        remove_files()

    coil_set_from_geometry.mesh_magnets_gmsh()
    coil_set_from_geometry.export_mesh_gmsh()
    assert Path("magnet_mesh.h5m").exists()

    remove_files()


def test_zero_volume_volumes():
    """Tests to make sure zero volume coil solids are not being added to
    MagnetSetFromFilament.coil_solids by testing if:
        * Any coil solids in the specially designed test set are zero
    """
    test_filaments = "files_for_tests/circular_coils.example"
    ms = magnet_coils.MagnetSetFromFilaments(test_filaments, 40, 40, 89.8)
    ms.populate_magnet_coils()
    ms.build_magnet_coils()
    for solids in ms.coil_solids:
        assert not np.isclose(solids[0].Volume(), 0)
