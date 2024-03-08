import src.magnet_coils as magnet_coils
from pathlib import Path

toroidal_extent = 90
export_dir = ''


def test_rectangular_magnets():

    magnets = {
        'file': Path('files_for_tests') / 'coils.txt',
        'cross_section': ['rectangle', 20, 60],
        'start': 3,
        'stop': None,
        'sample': 6,
        'name': 'magnets',
        'h5m_tag': 'magnets',
        'meshing': False
    }

    len_filaments_exp = 40
    average_radial_distance_exp = 1241.4516792609722
    len_filtered_filaments_exp = 18

    shape_exp = 'rectangle'
    shape_str_exp = 'rectangle width 60 height 20'
    mag_len_exp = 60

    len_test_coil_filament_exp = 23

    test_coil_set = magnet_coils.MagnetSet(
        magnets, toroidal_extent, export_dir
    )

    filaments = test_coil_set.filaments
    average_radial_distance = test_coil_set.average_radial_distance
    filtered_filaments = test_coil_set.filtered_filaments
    shape = test_coil_set.shape
    shape_str = test_coil_set.shape_str
    mag_len = test_coil_set.mag_len

    test_coil = test_coil_set.create_magnet_coils()[0]
    test_coil_filament = test_coil.filament

    assert len(filaments) == len_filaments_exp
    assert average_radial_distance == average_radial_distance_exp
    assert len(filtered_filaments) == len_filtered_filaments_exp
    assert shape == shape_exp
    assert shape_str == shape_str_exp
    assert mag_len == mag_len_exp
    assert len(test_coil_filament) == len_test_coil_filament_exp


def test_circular_magnets():

    magnets = {
        'file': Path('files_for_tests') / 'coils.txt',
        'cross_section': ['circle', 20],
        'start': 3,
        'stop': None,
        'sample': 6,
        'name': 'magnets',
        'h5m_tag': 'magnets',
        'meshing': False
    }

    len_filaments_exp = 40
    average_radial_distance_exp = 1241.4516792609722
    len_filtered_filaments_exp = 18

    shape_exp = 'circle'
    shape_str_exp = 'circle radius 20'
    mag_len_exp = 20

    len_test_coil_filament_exp = 23

    test_coil_set = magnet_coils.MagnetSet(
        magnets, toroidal_extent, export_dir
    )

    filaments = test_coil_set.filaments
    average_radial_distance = test_coil_set.average_radial_distance
    filtered_filaments = test_coil_set.filtered_filaments
    shape = test_coil_set.shape
    shape_str = test_coil_set.shape_str
    mag_len = test_coil_set.mag_len

    test_coil = test_coil_set.create_magnet_coils()[0]
    test_coil_filament = test_coil.filament

    assert len(filaments) == len_filaments_exp
    assert average_radial_distance == average_radial_distance_exp
    assert len(filtered_filaments) == len_filtered_filaments_exp
    assert shape == shape_exp
    assert shape_str == shape_str_exp
    assert mag_len == mag_len_exp
    assert len(test_coil_filament) == len_test_coil_filament_exp


def test_magnet_meshing():

    if Path('magnet_mesh.exo').exists():
        Path.unlink('magnet_mesh.exo')

    if Path('magnet_mesh.h5m').exists():
        Path.unlink('magnet_mesh.h5m')

    magnets = {
        'file': Path('files_for_tests') / 'coils.txt',
        'cross_section': ['circle', 20],
        'start': 3,
        'stop': None,
        'sample': 6,
        'name': 'magnets',
        'h5m_tag': 'magnets',
        'meshing': True
    }

    test_coil_set = magnet_coils.MagnetSet(
        magnets, toroidal_extent, export_dir
    )

    test_coil_set.build_magnet_coils()
    test_coil_set.mesh_magnets()

    assert Path('magnet_mesh.exo').exists() == True
    assert Path('magnet_mesh.h5m').exists() == True

    Path.unlink('magnet_mesh.exo')
    Path.unlink('magnet_mesh.h5m')
    Path.unlink('stellarator.log')
