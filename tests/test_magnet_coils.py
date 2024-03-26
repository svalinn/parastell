import src.magnet_coils as magnet_coils
from pathlib import Path

coils_file_path = Path('files_for_tests') / 'coils.txt'
start_line = 3
rect_cross_section = ['rectangle', 20, 60]
circ_cross_section = ['circle', 25]
toroidal_extent = 90.0
sample_mod = 6
scale = 100
mat_tag = 'magnets'

rect_coil_set = magnet_coils.MagnetSet(
    coils_file_path, start_line, rect_cross_section, toroidal_extent,
    sample_mod=sample_mod, scale=scale, mat_tag=mat_tag
)

circ_coil_set = magnet_coils.MagnetSet(
    coils_file_path, start_line, circ_cross_section, toroidal_extent,
    sample_mod=sample_mod, scale=scale, mat_tag=mat_tag
)


def test_rectangular_magnets():
    
    shape_exp = 'rectangle'
    shape_str_exp = 'rectangle width 60 height 20'
    mag_len_exp = 60

    shape = rect_coil_set.shape
    shape_str = rect_coil_set.shape_str
    mag_len = rect_coil_set.mag_len

    assert shape == shape_exp
    assert shape_str == shape_str_exp
    assert mag_len == mag_len_exp


def test_circular_magnets():
    
    len_filaments_exp = 40
    average_radial_distance_exp = 1241.4516792609722
    len_filtered_filaments_exp = 18
    shape_exp = 'circle'
    shape_str_exp = 'circle radius 25'
    mag_len_exp = 25
    len_test_coil_filament_exp = 23

    filaments = circ_coil_set.filaments
    average_radial_distance = circ_coil_set.average_radial_distance
    filtered_filaments = circ_coil_set.filtered_filaments
    shape = circ_coil_set.shape
    shape_str = circ_coil_set.shape_str
    mag_len = circ_coil_set.mag_len

    circ_coil_set.build_magnet_coils()

    test_coil = circ_coil_set.magnet_coils[0]
    test_coil_filament = test_coil.filament

    assert len(filaments) == len_filaments_exp
    assert average_radial_distance == average_radial_distance_exp
    assert len(filtered_filaments) == len_filtered_filaments_exp
    assert shape == shape_exp
    assert shape_str == shape_str_exp
    assert mag_len == mag_len_exp
    assert len(test_coil_filament) == len_test_coil_filament_exp


def test_magnet_exports():

    if Path('magnet_mesh.exo').exists():
        Path.unlink('magnet_mesh.exo')

    if Path('magnet_mesh.h5m').exists():
        Path.unlink('magnet_mesh.h5m')

    circ_coil_set.build_magnet_coils()
    circ_coil_set.export_step()
    circ_coil_set.mesh_magnets()
    circ_coil_set.export_mesh()

    assert Path('magnets.step').exists() == True
    assert Path('magnet_mesh.h5m').exists() == True

    Path.unlink('magnets.step')
    Path.unlink('magnet_mesh.h5m')
    Path.unlink('stellarator.log')
    Path.unlink('step_export.log')
