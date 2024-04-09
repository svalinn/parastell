from pathlib import Path

import numpy as np
import pytest

import parastell.parastell as ps


def remove_files():

    if Path('plasma.step').exists():
        Path.unlink('plasma.step')
    if Path('sol.step').exists():
        Path.unlink('sol.step')
    if Path('component.step').exists():
        Path.unlink('component.step')
    if Path('magnets.step').exists():
        Path.unlink('magnets.step')
    if Path('magnet_mesh.exo').exists():
        Path.unlink('magnet_mesh.exo')
    if Path('magnet_mesh.h5m').exists():
        Path.unlink('magnet_mesh.h5m')
    if Path('dagmc.h5m').exists():
        Path.unlink('dagmc.h5m')
    if Path('source_mesh.h5m').exists():
        Path.unlink('source_mesh.h5m')
    if Path('step_import.log').exists():
        Path.unlink('step_import.log')
    if Path('step_export.log').exists():
        Path.unlink('step_export.log')


@pytest.fixture
def stellarator():
    
    vmec_file = Path('files_for_tests') / 'wout_vmec.nc'

    stellarator_obj = ps.Stellarator(vmec_file)

    return stellarator_obj


def test_parastell(stellarator):

    toroidal_angles = [0.0, 5.0, 10.0, 15.0]
    poloidal_angles = [0.0, 120.0, 240.0, 360.0]

    invessel_build = {
        'toroidal_angles': toroidal_angles,
        'poloidal_angles': poloidal_angles,
        'radial_build': {
            'component': {
                'thickness_matrix': np.ones(
                    (len(toroidal_angles), len(poloidal_angles))
                )*10
            }
        },
        'wall_s': 1.08,
        'repeat': 0,
        'num_ribs': 11,
        'num_rib_pts': 67
    }

    magnets = {
        'coils_file_path': Path('files_for_tests') / 'coils.example',
        'start_line': 3,
        'cross_section': ['circle', 20],
        'toroidal_extent': 90.0,
        'sample_mod': 6,
        'export_mesh': True
    }

    source = {
        'num_s': 4,
        'num_theta': 8,
        'num_phi': 4,
        'toroidal_extent': 90.0
    }

    remove_files()

    stellarator.construct_invessel_build(invessel_build)
    stellarator.export_invessel_build(invessel_build)
    assert Path('plasma.step').exists()
    assert Path('sol.step').exists()
    assert Path('component.step').exists()

    stellarator.construct_magnets(magnets)
    stellarator.export_magnets(magnets)
    assert Path('magnets.step').exists()
    assert Path('magnet_mesh.h5m').exists()

    stellarator.construct_source_mesh(source)
    stellarator.export_source_mesh(source)
    assert Path('source_mesh.h5m').exists()

    stellarator.export_dagmc()
    assert Path('dagmc.h5m').exists()

    remove_files()
