from pathlib import Path

import numpy as np

import src.parastell as ps


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

vmec_file = Path('files_for_tests') / 'wout_vmec.nc'

toroidal_angles = [0.0, 30.0, 60.0, 90.0]
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
    'num_ribs': 61,
    'num_rib_pts': 61
}

magnets = {
    'coils_file_path': Path('files_for_tests') / 'coils.txt',
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

stellarator = ps.Stellarator(vmec_file)


def test_parastell():

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

    Path.unlink('plasma.step')
    Path.unlink('sol.step')
    Path.unlink('component.step')
    Path.unlink('magnets.step')
    Path.unlink('magnet_mesh.h5m')
    Path.unlink('dagmc.h5m')
    Path.unlink('source_mesh.h5m')
    Path.unlink('stellarator.log')
    Path.unlink('step_import.log')
    Path.unlink('step_export.log')

test_parastell()
