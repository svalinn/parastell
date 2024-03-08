import src.parastell as ps
import numpy as np
from pathlib import Path


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
    Path.unlink('SourceMesh.h5m')

vmec_file = Path('files_for_tests') / 'wout_vmec.nc'

phi_list = [0.0, 30.0, 60.0, 90.0]
theta_list = [0.0, 120.0, 240.0, 360.0]

build = {
    'phi_list': phi_list,
    'theta_list': theta_list,
    'wall_s': 1.08,
    'radial_build': {
        'component': {
            'thickness_matrix': np.ones((len(phi_list), len(theta_list)))*10
        }
    }
}

repeat = 0
num_phi = 61
num_theta = 61
scale = 100

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

source = {
    'num_s': 4,
    'num_theta': 8,
    'num_phi': 4,
    'tor_ext': 90.0
}

export_dict = {
    'exclude': [],
    'graveyard': False,
    'dir': '',
    'step_export': True,
    'h5m_export': 'cubit',
    'h5m_filename': 'dagmc',
    'plasma_h5m_tag': 'Vacuum',
    'sol_h5m_tag': 'Vacuum'
}

stellarator = ps.Stellarator(
    vmec_file, build, repeat, num_phi, num_theta, scale, magnets=magnets,
    source=source, export=export_dict
)


def test_export():

    stellarator.construct_invessel_build()
    stellarator.construct_magnets()
    stellarator.export_CAD_geometry()
    stellarator.construct_source_mesh()

    assert Path('plasma.step').exists() == True
    assert Path('sol.step').exists() == True
    assert Path('component.step').exists() == True
    assert Path('magnets.step').exists() == True
    assert Path('magnet_mesh.h5m').exists() == True
    assert Path('dagmc.h5m').exists() == True
    assert Path('source_mesh.h5m').exists() == True

    Path.unlink('plasma.step')
    Path.unlink('sol.step')
    Path.unlink('component.step')
    Path.unlink('magnets.step')
    Path.unlink('magnet_mesh.exo')
    Path.unlink('magnet_mesh.h5m')
    Path.unlink('dagmc.h5m')
    Path.unlink('source_mesh.h5m')
    Path.unlink('stellarator.log')
    Path.unlink('step_import.log')
    Path.unlink('step_export.log')
