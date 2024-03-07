import src.parastell as ps
import numpy as np
from pathlib import Path


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
    vmec_file, build, repeat, num_phi, num_theta, scale, export=export_dict
)

def test_export():

    stellarator.construct_invessel_build()
    stellarator.export_CAD_geometry()
    
    assert Path('plasma.step').exists() == True
    assert Path('sol.step').exists() == True
    assert Path('component.step').exists() == True
    assert Path('dagmc.h5m').exists() == True

    Path.unlink('plasma.step')
    Path.unlink('sol.step')
    Path.unlink('component.step')
    Path.unlink('dagmc.h5m')
    Path.unlink('stellarator.log')
    Path.unlink('step_import.log')

test_export()
