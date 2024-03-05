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
    'plas_h5m_tag': 'Vacuum',
    'sol_h5m_tag': 'Vacuum'
}

stellarator = ps.Stellarator(
    vmec_file, build, repeat, num_phi, num_theta, scale, export=export_dict
)


def test_ivc_data():

    stellarator.populate_data()

    assert np.allclose(
        stellarator.data.phi_list, np.deg2rad(build['phi_list'])
    )
    assert np.allclose(
        stellarator.data.theta_list, np.deg2rad(build['theta_list'])
    )
    assert stellarator.data.wall_s == build['wall_s']
    assert stellarator.data.repeat == repeat
    assert stellarator.data.num_phi == num_phi
    assert stellarator.data.num_theta == num_theta
    assert stellarator.data.scale == scale
    assert stellarator.data.seg_tor_ext == np.deg2rad(phi_list[-1])
    assert stellarator.data.tot_tor_ext == (
        repeat + 1)*np.deg2rad(phi_list[-1])
    assert len(stellarator.data.radial_build.keys()) == 3


def test_ivc_geom():

    stellarator.construct_geometry()

    assert stellarator.components['plasma']['h5m_tag'] == 'Vacuum'
    assert stellarator.components['plasma']['solid'] is not None
    assert stellarator.components['sol']['h5m_tag'] == 'Vacuum'
    assert stellarator.components['sol']['solid'] is not None
    assert stellarator.components['component']['h5m_tag'] == 'component'
    assert stellarator.components['component']['solid'] is not None


def test_ivc_export():

    stellarator.export_CAD_geometry()
    
    assert Path('plasma.step').exists() == True
    assert Path('sol.step').exists() == True
    assert Path('component.step').exists() == True
    assert Path('dagmc.h5m').exists() == True

    Path.unlink('plasma.step')
    Path.unlink('sol.step')
    Path.unlink('component.step')
    Path.unlink('dagmc.h5m')
