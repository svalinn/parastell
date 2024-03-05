import src.ivc as ivc
import read_vmec
import numpy as np
from pathlib import Path

vmec_file = Path('files_for_tests') / 'wout_vmec.nc'
vmec = read_vmec.vmec_data(vmec_file)

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
plasma_h5m_tag = 'Vacuum'
sol_h5m_tag = 'Vacuum'

invessel_comps = ivc.IVC(
    vmec, build, repeat, num_phi, num_theta, scale, plasma_h5m_tag, sol_h5m_tag
)


def test_ivc_data():

    invessel_comps.populate_data()

    assert np.allclose(
        invessel_comps.data.phi_list, np.deg2rad(build['phi_list'])
    )
    assert np.allclose(
        invessel_comps.data.theta_list, np.deg2rad(build['theta_list'])
    )
    assert invessel_comps.data.wall_s == build['wall_s']
    assert invessel_comps.data.repeat == repeat
    assert invessel_comps.data.num_phi == num_phi
    assert invessel_comps.data.num_theta == num_theta
    assert invessel_comps.data.scale == scale
    assert invessel_comps.data.seg_tor_ext == np.deg2rad(phi_list[-1])
    assert invessel_comps.data.tot_tor_ext == (
        repeat + 1)*np.deg2rad(phi_list[-1])
    assert len(invessel_comps.data.radial_build.keys()) == 3


def test_ivc_geom():

    invessel_comps.construct_geometry()

    assert invessel_comps.components['plasma']['h5m_tag'] == 'Vacuum'
    assert invessel_comps.components['plasma']['solid'] is not None
    assert invessel_comps.components['sol']['h5m_tag'] == 'Vacuum'
    assert invessel_comps.components['sol']['solid'] is not None
    assert invessel_comps.components['component']['h5m_tag'] == 'component'
    assert invessel_comps.components['component']['solid'] is not None

Path.unlink('stellarator.log')
