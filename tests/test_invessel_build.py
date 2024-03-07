import src.invessel_build as ivb
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

invessel_build = ivb.InVesselBuild(
    vmec, build, repeat, num_phi, num_theta, scale, plasma_h5m_tag, sol_h5m_tag
)


def test_ivb_basics():

    invessel_build.populate_surfaces()

    assert np.allclose(invessel_build.build['phi_list'], build['phi_list'])
    assert np.allclose(invessel_build.build['theta_list'], build['theta_list'])
    assert invessel_build.build['wall_s'] == build['wall_s']
    assert len(invessel_build.build['radial_build'].keys()) == 3
    assert (
        invessel_build.build['radial_build']['plasma']['h5m_tag'] == 'Vacuum'
    )
    assert (
        invessel_build.build['radial_build']['sol']['h5m_tag'] == 'Vacuum'
    )
    assert invessel_build.repeat == repeat
    assert invessel_build.num_phi == num_phi
    assert invessel_build.num_theta == num_theta
    assert invessel_build.scale == scale


def test_ivb_construction():

    invessel_build.calculate_loci()
    invessel_build.generate_components()

    assert len(invessel_build.Components) == 3

Path.unlink('stellarator.log')
