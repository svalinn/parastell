import src.parastell as ps
import numpy as np


# Define plasma equilibrium VMEC file
vmec_file = 'wout_vmec.nc'

# Define toroidal angles at which radial build is specified.
# Note that the initial toroidal angle (phi) must be zero
# Also note that it is generally not advised to have the toroidal extent to
# extend beyond one stellarator period
# To build a geometry extending beyond one period, make use of the 'repeat'
# parameter
phi_list = [0.0, 11.25, 22.5, 33.75, 45.0, 56.25, 67.5, 78.75, 90.0]
# Define poloidal angles at which radial builds is specified.
# Note that this should always span 360 degrees.
theta_list = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0, 360.0]

# Define radial build
# For each component, thickness matrices have rows corresponding to toroidal
# angles (phi_list) and columns corresponding to poloidal angles (theta_list)
build = {
    'phi_list': phi_list,
    'theta_list': theta_list,
    'wall_s': 1.08,
    'radial_build': {
        'first_wall': {
            'thickness_matrix': np.ones((len(phi_list), len(theta_list)))*5
        },
        'breeder': {
            'thickness_matrix': ([
                [75.0, 75.0, 75.0, 25.0, 25.0, 25.0, 75.0, 75.0, 75.0],
                [75.0, 75.0, 75.0, 25.0, 25.0, 75.0, 75.0, 75.0, 75.0],
                [75.0, 75.0, 25.0, 25.0, 75.0, 75.0, 75.0, 75.0, 75.0],
                [65.0, 25.0, 25.0, 65.0, 75.0, 75.0, 75.0, 75.0, 65.0],
                [45.0, 45.0, 75.0, 75.0, 75.0, 75.0, 75.0, 45.0, 45.0],
                [65.0, 75.0, 75.0, 75.0, 75.0, 65.0, 25.0, 25.0, 65.0],
                [75.0, 75.0, 75.0, 75.0, 75.0, 25.0, 25.0, 75.0, 75.0],
                [75.0, 75.0, 75.0, 75.0, 25.0, 25.0, 75.0, 75.0, 75.0],
                [75.0, 75.0, 75.0, 25.0, 25.0, 25.0, 75.0, 75.0, 75.0]
            ])
        },
        'back_wall': {
            'thickness_matrix': np.ones((len(phi_list), len(theta_list)))*5
        },
        'shield': {
            'thickness_matrix': np.ones((len(phi_list), len(theta_list)))*50
        },
        # Note that some neutron transport codes (such as OpenMC) will interpret
        # materials with "vacuum" in the name as void material
        'vacuum_vessel': {
            'thickness_matrix': np.ones((len(phi_list), len(theta_list)))*10,
            'h5m_tag': 'vac_vessel'
        }
    }
}
# Define number of times to repeat build
repeat = 0
# Define number of toroidal cross-sections to make
num_phi = 61
# Define number of poloidal points to include in each toroidal cross-section
num_theta = 61
# Define scaling factor converting VMEC data units to cm
scale = 100
# Define magnet coil parameters
magnets = {
    'file': 'coils.txt',
    'cross_section': ['circle', 25],
    'start': 3,
    'stop': None,
    'sample': 1,
    'name': 'magnet_coils',
    'h5m_tag': 'magnets',
    'meshing': True
}
# Define source mesh parameters
source = {
    'num_s': 11,
    'num_theta': 81,
    'num_phi': 61,
    'tor_ext': 90.0
}
# Define export parameters
export = {
    'exclude': [],
    'graveyard': False,
    'dir': '',
    'step_export': True,
    'h5m_export': 'cubit',
    'h5m_filename': 'dagmc',
    'plas_h5m_tag': 'Vacuum',
    'sol_h5m_tag': 'Vacuum',
    # Note the following export parameters are used only for Cubit H5M exports
    'facet_tol': 1,
    'len_tol': 5,
    'norm_tol': None,
    # Choose whether to use native Cubit meshing (v2023.11+) or legacy DAGMC
    # workflow
    'native_meshing': False,
    'anisotropic_ratio': 100,
    'deviation_angle': 5,
    # Note the following export parameters are used only for Gmsh H5M exports
    'min_mesh_size': 5.0,
    'max_mesh_size': 20.0,
    'volume_atol': 0.00001,
    'center_atol': 0.00001,
    'bounding_box_atol': 0.00001
}

# Create stellarator
stellarator = ps.Stellarator(
    vmec_file, build, repeat, num_phi, num_theta, scale,
    magnets = magnets, source = source, export = export
)
stellarator.construct_invessel_build()
stellarator.construct_magnets()
stellarator.export_CAD_geometry()
stellarator.construct_source_mesh()
