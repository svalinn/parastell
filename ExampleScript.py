import parastell


# Define plasma equilibrium VMEC file
plas_eq = 'plas_eq.nc'
# Define radial build
build = {
    'phi_list': [0.0, 22.5, 45.0, 67.5, 90.0],
    'theta_list': [0.0, 5.0, 90.0, 175.0, 180.0, 185.0, 270.0, 355.0, 360.0],
    'wall_s': 1.2,
    'radial_build': {
        'first_wall': {
            'thickness_matrix': [
                [5, 5, 5, 5, 5, 5, 5, 5, 5],
                [5, 5, 5, 5, 5, 5, 5, 5, 5],
                [5, 5, 5, 5, 5, 5, 5, 5, 5],
                [5, 5, 5, 5, 5, 5, 5, 5, 5],
                [5, 5, 5, 5, 5, 5, 5, 5, 5]
            ]
        },
        'breeder': {
            'thickness_matrix': [
                [100, 100, 30, 10, 10, 10, 30, 100, 100],
                [30,  30,  10, 5,  5,  5,  20, 30,  30],
                [25,  25,  25, 5,  5,  5,  25, 25,  25],
                [30,  30,  20, 5,  5,  5,  10, 30,  30],
                [100, 100, 30, 10, 10, 10, 30, 100, 100]
            ]
        },
        'back_wall': {
            'thickness_matrix': [
                [5, 5, 5, 5, 5, 5, 5, 5, 5],
                [5, 5, 5, 5, 5, 5, 5, 5, 5],
                [5, 5, 5, 5, 5, 5, 5, 5, 5],
                [5, 5, 5, 5, 5, 5, 5, 5, 5],
                [5, 5, 5, 5, 5, 5, 5, 5, 5]
            ]
        },
        'shield': {
            'thickness_matrix': [
                [25, 25, 25, 25, 25, 25, 25, 25, 25],
                [25, 25, 25, 25, 25, 25, 25, 25, 25],
                [25, 25, 25, 25, 25, 25, 25, 25, 25],
                [25, 25, 25, 25, 25, 25, 25, 25, 25],
                [25, 25, 25, 25, 25, 25, 25, 25, 25]
            ]
        },
        'manifolds': {
            'thickness_matrix': [
                [50, 50, 15, 5,  5,  5,  15, 50, 50],
                [20, 20, 5,  5,  5,  5,  15, 20, 20],
                [15, 15, 15, 5,  5,  5,  15, 15, 15],
                [20, 20, 15, 5,  5,  5,  5,  20, 20],
                [50, 50, 15, 5,  5,  5,  15, 50, 50]
            ]
        },
        'gap': {
            'thickness_matrix': [
                [5, 5, 5, 5, 5, 5, 5, 5, 5],
                [5, 5, 5, 5, 5, 5, 5, 5, 5],
                [5, 5, 5, 5, 5, 5, 5, 5, 5],
                [5, 5, 5, 5, 5, 5, 5, 5, 5],
                [5, 5, 5, 5, 5, 5, 5, 5, 5]
            ],
            'h5m_tag': 'Vacuum'
        },
        # Note that some neutron transport codes (such as OpenMC) will interpret
        # materials with "vacuum" in the name as void material
        'vacuum_vessel': {
            'thickness_matrix': [
                [15, 15, 15, 15, 15, 15, 15, 15, 15],
                [15, 15, 15, 15, 15, 15, 15, 15, 15],
                [15, 15, 15, 15, 15, 15, 15, 15, 15],
                [15, 15, 15, 15, 15, 15, 15, 15, 15],
                [15, 15, 15, 15, 15, 15, 15, 15, 15]
            ],
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
# Define magnet coil parameters
magnets = {
    'file': 'coils.txt',
    'cross_section': ['circle', 20],
    'start': 3,
    'stop': None,
    'sample': 6,
    'name': 'magnet_coils',
    'h5m_tag': 'magnets',
    'meshing': False
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
    'exclude': ['plasma'],
    'graveyard': False,
    'step_export': True,
    'h5m_export': 'Cubit',
    'plas_h5m_tag': 'Vacuum',
    'sol_h5m_tag': 'Vacuum',
    # Note the following export parameters are used only for Cubit H5M exports
    'facet_tol': 1,
    'len_tol': 5,
    'norm_tol': None,
    # Note the following export parameters are used only for Gmsh H5M exports
    'min_mesh_size': 5.0,
    'max_mesh_size': 20.0,
    'volume_atol': 0.00001,
    'center_atol': 0.00001,
    'bounding_box_atol': 0.00001
}

# Create stellarator
parastell.parastell(
    plas_eq, build, repeat, num_phi, num_theta,
    magnets = magnets, source = source, export = export
)
