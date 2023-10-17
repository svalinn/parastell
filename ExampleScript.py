import parastell
import logging


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
        'blanket': {
            'thickness_matrix': [
                [160, 160, 40, 5,  5,  5,  40, 160, 160],
                [40,  40,  20, 20, 20, 20, 30, 40,  40 ],
                [45,  45,  40, 10, 10, 10, 40, 45,  45 ],
                [40,  40,  30, 20, 20, 20, 20, 40,  40 ],
                [160, 160, 40, 5,  5,  5,  40, 160, 160]
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
        'coolant_manifolds': {
            'thickness_matrix': [
                [35, 35, 15, 5, 5, 5, 15, 35, 35],
                [15, 15, 5,  5, 5, 5, 5,  15, 15],
                [10, 10, 5,  5, 5, 5, 5,  10, 10],
                [15, 15, 5,  5, 5, 5, 5,  15, 15],
                [35, 35, 15, 5, 5, 5, 15, 35, 35]
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
                [20, 20, 20, 20, 20, 20, 20, 20, 20],
                [20, 20, 20, 20, 20, 20, 20, 20, 20],
                [20, 20, 20, 20, 20, 20, 20, 20, 20],
                [20, 20, 20, 20, 20, 20, 20, 20, 20],
                [20, 20, 20, 20, 20, 20, 20, 20, 20]
            ],
            'h5m_tag': 'vv'
        }
    }
}
# Define number of periods in stellarator plasma
num_periods = 4
# Define number of periods to generate
gen_periods = 1
# Define number of toroidal cross-sections to make
num_phi = 60
# Define number of poloidal points to include in each toroidal cross-section
num_theta = 60
# Define magnet coil parameters
magnets = {
    'file': 'coils.txt',
    'cross_section': ['circle', 20],
    'start': 3,
    'stop': None,
    'name': 'magnet_coils',
    'h5m_tag': 'magnets',
    'meshing': True
}
# Define source mesh parameters
source = {
    'num_s': 11,
    'num_theta': 81,
    'num_phi': 241
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

# Define logger. Note that this is identical to the default logger instatiated
# by log.py. If no logger is passed to parametric_stellarator, this is the
# logger that will be used.
logger = logging.getLogger('log')
# Configure base logger message level
logger.setLevel(logging.INFO)
# Configure stream handler
s_handler = logging.StreamHandler()
# Configure file handler
f_handler = logging.FileHandler('stellarator.log')
# Define and set logging format
format = logging.Formatter(
    fmt = '%(asctime)s: %(message)s',
    datefmt = '%H:%M:%S'
)
s_handler.setFormatter(format)
f_handler.setFormatter(format)
# Add handlers to logger
logger.addHandler(s_handler)
logger.addHandler(f_handler)

# Create stellarator
parastell.parastell(
    plas_eq, num_periods, build, gen_periods, num_phi, num_theta,
    magnets = magnets, source = source,
    export = export, logger = logger
)
