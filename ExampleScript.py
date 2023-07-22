import parametric_stellarator
import logging


# Define plasma equilibrium VMEC file
plas_eq = 'plas_eq.nc'
# Define number of periods in stellarator plasma
num_periods = 4
# Define radial build
radial_build = {
    'sol': {'thickness': 10, 'h5m_tag': 'Vacuum'},
    'first_wall': {'thickness': 5},
    'blanket': {'thickness': 5},
    'back_wall': {'thickness': 5},
    'shield': {'thickness': 20},
    'coolant_manifolds': {'thickness': 5},
    'gap': {'thickness': 5, 'h5m_tag': 'Vacuum'},
    # Note that some neutron transport codes (such as OpenMC) will interpret
    # materials with "vacuum" in the name as void material
    'vacuum_vessel': {'thickness': 20, 'h5m_tag': 'vv'}
}
# Define number of periods to generate
gen_periods = 1
# Define number of toroidal cross-sections to make
num_phi = 60
# Define number of poloidal points to include in each toroidal cross-section
num_theta = 100
# Define magnet coil parameters
magnets = {
    'file': 'coils.txt',
    'cross_section': ['circle', 20],
    'start': 3,
    'stop': None,
    'name': 'magnet_coils',
    'h5m_tag': 'magnets'
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
    'plas_h5m_tag': None,
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
parametric_stellarator.parametric_stellarator(
    plas_eq, num_periods, radial_build, gen_periods, num_phi, num_theta,
    magnets = magnets, source = source,
    export = export, logger = logger
    )
