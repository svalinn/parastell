import NWL as NWL
import numpy as np


# Define first wall geometry parameters
plas_eq = 'plas_eq.nc'
wall_s = 1.2
tor_ext = 90.0
num_phi = 61
num_theta = 61

# Define fusion neutron source parameters
source = {
    'num_s': 11,
    'num_theta': 81,
    'num_phi': 61,
    'tor_ext': tor_ext
}

# Define export parameters
export = {
    'step_export': True,
    'h5m_export': 'Cubit',
    'h5m_filename': 'first_wall',
    'dir': '',
    'native_meshing': False,
    'facet_tol': 1,
    'len_tol': 5,
    'norm_tol': None
}

NWL.NWL_geom(
    plas_eq, wall_s, tor_ext, num_phi, num_theta, source = source,
    export = export
)
