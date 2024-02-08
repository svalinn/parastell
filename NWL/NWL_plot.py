import NWL
import numpy as np


# Define first wall geometry and plotting parameters
source_file = 'surface_source.h5'
ss_file = 'strengths.txt'
plas_eq = 'plas_eq.nc'
tor_ext = 90.0
pol_ext = 360.0
num_phi = 101
num_theta = 101
wall_s = 1.2
num_levels = 10

NWL.NWL_plot(
    source_file, ss_file, plas_eq, tor_ext, pol_ext, wall_s, num_phi,
    num_theta, num_levels
)
