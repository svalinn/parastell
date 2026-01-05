import numpy as np

import parastell.parastell as ps


# Define directory to export all output files to
export_dir = ""
# Define plasma equilibrium VMEC file
vmec_file = "wout_vmec.nc"

# Instantiate ParaStell build
stellarator = ps.Stellarator(vmec_file)

# Define source mesh parameters
cfs_values = np.linspace(1.0, 1.2, num=3)
poloidal_angles = np.linspace(0.0, 360.0, num=61)
toroidal_angles = np.linspace(0.0, 90.0, num=61)
# Construct source
stellarator.construct_source_mesh(cfs_values, poloidal_angles, toroidal_angles)
# Export source file
stellarator.export_source_mesh(filename="source_mesh", export_dir=export_dir)
