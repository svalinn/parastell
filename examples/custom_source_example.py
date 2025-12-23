import parastell.source_mesh as sm
from parastell.pystell import read_vmec
import numpy as np


# Parastell supports defining custom plasma conditions and reaction rates.
# These can be passed as keyword arguments both to SourceMesh and to
# Stellarator.construct_source_mesh(). Included is an example of defining
# custom (non-physical) plasma conditions and reaction rates. See the
# docstrings in source_mesh.py for additional information.


def my_custom_plasma_conditions(s):
    T_i = np.sin(s * np.pi)
    n_i = 1
    return n_i, T_i


def my_custom_reaction_rate(n_i, T_i):
    return n_i * T_i


vmec_file = "wout_vmec.nc"

vmec_obj = read_vmec.VMECData(vmec_file)

cfs_values = np.linspace(0.0, 1.0, num=11)
poloidal_angles = np.linspace(0.0, 360.0, num=61)
toroidal_angles = np.linspace(0.0, 90.0, num=61)

source_mesh_obj = sm.SourceMesh(
    vmec_obj,
    cfs_values,
    poloidal_angles,
    toroidal_angles,
    plasma_conditions=my_custom_plasma_conditions,
    reaction_rate=my_custom_reaction_rate,
)
source_mesh_obj.create_vertices()
source_mesh_obj.create_mesh()

source_mesh_obj.export_mesh("source_mesh.h5m")
