import subprocess
import numpy as np
import pystell.read_vmec as read_vmec
import parastell.source_mesh as sm


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

mesh_size = (11, 81, 61)
toroidal_extent = 90.0

source_mesh_obj = sm.SourceMesh(
    vmec_obj,
    mesh_size,
    toroidal_extent,
    plasma_conditions=my_custom_plasma_conditions,
    reaction_rate=my_custom_reaction_rate,
)
source_mesh_obj.create_vertices()
source_mesh_obj.create_mesh()

# export and convert to vtk for visualization
source_mesh_obj.export_mesh()
subprocess.run("mbconvert source_mesh.h5m source_mesh.vtk", shell=True)
