from pathlib import Path

import numpy as np

import parastell.parastell as ps


# Define directory to export all output files to
export_dir = ''
# Define plasma equilibrium VMEC file
vmec_file = 'wout_vmec.nc'

# Instantiate ParaStell build
stellarator = ps.Stellarator(vmec_file)

# Define build parameters for in-vessel components
toroidal_angles = [0.0, 30.0, 60.0, 90.0]
poloidal_angles = [0.0, 120.0, 240.0, 360.0]
wall_s = 1.08

empty_radial_build_dict = {}
# Construct in-vessel components
stellarator.construct_invessel_build(
    toroidal_angles,
    poloidal_angles,
    wall_s,
    empty_radial_build_dict,
    split_chamber=False
)
# Export in-vessel component files
stellarator.export_invessel_build(
    export_cad_to_dagmc=False,
    export_dir=export_dir
)

# Define source mesh parameters
mesh_size = (11, 81, 61)
toroidal_extent = 90.0
# Construct source
stellarator.construct_source_mesh(
    mesh_size,
    toroidal_extent
)
# Export source file
stellarator.export_source_mesh(
    filename='source_mesh',
    export_dir=export_dir
)

strengths = stellarator.source_mesh.strengths
        
filepath = Path(export_dir) / 'source_strengths.txt'

file = open(filepath, 'w')
for tet in strengths:
    file.write(f'{tet}\n')

# Export DAGMC neutronics H5M file
stellarator.export_dagmc(
    skip_imprint=True,
    legacy_faceting=True,
    filename='nwl_geom',
    export_dir=export_dir
)
