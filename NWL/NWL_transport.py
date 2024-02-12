import NWL.NWL as NWL


# Define simulation parameters
dagmc_geom = 'first_wall.h5m'
source_mesh = 'SourceMesh.h5m'
tor_ext = 90.0
ss_file = 'strengths.txt'
num_parts = 100000

NWL.NWL_transport(dagmc_geom, source_mesh, tor_ext, ss_file, num_parts)
