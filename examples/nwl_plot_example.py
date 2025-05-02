import parastell.nwl_utils as nwl


# Define simulation parameters
dagmc_geom = "nwl_geom.h5m"
source_mesh = "source_mesh.h5m"
tor_ext = 90.0
ss_file = "source_strengths.txt"
num_parts = 100000

nwl.nwl_transport(dagmc_geom, source_mesh, tor_ext, ss_file, num_parts)

# Define first wall geometry and plotting parameters
source_file = "surface_source.h5"
plas_eq = "wout_vmec.nc"
tor_ext = 90.0
pol_ext = 360.0
wall_s = 1.08

nwl.nwl_plot(
    source_file, ss_file, plas_eq, tor_ext, pol_ext, wall_s, num_threads=2
)
