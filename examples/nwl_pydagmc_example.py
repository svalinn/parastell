from pathlib import Path

import parastell.parastell as ps
from parastell import nwl_utils
import numpy as np


# Define plasma equilibrium VMEC file
vmec_file = "wout_vmec.nc"

# Instantiate ParaStell build
stellarator = ps.Stellarator(vmec_file)

# Define parameters for geometry
toroidal_extent = 90.0
toroidal_angles = np.linspace(0.0, toroidal_extent, num=4)
poloidal_angles = np.linspace(0.0, 360.0, num=4)
wall_s = 1.08
# Use a "void component" (thickness is arbitrary) since PyDAGMC routine cannot
# model plasma/chamber
radial_build = {
    "void_component": {
        "thickness_matrix": 10
        * np.ones((len(toroidal_angles), len(poloidal_angles))),
        "mat_tag": "Vacuum",
    }
}

# Construct and export in-vessel build
stellarator.construct_invessel_build(
    toroidal_angles,
    poloidal_angles,
    wall_s,
    radial_build,
    split_chamber=False,
    use_pydagmc=True,
)

# Construct and export source mesh
cfs_values = np.linspace(0.0, 1.0, num=11)
poloidal_angles = np.linspace(0.0, 360.0, num=61)
toroidal_angles = np.linspace(0.0, 90.0, num=61)

stellarator.construct_source_mesh(cfs_values, poloidal_angles, toroidal_angles)
source_mesh_filename = Path("source_mesh").with_suffix(".h5m")
stellarator.construct_source_mesh(cfs_values, poloidal_angles, toroidal_angles)
stellarator.export_source_mesh(filename=source_mesh_filename)
strengths = stellarator.source_mesh.strengths

# Construct and export DAGMC neutronics model
dagmc_filename = Path("nwl_geom").with_suffix(".h5m")
stellarator.build_pydagmc_model(
    magnet_exporter="cad_to_dagmc", max_mesh_size=60
)

# Add Vacuum boundary condition to first wall surface to terminate particles
wall_surface = stellarator.invessel_build.dag_model.surfaces_by_id[1]
group_id = 2
group_map = {("boundary:Vacuum", group_id): [wall_surface]}
stellarator.invessel_build.dag_model.add_groups(group_map)

stellarator.export_pydagmc_model(filename=dagmc_filename)

# Define simulation parameters
num_parts = 1_000_000
neutron_energy = 14.1e6 * 1.60218e-19 * 1e-6  # eV to MJ
neutron_power = neutron_energy * np.sum(strengths)

source_file = nwl_utils.fire_rays(
    dagmc_filename, source_mesh_filename, toroidal_extent, strengths, num_parts
)
nwl_mean, nwl_std_dev, toroidal_bins, poloidal_bins, area_mat = (
    nwl_utils.compute_nwl(
        source_file,
        vmec_file,
        wall_s,
        toroidal_extent,
        neutron_power,
        num_batches=30,
        num_threads=6,
    )
)
nwl_utils.plot_nwl(nwl_mean, toroidal_bins, poloidal_bins, filename="nwl_mean")
nwl_utils.plot_nwl(
    nwl_std_dev, toroidal_bins, poloidal_bins, filename="nwl_std_dev"
)
