import parastell.invessel_build as ivb
import parastell.parastell as ps
from parastell.utils import ribs_from_kisslinger_format
import numpy as np

# Get a predefined set of points representing the first wall.
(
    custom_surface_toroidal_angles,
    num_toroidal_angles,
    num_poloidal_angles,
    periods,
    custom_surface_rz_ribs,
) = ribs_from_kisslinger_format(
    "../tests/files_for_tests/kisslinger_file_example.txt",
    delimiter=" ",
    scale=1,
)

# For this example, the ribs and points on the ribs are evenly spaced, which
# is not required.
custom_surface_poloidal_angles = np.linspace(0, 360, num_poloidal_angles)

# Create a ReferenceSurface object from the known points and corresponding
# toroidal and poloidal angles.
ks = ivb.RibBasedSurface(
    custom_surface_rz_ribs,
    custom_surface_toroidal_angles,
    custom_surface_poloidal_angles,
)

# Define directory to export all output files to
export_dir = ""

# A VMEC file is required for source mesh generation
vmec_file = "wout_vmec.nc"

# Instantiate ParaStell build. The Cubit workflow is shown here, however,
# the Cad-to-Dagmc and PyDAGMC workflows also support the use of custom first
# wall profiles.
stellarator = ps.Stellarator(vmec_file, ref_surf=ks)

# Define desired toroidal and poloidal angles for building the stellarator
toroidal_angles = [0.0, 11.25, 22.5, 33.75, 45.0, 56.25, 67.5, 78.75, 90.0]
poloidal_angles = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0, 360.0]
wall_s = 1.08

# Define a matrix of uniform unit thickness
uniform_unit_thickness = np.ones((len(toroidal_angles), len(poloidal_angles)))

radial_build_dict = {
    "first_wall": {"thickness_matrix": uniform_unit_thickness * 5},
    "breeder": {
        "thickness_matrix": (
            [
                [75.0, 75.0, 75.0, 25.0, 25.0, 25.0, 75.0, 75.0, 75.0],
                [75.0, 75.0, 75.0, 25.0, 25.0, 75.0, 75.0, 75.0, 75.0],
                [75.0, 75.0, 25.0, 25.0, 75.0, 75.0, 75.0, 75.0, 75.0],
                [65.0, 25.0, 25.0, 65.0, 75.0, 75.0, 75.0, 75.0, 65.0],
                [45.0, 45.0, 75.0, 75.0, 75.0, 75.0, 75.0, 45.0, 45.0],
                [65.0, 75.0, 75.0, 75.0, 75.0, 65.0, 25.0, 25.0, 65.0],
                [75.0, 75.0, 75.0, 75.0, 75.0, 25.0, 25.0, 75.0, 75.0],
                [75.0, 75.0, 75.0, 75.0, 25.0, 25.0, 75.0, 75.0, 75.0],
                [75.0, 75.0, 75.0, 25.0, 25.0, 25.0, 75.0, 75.0, 75.0],
            ]
        )
    },
    "back_wall": {"thickness_matrix": uniform_unit_thickness * 5},
    "shield": {"thickness_matrix": uniform_unit_thickness * 30},
    "vacuum_vessel": {
        "thickness_matrix": uniform_unit_thickness * 10,
        "mat_tag": "vac_vessel",
    },
}
# Construct in-vessel components
stellarator.construct_invessel_build(
    toroidal_angles, poloidal_angles, wall_s, radial_build_dict, scale=1
)
# Export in-vessel component files
stellarator.export_invessel_build_step(export_dir=export_dir)

# Define build parameters for magnet coils
coils_file = "coils.example"
width = 40.0
thickness = 50.0
toroidal_extent = 90.0
# Construct magnets
stellarator.construct_magnets_from_filaments(
    coils_file, width, thickness, toroidal_extent, sample_mod=6
)
# Export magnet files
stellarator.export_magnets_step(filename="magnets", export_dir=export_dir)
stellarator.export_magnet_mesh_cubit(
    filename="magnet_mesh", export_dir=export_dir
)

# Define source mesh parameters
cfs_values = np.linspace(0.0, 1.0, num=11)
poloidal_angles = np.linspace(0.0, 360.0, num=61)
toroidal_angles = np.linspace(0.0, 90.0, num=61)
# Construct source
stellarator.construct_source_mesh(cfs_values, poloidal_angles, toroidal_angles)
# Export source file
stellarator.export_source_mesh(filename="source_mesh", export_dir=export_dir)

# Build Cubit model of Parastell Components
stellarator.build_cubit_model(skip_imprint=True)

# Export DAGMC neutronics H5M file
stellarator.export_cubit_dagmc(filename="dagmc", export_dir=export_dir)
