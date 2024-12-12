import numpy as np

import parastell.parastell as ps
import parastell.radial_distance_utils as rdu
from parastell.utils import enforce_helical_symmetry, smooth_matrix


# Define directory to export all output files to
export_dir = ""
# Define plasma equilibrium VMEC file
vmec_file = "wout_vmec.nc"

# Instantiate ParaStell build
stellarator = ps.Stellarator(vmec_file)

# Define build parameters for in-vessel components
# Use 13 x 13 uniformly spaced grid for in-vessel build
toroidal_angles = np.linspace(0, 90, num=13)
poloidal_angles = np.linspace(0, 360, num=13)
wall_s = 1.08
# Define build parameters for magnet coils
coils_file = "coils.example"
width = 40.0
thickness = 50.0
toroidal_extent = 90.0

# Measure separation between first wall and coils
available_space = rdu.measure_fw_coils_separation(
    vmec_file,
    toroidal_angles,
    poloidal_angles,
    wall_s,
    coils_file,
    width,
    thickness,
    sample_mod=1,
)
# For matrices defined by angles that are regularly spaced, measurement can
# result in matrix elements that are close to, but not exactly, helcially
# symmetric
available_space = enforce_helical_symmetry(available_space)
# Smooth matrix
steps = 25  # Apply Gaussian filter 25 times
sigma = 0.5  # Smooth using half a standard deviation for the Gaussian kernel
available_space = smooth_matrix(available_space, steps, sigma)
# For matrices defined by angles that are regularly spaced, matrix smoothing
# can result in matrix elements that are close to, but not exactly, helcially
# symmetric
available_space = enforce_helical_symmetry(available_space)
# Modify available space to account for thickness of magnets
available_space = available_space - max(width, thickness)

# Define a matrix of uniform unit thickness
uniform_unit_thickness = np.ones((len(toroidal_angles), len(poloidal_angles)))
# Define thickness matrices for each in-vessel component of uniform thickness
first_wall_thickness_matrix = uniform_unit_thickness * 5
back_wall_thickness_matrix = uniform_unit_thickness * 5
shield_thickness_matrix = uniform_unit_thickness * 35
vacuum_vessel_thickness_matrix = uniform_unit_thickness * 30

# Compute breeder thickness matrix
breeder_thickness_matrix = (
    available_space
    - first_wall_thickness_matrix
    - back_wall_thickness_matrix
    - shield_thickness_matrix
    - vacuum_vessel_thickness_matrix
)

radial_build_dict = {
    "first_wall": {"thickness_matrix": first_wall_thickness_matrix},
    "breeder": {"thickness_matrix": breeder_thickness_matrix},
    "back_wall": {"thickness_matrix": back_wall_thickness_matrix},
    "shield": {"thickness_matrix": shield_thickness_matrix},
    "vacuum_vessel": {
        "thickness_matrix": vacuum_vessel_thickness_matrix,
        "mat_tag": "vac_vessel",
    },
}

# Construct in-vessel components
stellarator.construct_invessel_build(
    toroidal_angles,
    poloidal_angles,
    wall_s,
    radial_build_dict,
    # Set num_ribs and num_rib_pts such that four (60/12 - 1) additional
    # locations are interpolated between each specified location
    num_ribs=61,
    num_rib_pts=61,
)
# Export in-vessel component files
stellarator.export_invessel_build()

# Construct magnets
stellarator.construct_magnets(
    coils_file, width, thickness, toroidal_extent, sample_mod=6
)
# Export magnet files
stellarator.export_magnets()

# Define source mesh parameters
mesh_size = (11, 81, 61)
toroidal_extent = 90.0
# Construct source
stellarator.construct_source_mesh(mesh_size, toroidal_extent)
# Export source file
stellarator.export_source_mesh(filename="source_mesh", export_dir=export_dir)

# Build Cubit model of Parastell Components
stellarator.build_cubit_model(skip_imprint=False)

# Export DAGMC neutronics H5M file
stellarator.export_cf_dagmc(filename="dagmc", export_dir=export_dir)
