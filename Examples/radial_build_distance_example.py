import numpy as np
from scipy.ndimage import gaussian_filter

import parastell.parastell as ps
import parastell.radial_distance_utils as rdu


def smooth_matrix(matrix, steps):
    """Smooths a matrix via Gaussian filtering, without allowing matrix
    elements to increase in value.

    Arguments:
        matrix (2-D iterable of float): matrix to be smoothed.
        steps (int): number of smoothing steps. Analagous to Gaussian sigma.

    Returns:
        smoothed_matrix (2-D iterable of float): smoothed matrix.
    """
    previous_iteration = matrix

    for step in range(steps):
        smoothed_matrix = np.minimum(
            previous_iteration, gaussian_filter(previous_iteration, sigma=1)
        )
        previous_iteration = smoothed_matrix

    return smoothed_matrix


# Define directory to export all output files to
export_dir = ""
# Define plasma equilibrium VMEC file
vmec_file = "wout_vmec.nc"

# Instantiate ParaStell build
stellarator = ps.Stellarator(vmec_file)

# Define build parameters for in-vessel components
toroidal_angles = np.linspace(0, 90, num=21)
poloidal_angles = np.linspace(0, 360, num=23)
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
    sample_mod=2,
)
# Modify available space to account for thickness of magnets
available_space = available_space - np.sqrt(2) * thickness / 2

# Ensure poloidal symmetry at toroidal angles 0 and 45 degrees
for index in range(
    len(poloidal_angles) - 1, int((len(poloidal_angles) - 1) / 2), -1
):
    available_space[0, index] = np.flip(
        available_space[0, len(poloidal_angles) - 1 - index]
    )
    available_space[int((len(toroidal_angles) - 1) / 2), index] = np.flip(
        available_space[
            int((len(toroidal_angles) - 1) / 2),
            len(poloidal_angles) - 1 - index,
        ]
    )
# Ensure quasi-symmetry toroidally and poloidally
for index in range(
    len(toroidal_angles) - 1, int((len(toroidal_angles) - 1) / 2), -1
):
    available_space[index] = np.flip(
        available_space[len(toroidal_angles) - 1 - index]
    )

available_space = smooth_matrix(available_space, 100)

# Ensure poloidal symmetry at toroidal angles 0 and 45 degrees
for index in range(
    len(poloidal_angles) - 1, int((len(poloidal_angles) - 1) / 2), -1
):
    available_space[0, index] = np.flip(
        available_space[0, len(poloidal_angles) - 1 - index]
    )
    available_space[int((len(toroidal_angles) - 1) / 2), index] = np.flip(
        available_space[
            int((len(toroidal_angles) - 1) / 2),
            len(poloidal_angles) - 1 - index,
        ]
    )
# Ensure quasi-symmetry toroidally and poloidally
for index in range(
    len(toroidal_angles) - 1, int((len(toroidal_angles) - 1) / 2), -1
):
    available_space[index] = np.flip(
        available_space[len(toroidal_angles) - 1 - index]
    )

print(available_space)

# Define a matrix of uniform unit thickness
uniform_unit_thickness = np.ones((len(toroidal_angles), len(poloidal_angles)))

# Define thickness matrices for each in-vessel component of uniform thickness
first_wall_thickness_matrix = uniform_unit_thickness * 5
back_wall_thickness_matrix = uniform_unit_thickness * 5
shield_thickness_matrix = uniform_unit_thickness * 50
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
    num_ribs=61,
    num_rib_pts=67,
)
# Export in-vessel component files
stellarator.export_invessel_build(
    export_cad_to_dagmc=False, export_dir=export_dir
)

# Construct magnets
stellarator.construct_magnets(
    coils_file, width, thickness, toroidal_extent, sample_mod=6
)
# Export magnet files
stellarator.export_magnets(
    step_filename="magnets",
    # export_mesh=True,
    mesh_filename="magnet_mesh",
    export_dir=export_dir,
)
"""
# Define source mesh parameters
mesh_size = (11, 81, 61)
toroidal_extent = 90.0
# Construct source
stellarator.construct_source_mesh(mesh_size, toroidal_extent)
# Export source file
stellarator.export_source_mesh(filename="source_mesh", export_dir=export_dir)

# Build Cubit model of Parastell Components
stellarator.build_cubit_model(skip_imprint=False, legacy_faceting=True)

# Export DAGMC neutronics H5M file
stellarator.export_dagmc(filename="dagmc", export_dir=export_dir)
"""
