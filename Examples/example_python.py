import numpy as np

from parastell.source_mesh import set_plasma_conditions, set_rxn_rate
import parastell.parastell as ps

def my_plasma_conditions(s):

    if s >= 1:
        n_i = 0
        T_i = 0
    else:
        n_i = 5e18 * (1 - s)**4
        T_i = 1e6/s

    return n_i, T_i

def my_rxn_rate(n_i, T_i):

    return 1e18 * n_i * n_i * T_i**4

set_plasma_conditions(my_plasma_conditions)
set_rxn_rate(my_rxn_rate)

# Define directory to export all output files to
export_dir = ''
# Define plasma equilibrium VMEC file
vmec_file = 'wout_vmec.nc'

# Instantiate ParaStell build
stellarator = ps.Stellarator(vmec_file)

# Define build parameters for in-vessel components
toroidal_angles = [0.0, 11.25, 22.5, 33.75, 45.0, 56.25, 67.5, 78.75, 90.0]
poloidal_angles = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0, 360.0]
wall_s = 1.08

# Define a matrix of uniform unit thickness
uniform_unit_thickness = np.ones((len(toroidal_angles), len(poloidal_angles)))

radial_build_dict = {
    'first_wall': {
        'thickness_matrix': uniform_unit_thickness * 5
    },
    'breeder': {
        'thickness_matrix': ([
            [75.0, 75.0, 75.0, 25.0, 25.0, 25.0, 75.0, 75.0, 75.0],
            [75.0, 75.0, 75.0, 25.0, 25.0, 75.0, 75.0, 75.0, 75.0],
            [75.0, 75.0, 25.0, 25.0, 75.0, 75.0, 75.0, 75.0, 75.0],
            [65.0, 25.0, 25.0, 65.0, 75.0, 75.0, 75.0, 75.0, 65.0],
            [45.0, 45.0, 75.0, 75.0, 75.0, 75.0, 75.0, 45.0, 45.0],
            [65.0, 75.0, 75.0, 75.0, 75.0, 65.0, 25.0, 25.0, 65.0],
            [75.0, 75.0, 75.0, 75.0, 75.0, 25.0, 25.0, 75.0, 75.0],
            [75.0, 75.0, 75.0, 75.0, 25.0, 25.0, 75.0, 75.0, 75.0],
            [75.0, 75.0, 75.0, 25.0, 25.0, 25.0, 75.0, 75.0, 75.0]
        ])
    },
    'back_wall': {
        'thickness_matrix': uniform_unit_thickness * 5
    },
    'shield': {
        'thickness_matrix': uniform_unit_thickness * 50
    },
    'vacuum_vessel': {
        'thickness_matrix': uniform_unit_thickness * 10,
        'h5m_tag': 'vac_vessel'
    }
}
# Construct in-vessel components
stellarator.construct_invessel_build(
    toroidal_angles,
    poloidal_angles,
    wall_s,
    radial_build_dict
)
# Export in-vessel component files
stellarator.export_invessel_build(
    export_cad_to_dagmc=False,
    export_dir=export_dir
)

# Define build parameters for magnet coils
coils_file = 'coils.example'
cross_section = ['circle', 20]
toroidal_extent = 90.0
# Construct magnets
stellarator.construct_magnets(
    coils_file,
    cross_section,
    toroidal_extent,
    sample_mod=6
)
# Export magnet files
stellarator.export_magnets(
    step_filename='magnets',
    export_mesh=True,
    mesh_filename='magnet_mesh',
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

# Export DAGMC neutronics H5M file
stellarator.export_dagmc(
    skip_imprint=False,
    legacy_faceting=True,
    filename='dagmc',
    export_dir=export_dir
)
