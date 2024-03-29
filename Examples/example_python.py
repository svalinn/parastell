import numpy as np

import parastell as ps

# Define directory to export all output files to
export_dir = ''
# Define plasma equilibrium VMEC file
vmec_file = 'wout_vmec.nc'

# Define toroidal angles at which radial build is specified.
toroidal_angles = [0.0, 11.25, 22.5, 33.75, 45.0, 56.25, 67.5, 78.75, 90.0]
# Define poloidal angles at which radial build is specified.
poloidal_angles = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0, 360.0]
# Define build parameters for in-vessel componenets
invessel_build = {
    'toroidal_angles': toroidal_angles,
    'poloidal_angles': poloidal_angles,
    'radial_build': {
        'first_wall': {
            'thickness_matrix': np.ones(
                (len(toroidal_angles), len(poloidal_angles))
            )*5
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
            'thickness_matrix': np.ones(
                (len(toroidal_angles), len(poloidal_angles))
            )*5
        },
        'shield': {
            'thickness_matrix': np.ones(
                (len(toroidal_angles), len(poloidal_angles))
            )*50
        },
        'vacuum_vessel': {
            'thickness_matrix': np.ones(
                (len(toroidal_angles), len(poloidal_angles))
            )*10,
            'h5m_tag': 'vac_vessel'
        }
    },
    'wall_s': 1.08,
    'repeat': 0,
    'num_ribs': 61,
    'num_rib_pts': 61,
    'scale': 100,
    'plasma_mat_tag': 'Vacuum',
    'sol_mat_tag': 'Vacuum',
    'export_dir': export_dir
}

# Define magnet coil parameters
magnets = {
    'coils_file_path': 'coils.txt',
    'start_line': 3,
    'cross_section': ['circle', 20],
    'toroidal_extent': 90.0,
    'sample_mod': 6,
    'scale': 100,
    'step_filename': 'magnet_coils',
    'mat_tag': 'magnets',
    'export_mesh': True,
    'mesh_filename': 'magnet_mesh',
    'export_dir': export_dir
}

# Define source mesh parameters
source = {
    'num_s': 11,
    'num_theta': 81,
    'num_phi': 61,
    'toroidal_extent': 90.0,
    'scale': 100,
    'filename': 'source_mesh',
    'export_dir': export_dir
}

# Define DAGMC export parameters
dagmc_export = {
    'skip_imprint': False,
    'legacy_faceting': True,
    'faceting_tolerance': 1,
    'length_tolerance': 5,
    'normal_tolerance': None,
    'filename': 'dagmc',
    'export_dir': export_dir
}

# Generate parametric stellarator model
stellarator = ps.Stellarator(vmec_file)
stellarator.construct_invessel_build(invessel_build)
stellarator.construct_magnets(magnets)
stellarator.construct_source_mesh(source)
stellarator.export_dagmc(dagmc_export)
