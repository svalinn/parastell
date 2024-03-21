import numpy as np
import math


def normalize(vec_list):
    """Normalizes a set of vectors.

    Arguments:
        vec_list (1 or 2D np array): single 1D vector or array of 1D vectors
            to be normalized
    Returns:
        vec_list (np array of same shape as input): single 1D normalized vector
            or array of normalized 1D vectors
    """
    if len(vec_list.shape) == 1:
        return vec_list / np.linalg.norm(vec_list)
    elif len(vec_list.shape) == 2:
        return vec_list / np.linalg.norm(vec_list, axis=1)[:, np.newaxis]
    else:
        print('Input \'vec_list\' must be 1-D or 2-D NumPy array')


def expand_ang_list(ang_list, num_ang):
    """Expands list of angles by linearly interpolating according to specified
    number to include in stellarator build.

    Arguments:
        ang_list (list of double): user-supplied list of toroidal or poloidal
            angles (rad).
        num_ang (int): number of angles to include in stellarator build.

    Returns:
        ang_list_exp (list of double): interpolated list of angles (rad).
    """
    ang_list = np.deg2rad(ang_list)

    ang_list_exp = []

    init_ang = ang_list[0]
    final_ang = ang_list[-1]
    ang_extent = final_ang - init_ang

    ang_diff_avg = ang_extent/(num_ang - 1)

    for ang, next_ang in zip(ang_list[:-1], ang_list[1:]):
        n_ang = math.ceil((next_ang - ang)/ang_diff_avg)

        ang_list_exp = np.append(
            ang_list_exp,
            np.linspace(ang, next_ang, num=n_ang + 1)[:-1]
        )

    ang_list_exp = np.append(ang_list_exp, ang_list[-1])

    return ang_list_exp


def def_default_params():
    """Define default parameters for ParaStell.

    Returns:
        m2cm (float): factor to convert meters to cm.
        cubit_flag (bool): flag to indicate whether Coreform Cubit has been
            initialized.
        invessel_build_def (dict): dictionary of in-vessel component
            parameters, including
            {
                'repeat': number of times to repeat build segment for full
                    model (int, defaults to 0).
                'num_ribs': total number of ribs over which to loft for each
                    build segment (int, defaults to 61). Ribs are set at
                    toroidal angles interpolated between those specified in
                    'toroidal_angles' if this value is greater than the number
                    of entries in 'toroidal_angles'.
                'num_rib_pts': total number of points defining each rib spline
                    (int, defaults to 61). Points are set at poloidal angles
                    interpolated between those specified in 'poloidal_angles'
                    if this value is greater than the number of entries in
                    'poloidal_angles'.
                'scale': a scaling factor between the units of VMEC and [cm]
                    (double, defaults to m2cm = 100).
                'export_cad_to_dagmc': export DAGMC neutronics H5M file of
                    in-vessel components via CAD-to-DAGMC (bool, defaults to
                    False).
                'plasma_mat_tag': alternate DAGMC material tag to use for
                    plasma. If none is supplied, 'plasma' will be used (str,
                    defaults to None).
                'sol_mat_tag': alternate DAGMC material tag to use for
                    scrape-off layer. If none is supplied, 'sol' will be used
                    (str, defaults to None).
                'dagmc_filename': name of DAGMC output file, excluding '.h5m'
                    extension (str, defaults to 'dagmc').
                'export_dir': directory to which to export the output files
                    (str, defaults to empty string).
            }
        magnets_def (dict): dictionary of magnet parameters, including
            {
                'sample_mod': sampling modifier for filament points (int,
                    defaults to 1). For a user-supplied value of n, sample
                    every n points in list of points in each filament.
                'scale': a scaling factor between the units of the filament
                    data and [cm] (double, defaults to m2cm = 100).
                'step_filename': name of STEP export output file, excluding
                    '.step' extension (str, defaults to 'magnets').
                'mat_tag': DAGMC material tag for magnets in DAGMC neutronics
                    model (str, defaults to 'magnets').
                'export_mesh': flag to indicate tetrahedral mesh generation for
                    magnet volumes (bool, defaults to False).
                'mesh_filename': name of tetrahedral mesh H5M file, excluding
                    '.h5m' extension (str, defaults to 'magnet_mesh').
                'export_dir': directory to which to export output files (str,
                    defaults to empty string).
            }
        source_def (dict): dictionary of source mesh parameters including
            {
                'scale': a scaling factor between the units of VMEC and [cm]
                    (double, defaults to m2cm = 100).
                'filename': name of H5M output file, excluding '.h5m' extension
                    (str, defaults to 'source_mesh').
                'export_dir': directory to which to export H5M output file
                    (str, defaults to empty string).
            }
        dagmc_export_def (dict): dictionary of DAGMC export parameters including
            {
                'skip_imprint': choose whether to imprint and merge all in
                    Coreform Cubit or to merge surfaces based on import order
                    and geometry information (bool).
                'legacy_faceting': choose legacy or native faceting for DAGMC
                    export (bool).
                'faceting_tolerance': maximum distance a facet may be from
                    surface of CAD representation for DAGMC export (double).
                'length_tolerance': maximum length of facet edge for DAGMC
                    export (double).
                'normal_tolerance': maximum change in angle between normal
                    vector of adjacent facets (double).
                'anisotropic_ratio': controls edge length ratio of elements
                    (double).
                'deviation_angle': controls deviation angle of facet from
                    surface (i.e., lesser deviation angle results in more
                    elements in areas with greater curvature) (double).
                'filename': name of DAGMC output file, excluding '.h5m'
                    extension (str, defaults to 'dagmc').
                'export_dir': directory to which to export DAGMC output file
                    (str, defaults to empty string).
            }
    """
    m2cm = 100
    cubit_flag = False
    invessel_build_def = {
        'repeat': 0,
        'num_ribs': 61,
        'num_rib_pts': 61,
        'scale': m2cm,
        'export_cad_to_dagmc': False,
        'plasma_mat_tag': None,
        'sol_mat_tag': None,
        'dagmc_filename': 'dagmc',
        'export_dir': ''
    }
    magnets_def = {
        'sample_mod': 1,
        'scale': m2cm,
        'step_filename': 'magnets',
        'mat_tag': 'magnets',
        'export_mesh': False,
        'mesh_filename': 'magnet_mesh',
        'export_dir': ''
    }
    source_def = {
        'scale': m2cm,
        'filename': 'source_mesh',
        'export_dir': ''
    }
    dagmc_export_def = {
        'skip_imprint': False,
        'legacy_faceting': True,
        'faceting_tolerance': None,
        'length_tolerance': None,
        'normal_tolerance': None,
        'anisotropic_ratio': 100,
        'deviation_angle': 5,
        'filename': 'dagmc',
        'export_dir': ''
    }

    return (
        m2cm, cubit_flag, invessel_build_def, magnets_def, source_def,
        dagmc_export_def
    )
