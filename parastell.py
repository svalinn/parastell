import magnet_coils
import source_mesh
import log
import read_vmec
import cadquery as cq
import cubit
import cad_to_dagmc
import numpy as np
import math
from scipy.interpolate import RegularGridInterpolator
import os
from pymoab import core, types
import inspect


def cubit_export(components, export, magnets):
    """Export H5M neutronics model via Cubit.

    Arguments:
        components (dict of dicts): dictionary of component names, each with a
            corresponding dictionary of CadQuery solid and material tag to be
            used in H5M neutronics model in the form
            {'name': {'solid': CadQuery solid (object), 'h5m_tag':
            h5m_tag (string)}}
        export (dict): dictionary of export parameters including
            {
                'exclude': names of components not to export (list of str,
                    defaults to empty),
                'graveyard': generate graveyard volume as additional component
                    (bool, defaults to False),
                'step_export': export component STEP files (bool, defaults to
                    True),
                'h5m_export': export DAGMC-compatible neutronics H5M file using
                    Cubit or Gmsh. Acceptable values are None or a string value
                    of 'Cubit' or 'Gmsh' (str, defaults to None). The string is
                    case-sensitive. Note that if magnets are included, 'Cubit'
                    must be used,
                'plas_h5m_tag': optional alternate material tag to use for
                    plasma. If none is supplied and the plasma is not excluded,
                    'plasma' will be used (str, defaults to None),
                'facet_tol': maximum distance a facet may be from surface of
                    CAD representation for Cubit export (float, defaults to
                    None),
                'len_tol': maximum length of facet edge for Cubit export
                    (float, defaults to None),
                'norm_tol': maximum change in angle between normal vector of
                    adjacent facets (float, defaults to None),
                'min_mesh_size': minimum mesh element size for Gmsh export
                    (float, defaults to 5.0),
                'max_mesh_size': maximum mesh element size for Gmsh export
                    (float, defaults to 20.0),
                'volume_atol': absolute volume tolerance to allow when matching
                    parts in intermediate BREP file with CadQuery parts for
                    Gmsh export(float, defaults to 0.00001),
                'center_atol': absolute center coordinates tolerance to allow
                    when matching parts in intermediate BREP file with CadQuery
                    parts for Gmsh export (float, defaults to 0.00001),
                'bounding_box_atol': absolute bounding box tolerance  to allow
                    when matching parts in intermediate BREP file with CadQuery
                    parts for Gmsh export (float, defaults to 0.00001).
            }
        magnets (dict): dictionary of magnet parameters including
            {
                'file': path to magnet coil point-locus data file (str),
                'cross_section': coil cross-section definition (list),
                'start': starting line index for data in file (int),
                'stop': stopping line index for data in file (int),
                'name': name to use for STEP export (str),
                'h5m_tag': material tag to use in H5M neutronics model (str)
            }
            For the list defining the coil cross-section, the cross-section
            shape must be either a circle or rectangle. For a circular
            cross-section, the list format is
            ['circle' (str), radius (float, cm)]
            For a rectangular cross-section, the list format is
            ['rectangle' (str), width (float, cm), thickness (float, cm)]
    """
    # Extract Cubit export parameters
    facet_tol = export['facet_tol']
    len_tol = export['len_tol']
    norm_tol = export['norm_tol']

    # Get current working directory
    cwd = os.getcwd()

    # Import solids
    for name in components.keys():
        cubit.cmd(f'import step "' + cwd + '/' + name + '.step" heal')
        components[name]['vol_id'] = cubit.get_last_id("volume")

    # Imprint and merge all volumes
    cubit.cmd('imprint volume all')
    cubit.cmd('merge volume all')

    # Conditionally assign magnet material group
    if magnets is not None:
        magnet_h5m_tag = magnets['h5m_tag']
        cubit.cmd(
            f'group "mat:{magnet_h5m_tag}" add volume '
            + " ".join(str(i) for i in magnets['vol_id'])
        )
    
    # Assign material groups
    for comp in components.values():
        cubit.cmd(f'group "mat:{comp["h5m_tag"]}" add volume {comp["vol_id"]}')

    # Initialize tolerance strings for export statement as empty strings
    facet_tol_str = ''
    len_tol_str = ''
    norm_tol_str = ''

    # Conditionally fill tolerance strings
    if facet_tol is not None:
        facet_tol_str = f'faceting_tolerance {facet_tol}'
    if len_tol is not None:
        len_tol_str = f'length_tolerance {len_tol}'
    if norm_tol is not None:
        norm_tol_str = f'normal_tolerance {norm_tol}'
    
    # DAGMC export
    cubit.cmd(
        f'export dagmc "dagmc.h5m" {facet_tol_str} {len_tol_str} {norm_tol_str}'
        f' make_watertight'
    )


def exports(export, components, magnets, logger):
    """Export components.

    Arguments:
        export (dict): dictionary of export parameters including
            {
                'exclude': names of components not to export (list of str,
                    defaults to empty),
                'graveyard': generate graveyard volume as additional component
                    (bool, defaults to False),
                'step_export': export component STEP files (bool, defaults to
                    True),
                'h5m_export': export DAGMC-compatible neutronics H5M file using
                    Cubit or Gmsh. Acceptable values are None or a string value
                    of 'Cubit' or 'Gmsh' (str, defaults to None). The string is
                    case-sensitive. Note that if magnets are included, 'Cubit'
                    must be used,
                'plas_h5m_tag': optional alternate material tag to use for
                    plasma. If none is supplied and the plasma is not excluded,
                    'plasma' will be used (str, defaults to None),
                'facet_tol': maximum distance a facet may be from surface of
                    CAD representation for Cubit export (float, defaults to
                    None),
                'len_tol': maximum length of facet edge for Cubit export
                    (float, defaults to None),
                'norm_tol': maximum change in angle between normal vector of
                    adjacent facets (float, defaults to None),
                'min_mesh_size': minimum mesh element size for Gmsh export
                    (float, defaults to 5.0),
                'max_mesh_size': maximum mesh element size for Gmsh export
                    (float, defaults to 20.0),
                'volume_atol': absolute volume tolerance to allow when matching
                    parts in intermediate BREP file with CadQuery parts for
                    Gmsh export(float, defaults to 0.00001),
                'center_atol': absolute center coordinates tolerance to allow
                    when matching parts in intermediate BREP file with CadQuery
                    parts for Gmsh export (float, defaults to 0.00001),
                'bounding_box_atol': absolute bounding box tolerance  to allow
                    when matching parts in intermediate BREP file with CadQuery
                    parts for Gmsh export (float, defaults to 0.00001).
            }
        components (dict of dicts): dictionary of component names, each with a
            corresponding dictionary of CadQuery solid and material tag to be
            used in H5M neutronics model in the form
            {'name': {'solid': CadQuery solid (object), 'h5m_tag':
            h5m_tag (string)}}
        magnets (dict): dictionary of magnet parameters including
            {
                'file': path to magnet coil point-locus data file (str),
                'cross_section': coil cross-section definition (list),
                'start': starting line index for data in file (int),
                'stop': stopping line index for data in file (int),
                'name': name to use for STEP export (str),
                'h5m_tag': material tag to use in H5M neutronics model (str)
                'meshing': setting for tetrahedral mesh generation (bool)
            }
            For the list defining the coil cross-section, the cross-section
            shape must be either a circle or rectangle. For a circular
            cross-section, the list format is
            ['circle' (str), radius (float, cm)]
            For a rectangular cross-section, the list format is
            ['rectangle' (str), width (float, cm), thickness (float, cm)]
        logger (object): logger object.
    """
    # Check that h5m_export has an appropriate value
    if export['h5m_export'] not in [None, 'Cubit', 'Gmsh']:
        raise ValueError(
            'h5m_export must be None or have a string value of \'Cubit\' or '
            '\'Gmsh\''
        )
    # Check that Cubit export has STEP files to use
    if export['h5m_export'] == 'Cubit' and not export['step_export']:
        raise ValueError('H5M export via Cubit requires STEP files')
    # Check that H5M export of magnets uses Cubit
    if export['h5m_export'] == 'Gmsh' and magnets is not None:
        raise ValueError(
            'Inclusion of magnets in H5M model requires Cubit export'
        )
    
    # Conditionally export STEP files
    if export['step_export']:
        # Signal STEP export
        logger.info('Exporting STEP files...')
        for name, comp in components.items():
            cq.exporters.export(comp['solid'], name + '.step')
            
    # Conditionally export tetrahedral mesh
    if magnets is not None and magnets['meshing']:
        # Signal STEP export
        logger.info('Exporting coil mesh...')
        # Assign export paths
        cwd = os.getcwd()
        base_name = 'coil_mesh'
        general_export_path = f"{cwd}/{base_name}"
        exo_path = f'{general_export_path}.exo'
        h5m_path = f'{general_export_path}.h5m'
        # EXODUS export
        cubit.cmd(f'export mesh "{exo_path}"')
        # Convert EXODUS to H5M
        mb = core.Core()
        exodus_set = mb.create_meshset()
        mb.load_file(exo_path, exodus_set)
        mb.write_file(h5m_path, [exodus_set])
    
    # Conditinally export H5M file via Cubit
    if export['h5m_export'] == 'Cubit':
        # Signal H5M export via Cubit
        logger.info('Exporting neutronics H5M file via Cubit...')
        # Export H5M file via Cubit
        cubit_export(components, export, magnets)
    
    # Conditionally export H5M file via Gmsh
    if export['h5m_export'] == 'Gmsh':
        # Signal H5M export via Gmsh
        logger.info('Exporting neutronics H5M file via Gmsh...')
        # Initialize H5M model
        model = cad_to_dagmc.CadToDagmc()
        # Extract component data
        for comp in components.values():
            model.add_cadquery_object(
                comp['solid'],
                material_tags = [comp['h5m_tag']]
            )
        # Export H5M file via Gmsh
        model.export_dagmc_h5m_file()


def graveyard(vmec, offset, components, logger):
    """Creates graveyard component.

    Arguments:
        vmec (object): plasma equilibrium object.
        offset (float): total offset of layer from plamsa (m).
        components (dict of dicts): dictionary of component names, each with a
            corresponding dictionary of CadQuery solid and material tag to be
            used in H5M neutronics model in the form
            {'name': {'solid': CadQuery solid (object), 'h5m_tag':
            h5m_tag (string)}}
        logger (object): logger object.
    
    Returns:
        components (dict of dicts): dictionary of component names, each with a
            corresponding dictionary of CadQuery solid and material tag to be
            used in H5M neutronics model in the form
            {'name': {'solid': CadQuery solid (object), 'h5m_tag':
            h5m_tag (string)}}
    """
    # Signal graveyard generation
    logger.info('Building graveyard...')
    
    # Determine maximum plasma edge radial position
    R = vmec.vmec2rpz(1.0, 0.0, 0.0)[0]

    # Define length of graveyard and convert from m to cm
    L = 4*(R + offset)*100

    # Create graveyard volume
    graveyard = cq.Workplane("XY").box(L, L, L).shell(5.0,
        kind = 'intersection')

    # Define name for graveyard component
    name = 'Graveyard'

    # Append graveyard to storage lists
    components[name]['solid'] = graveyard
    components[name]['h5m_tag'] = name

    return components


def offset_point(vmec, s, phi, theta, offset, plane_norm):
    """Stellarator offset surface root-finding problem.

    Arguments:
        vmec (object): plasma equilibrium object.
        s (float): normalized magnetic closed flux surface value.
        phi (float): toroidal angle being solved for (rad).
        theta (float): poloidal angle of interest (rad).
        offset (float): total offset of layer from plamsa (m).
        plane_norm (array of float): normal direction of toroidal plane.
    
    Returns:
        pt (tuple): (x, y, z) tuple defining Cartesian point offset from
            stellarator plasma (m).
    """
    # Define small value
    eps = 0.000001

    # Compute Cartesian coordinates of reference point
    ref_pt = np.array(vmec.vmec2xyz(s, theta, phi))
    # Vary poloidal angle by small amount
    next_pt = np.array(vmec.vmec2xyz(s, theta + eps, phi))
    
    # Take difference from reference point to approximate tangent
    tang = next_pt - ref_pt

    # Compute normal of poloidal profile
    norm = np.cross(tang, plane_norm)
    norm = norm/np.linalg.norm(norm)

    # Define offset point
    pt = ref_pt + offset*norm

    return pt


def offset_surface(vmec, s, theta, phi, offset, plane_norm):
    """Computes offset surface point.

    Arguments:
        vmec (object): plasma equilibrium object.
        s (float): normalized magnetic closed flux surface value.
        theta (float): poloidal angle of interest (rad).
        phi (float): toroidal angle of interest (rad).
        offset (float): total offset of layer from plamsa (m).
        plane_norm (array of float): normal direction of toroidal plane.

    Returns:
        r (array): offset suface point (m).
    """
    # Conditionally offset poloidal profile
    # Use VMEC plasma edge value for offset of 0
    if offset == 0:
        # Compute plasma edge point
        r = np.array(vmec.vmec2xyz(s, theta, phi))
    
    # Compute offset greater than zero
    elif offset > 0:
        # Compute offset surface point
        r = offset_point(vmec, s, phi, theta, offset, plane_norm)
    
    # Raise error for negative offset values
    elif offset < 0:
        raise ValueError(
            'Offset must be greater than or equal to 0. Check thickness inputs '
            'for negative values'
        )

    return r


def stellarator_torus(
    vmec, s, tor_ext, repeat, phi_list_exp, theta_list_exp, interpolator,
    cutter):
    """Creates a stellarator helical torus as a CadQuery object.
    
    Arguments:
        vmec (object): plasma equilibrium object.
        s (float): normalized magnetic closed flux surface value.
        tor_ext (float): toroidal extent of build (deg).
        repeat (int): number of times to repeat build.
        phi_list_exp (list of float): interpolated list of toroidal angles
            (deg).
        theta_list_exp (list of float): interpolated list of poloidal angles
            (deg).
        interpolator (object): scipy.interpolate.RegularGridInterpolator object.
        cutter (object): CadQuery solid object used to cut each segment of
            stellarator torus.
    
    Returns:
        torus (object): stellarator torus CadQuery solid object.
        cutter (object): updated cutting volume CadQuery solid object.
    """
    # Define initial angles defining segment of build
    initial_angles = np.linspace(
        tor_ext, tor_ext * repeat, num = repeat
    )

    # Initialize construction
    period = cq.Workplane("XY")

    # Generate poloidal profiles
    for phi in phi_list_exp:
        # Initialize points in poloidal profile
        pts = []

        # Convert toroidal (phi) angle from degrees to radians
        phi = np.deg2rad(phi)

        # Compute radial direction of poloidal profile
        x = np.cos(phi)
        y = np.sin(phi)
        dir = np.array([x, y, 0])
        # Compute norm of poloidal profile
        plane_norm = np.cross(dir, np.array([0, 0, 1]))
        plane_norm = plane_norm/np.linalg.norm(plane_norm)

        # Compute array of points along poloidal profile
        for theta in theta_list_exp[:-1]:
            # Convert poloidal (theta) angle from degrees to radians
            theta = np.deg2rad(theta)

            # Interpolate offset according to toroidal and poloidal angles
            offset = interpolator([np.rad2deg(phi), np.rad2deg(theta)])[0]

            # Compute offset surface point
            x, y, z = offset_surface(vmec, s, theta, phi, offset, plane_norm)
            # Convert from m to cm
            pt = (x*100, y*100, z*100)
            # Append point to poloidal profile
            pts += [pt]
        
        # Ensure final point is same as initial
        pts += [pts[0]]

        # Generate poloidal profile
        period = period.spline(pts).close()
    
    # Loft along poloidal profiles to generate period
    period = period.loft()

    # Conditionally cut period if not plasma volume
    if cutter is not None:
        period_cut = period - cutter
    else:
        period_cut = period

    # Update cutting volume
    cutter = period

    # Initialize torus with conditionally cut period
    torus = period_cut

    # Generate additional profiles
    for angle in initial_angles:
        period = period_cut.rotate((0, 0, 0), (0, 0, 1), angle)
        torus = torus.union(period)

    return torus, cutter


def expand_ang(ang_list, num_ang):
    """Expands list of angles by linearly interpolating according to specified
    number to include in stellarator build.

    Arguments:
        ang_list (list of float): user-supplied list of toroidal or poloidal
            angles (deg).
        num_ang (int): number of angles to include in stellarator build.
    
    Returns:
        ang_list_exp (list of float): interpolated list of angles (deg).
    """
    # Initialize interpolated list of angles
    ang_list_exp = []

    # Compute total angular extent of supplied list
    ang_ext = ang_list[-1] - ang_list[0]

    # Compute average distance between angles to include in stellarator build
    ang_diff_avg = ang_ext/(num_ang - 1)
    
    # Loop over supplied angles
    for ang, next_ang in zip(ang_list[:-1], ang_list[1:]):
        # Compute number of angles to interpolate
        n_ang = math.ceil((next_ang - ang)/ang_diff_avg)
        # Interpolate angles and append to storage list
        ang_list_exp = np.append(
            ang_list_exp,
            np.linspace(ang, next_ang, num = n_ang + 1)[:-1]
        )

    # Append final specified angle to storage list
    ang_list_exp = np.append(ang_list_exp, ang_list[-1])

    return ang_list_exp


# Define default export dictionary
export_def = {
    'exclude': [],
    'graveyard': False,
    'step_export': True,
    'h5m_export': None,
    'plas_h5m_tag': None,
    'sol_h5m_tag': None,
    'facet_tol': None,
    'len_tol': None,
    'norm_tol': None,
    'min_mesh_size': 5.0,
    'max_mesh_size': 20.0,
    'volume_atol': 0.00001,
    'center_atol': 0.00001,
    'bounding_box_atol': 0.00001
}


def parastell(
    plas_eq, build, repeat, num_phi = 61, num_theta = 61, magnets = None,
    source = None, export = export_def, logger = None):
    """Generates CadQuery workplane objects for components of a
    parametrically-defined stellarator, based on user-supplied plasma
    equilibrium VMEC data and a user-defined radial build. Each component is
    of uniform thickness, concentric about the plasma edge. The user may
    export STEP files for each reactor component and/or a DAGMC-compatible
    H5M file using Cubit or Gmsh for use in neutronics simulations.

    Arguments:
        plas_eq (str): path to plasma equilibrium NetCDF file.
        build (dict): dictionary of list of toroidal and poloidal angles, as
            well as dictionary of component names with corresponding thickness
            matrix and optional material tag to use in H5M neutronics model.
            The thickness matrix specifies component thickness at specified
            (polidal angle, toroidal angle) pairs. This dictionary takes the
            form
            {
                'phi_list': list of float (deg),
                'theta_list': list of float (deg),
                'wall_s': closed flux index extrapolation at wall (float),
                'radial_build': {
                    'component': {
                        'thickness_matrix': list of list of float (cm),
                        'h5m_tag': h5m_tag (str)
                    }
                }
            }
            If no alternate material tag is supplied for the H5M file, the
            given component name will be used.
        repeat (int): number of times to repeat build.
        num_phi (int): number of phi geometric cross-sections to make for each
            build segment (defaults to 61).
        num_theta (int): number of points defining the geometric cross-section
            (defaults to 100).
        magnets (dict): dictionary of magnet parameters including
            {
                'file': path to magnet coil point-locus data file (str),
                'cross_section': coil cross-section definition (list),
                'start': starting line index for data in file (int),
                'stop': stopping line index for data in file (int),
                'sample': sampling modifier for filament points (int). For a
                    user-supplied value of n, sample every n points in list of
                    points in each filament,
                'name': name to use for STEP export (str),
                'h5m_tag': material tag to use in H5M neutronics model (str)
                'meshing': setting for tetrahedral mesh generation (bool)
            }
            For the list defining the coil cross-section, the cross-section
            shape must be either a circle or rectangle. For a circular
            cross-section, the list format is
            ['circle' (str), radius (float, cm)]
            For a rectangular cross-section, the list format is
            ['rectangle' (str), width (float, cm), thickness (float, cm)]
        source (dict): dictionary of source mesh parameters including
            {
                'num_s': number of closed magnetic flux surfaces defining mesh
                    (int),
                'num_theta': number of poloidal angles defining mesh (int),
                'num_phi': number of toroidal angles defining mesh (int)
            }
        export (dict): dictionary of export parameters including
            {
                'exclude': names of components not to export (list of str,
                    defaults to empty),
                'graveyard': generate graveyard volume as additional component
                    (bool, defaults to False),
                'step_export': export component STEP files (bool, defaults to
                    True),
                'h5m_export': export DAGMC-compatible neutronics H5M file using
                    Cubit or Gmsh. Acceptable values are None or a string value
                    of 'Cubit' or 'Gmsh' (str, defaults to None). The string is
                    case-sensitive. Note that if magnets are included, 'Cubit'
                    must be used,
                'plas_h5m_tag': optional alternate material tag to use for
                    plasma. If none is supplied and the plasma is not excluded,
                    'plasma' will be used (str, defaults to None),
                'sol_h5m_tag': optional alternate material tag to use for 
                    scrape-off layer. If none is supplied and the scrape-off
                    layer is not excluded, 'sol' will be used (str, defaults to
                    None),
                'facet_tol': maximum distance a facet may be from surface of
                    CAD representation for Cubit export (float, defaults to
                    None),
                'len_tol': maximum length of facet edge for Cubit export
                    (float, defaults to None),
                'norm_tol': maximum change in angle between normal vector of
                    adjacent facets (float, defaults to None),
                'min_mesh_size': minimum mesh element size for Gmsh export
                    (float, defaults to 5.0),
                'max_mesh_size': maximum mesh element size for Gmsh export
                    (float, defaults to 20.0),
                'volume_atol': absolute volume tolerance to allow when matching
                    parts in intermediate BREP file with CadQuery parts for
                    Gmsh export(float, defaults to 0.00001),
                'center_atol': absolute center coordinates tolerance to allow
                    when matching parts in intermediate BREP file with CadQuery
                    parts for Gmsh export (float, defaults to 0.00001),
                'bounding_box_atol': absolute bounding box tolerance  to allow
                    when matching parts in intermediate BREP file with CadQuery
                    parts for Gmsh export (float, defaults to 0.00001).
            }
        logger (object): logger object (defaults to None). If no logger is
            supplied, a default logger will be instantiated.

    Returns:
        strengths (list): list of source strengths for each tetrahedron (1/s).
            Returned only if source mesh is generated.
    """
    # Conditionally instantiate logger
    if logger == None or not logger.hasHandlers():
        logger = log.init()
    
    # Signal new stellarator build
    logger.info('New stellarator build')
    
    # Update export dictionary
    export_dict = export_def.copy()
    export_dict.update(export)
    
    # Load plasma equilibrium data
    vmec = read_vmec.vmec_data(plas_eq)

    # Initialize component storage dictionary
    components = {}
    
    # Extract toroidal (phi) and poloidal (theta) arrays
    phi_list = build['phi_list']
    theta_list = build['theta_list']
    # Extract closed flux surface index extrapolation at wall
    wall_s = build['wall_s']
    # Extract radial build
    radial_build = build['radial_build']
    # Extract array dimensions
    n_phi = len(phi_list)
    n_theta = len(theta_list)

    # Compute toroidal extent of build in degrees
    tor_ext = phi_list[-1] - phi_list[0]

    # Check if total toroidal extent exceeds 360 degrees
    try:
        assert repeat*tor_ext <= 360.0, \
            'Total toroidal extent requested with repeated geometry exceeds ' \
            '360 degrees'
    except AssertionError as e:
        logger.error(e.args[0])
        raise e

    # Conditionally prepend scrape-off layer to radial build
    if wall_s != 1.0:
        sol_thickness_mat = np.zeros((n_phi, n_theta))
        radial_build = {
            'sol': {'thickness_matrix': sol_thickness_mat},
            **radial_build
        }
    
    # Prepend plasma to radial build
    plas_thickness_mat = np.zeros((n_phi, n_theta))
    radial_build = {
            'plasma': {'thickness_matrix': plas_thickness_mat},
            **radial_build
        }

    # Initialize volume used to cut segments
    cutter = None

    # Linearly interpolate angles to expand phi and theta lists
    phi_list_exp = expand_ang(phi_list, num_phi)
    theta_list_exp = expand_ang(theta_list, num_theta)

    # Initialize offset matrix
    offset_mat = np.zeros((n_phi, n_theta))
    
    # Generate components in radial build
    for name, layer_data in radial_build.items():
        # Notify which component is being generated
        logger.info(f'Building {name}...')
        
        # Conditionally assign plasma h5m tag and reference closed flux surface
        if name == 'plasma':
            if export_dict['plas_h5m_tag'] is not None:
                layer_data['h5m_tag'] = export_dict['plas_h5m_tag']
            s = 1.0
        else:
            s = wall_s

        # Conditionally assign scrape-off layer h5m tag
        if name == 'sol':
            if export_dict['sol_h5m_tag'] is not None:
                layer_data['h5m_tag'] = export_dict['sol_h5m_tag']
        
        # Conditionally populate h5m tag for layer
        if 'h5m_tag' not in layer_data:
            layer_data['h5m_tag'] = name

        # Extract layer thickness matrix
        thickness_mat = layer_data['thickness_matrix']
        # Compute offset list, converting from cm to m
        offset_mat += np.array(thickness_mat)/100

        # Build offset interpolator
        interp = RegularGridInterpolator((phi_list, theta_list), offset_mat)
        
        # Generate component
        try:
            torus, cutter = stellarator_torus(
                vmec, s, tor_ext, repeat, phi_list_exp, theta_list_exp, interp,
                cutter
            )
        except ValueError as e:
            logger.error(e.args[0])
            raise e

        # Store solid and name tag
        if name not in export_dict['exclude']:
            components[name] = {}
            components[name]['solid'] = torus
            components[name]['h5m_tag'] = layer_data['h5m_tag']

    # Conditionally build graveyard volume
    if export_dict['graveyard']:
        # Extract maximum offset
        offset = max(max(offset_mat))
        # Build graveyard
        components = graveyard(vmec, offset, components, logger)

    # Conditionally initialize Cubit
    if export_dict['h5m_export'] == 'Cubit' or magnets is not None:
        # Retrieve Cubit module directory
        cubit_dir = os.path.dirname(inspect.getfile(cubit))
        # Append plugins directory to Cubit module directory
        cubit_dir = cubit_dir + '/plugins/'
        # Initialize Cubit
        cubit.init([
            'cubit',
            '-nojournal',
            '-nographics',
            '-information', 'off',
            '-warning', 'off',
            '-commandplugindir',
            cubit_dir
        ])

    # Conditionally build magnet coils and store volume indices
    if magnets is not None:
        magnets['vol_id'] = magnet_coils.magnet_coils(
            magnets, tor_ext, logger = logger
        )

    # Export components
    try:
        exports(export_dict, components, magnets, logger)
    except ValueError as e:
        logger.error(e.args[0])
        raise e
    
    # Conditionally create source mesh
    if source is not None:
        strengths = source_mesh.source_mesh(vmec, source, logger = logger)
        return strengths
