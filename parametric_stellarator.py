import magnet_coils
import source_mesh
import log
import read_vmec
import cadquery as cq
import cubit
import cad_to_dagmc
import numpy as np
from scipy.optimize import root_scalar
import os
import sys
from pymoab import core, types


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
            
        # Conditionally export tetrahedral meshing
        if magnets['meshing']:
            # Assign export paths
            file_path = os.getcwd()
            base_name = 'coil_mesh'
            general_export_path = f"{cwd}/{base_name}"
            exo_path = f'{general_export_path}.exo'
            h5m_path = f'{general_export_path}.h5m'
            # Exodus export
            cubit.cmd(f'export mesh "{exo_path}"')
            # Convert EXODUS to .h5m
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
    L = 2*(R + offset)*1.25*100

    # Create graveyard volume
    graveyard = cq.Workplane("XY").box(L, L, L).shell(5.0,
        kind = 'intersection')

    # Define name for graveyard component
    name = 'Graveyard'

    # Append graveyard to storage lists
    components[name]['solid'] = graveyard
    components[name]['h5m_tag'] = name

    return components


def offset_point(vmec, zeta, theta, offset):
    """Stellarator offset surface root-finding problem.

    Arguments:
        vmec (object): plasma equilibrium object.
        zeta (float): toroidal angle being solved for (rad).
        theta (float): poloidal angle of interest (rad).
        offset (float): total offset of layer from plamsa (m).
    
    Returns:
        pt (tuple): (x, y, z) tuple defining Cartesian point offset from
            stellarator plasma (m).
    """
    # Compute (x, y, z) point at edge of plasma for toroidal and poloidal
    # angles of interest
    r = np.array(vmec.vmec2xyz(1.0, theta, zeta))
    
    # Define small number
    eta = 0.000001

    # Vary poloidal and toroidal angles by small amount
    r_phi = np.array(vmec.vmec2xyz(1.0, theta, zeta + eta))
    r_theta = np.array(vmec.vmec2xyz(1.0, theta + eta, zeta))
    
    # Take difference from plasma edge point
    delta_phi = r_phi - r
    delta_theta = r_theta - r

    # Compute surface normal
    norm = np.cross(delta_phi, delta_theta)
    norm_mag = np.sqrt(sum(k**2 for k in norm))
    n = norm/norm_mag

    # Define offset point
    pt = r + offset*n

    return pt


def root_problem(zeta, vmec, theta, phi, offset):
    """Stellarator offset surface root-finding problem. The algorithm finds the
    point on the plasma surface whose unit normal, multiplied by a factor of
    offset, reaches the desired point on the toroidal plane defined by phi.

    Arguments:
        zeta (float): toroidal angle being solved for (rad).
        vmec (object): plasma equilibrium object.
        theta (float): poloidal angle of interest (rad).
        phi (float): toroidal angle of interest (rad).
        offset (float): total offset of layer from plasma (m).

    Returns:
        diff (float): difference between computed toroidal angle and toroidal
            angle of interest (root-finding problem definition).
    """
    # Define offset surface
    x, y, z = offset_point(vmec, zeta, theta, offset)

    # Compute solved toroidal angle
    offset_phi = np.arctan2(y, x)

    # If solved toroidal angle is negative, convert to positive angle
    if offset_phi < phi - np.pi:
        offset_phi += 2*np.pi

    # Compute difference between solved and defined toroidal angles
    diff = offset_phi - phi

    return diff


def offset_surface(vmec, theta, phi, offset, period_ext):
    """Computes offset surface point.

    Arguments:
        vmec (object): plasma equilibrium object.
        theta (float): poloidal angle of interest (rad).
        phi (float): toroidal angle of interest (rad).
        offset (float): total offset of layer from plamsa (m).
        period_ext (float): toroidal extent of each period (rad).

    Returns:
        r (array): offset suface point (m).
    """
    # Conditionally offset poloidal profile
    # Use VMEC plasma edge value for offset of 0
    if offset == 0:
        # Compute plasma edge point
        r = vmec.vmec2xyz(1.0, theta, phi)
    
    # Compute offset greater than zero
    elif offset > 0:
        # Root-solve for the toroidal angle at which the plasma
        # surface normal points to the toroidal angle of interest
        zeta = root_scalar(
            root_problem, args = (vmec, theta, phi, offset), x0 = phi,
            method = 'bisect',
            bracket = [phi - period_ext/2, phi + period_ext/2]
        )
        zeta = zeta.root

        # Compute offset surface point
        r = offset_point(vmec, zeta, theta, offset)
    
    # Raise error for negative offset values
    elif offset < 0:
        raise ValueError(
            'Offset must be greater than or equal to 0. Check thickness inputs '
            'for negative values'
        )

    return r


def expand_offset(offset_vals, ang_list, ang_list_exp, num_ang_list):
    """Expands offset list or matrix by linearly interpolating offset values
    according to given list of angles.

    Arguments:
        offset_vals (list or list of list of float): offset values to
            interpolate between (cm).
        ang_list (list of float): user-supplied list of angles (deg).
        ang_list_exp (list of float): interpolated list of angles (deg).
        num_ang_list (list of int): number of angles interpolated between
            user-supplied list.

    Returns:
        offset_vals_exp (list of list of float): interpolated matrix of offset
            values (cm).
    """
    # Initialize expanded list or matrix of offsets
    offset_vals_exp = []
    # Initialize previous number of offset values interpolated
    num_offset_prev = 0

    # Loop over supplied offset values
    for i, offset in enumerate(offset_vals[:-1]):
        # Extract number of offset values to interpolate
        num_offset = num_ang_list[i]
        # Extract next offset value or list in original list or matrix
        offset_next = offset_vals[i + 1]
        # Compute difference between next and current offset values
        offset_diff = offset_next - offset

        # Extract angle for current index
        ang = ang_list[i]
        # Extract angle for next index
        next_ang = ang_list[i + 1]
        # Compute difference between next and current angles
        ang_diff = next_ang - ang

        # Compute ratio of offset and angle differences
        diff_ratio = offset_diff/ang_diff

        # Interpolate between supplied offset values
        for j in range(num_offset):
            # Extract angle from interpolated list
            ang_int = ang_list_exp[num_offset_prev + j]
            # Interpolate offset
            offset_int = offset + diff_ratio*(ang_int - ang)
            # Append interpolated offset to storage list
            offset_vals_exp.append(offset_int)

        # Update number of offset values previously interpolated
        num_offset_prev += num_offset

    # Append final offset value to expanded list
    offset_vals_exp.append(offset_vals[-1])
    
    return offset_vals_exp


def stellarator_torus(
    vmec, num_periods, offset_mat, cutter, gen_periods, phi_list, phi_list_exp,
    num_phi_list, theta_list, theta_list_exp, num_theta_list):
    """Creates a stellarator helical torus as a CadQuery object.
    
    Arguments:
        vmec (object): plasma equilibrium object.
        num_periods (int): number of periods in stellarator geometry.
        offset_mat (list of list of float): matrix of (poloidal angle, toroidal
            angle) pairs of total offsets of layer from plamsa (cm).
        cutter (object): CadQuery solid object used to cut period of
            stellarator torus.
        gen_periods (int): number of stellarator periods to build in model.
        phi_list (list of float): user-suppied list of toroidal angles (deg).
        phi_list_exp (list of float): interpolated list of toroidal angles
            (deg).
        num_phi_list (list of int): number of toroidal angles interpolated
            between user-supplied list.
        theta_list (list of float): user-suppied list of poloidal angles (deg).
        theta_list_exp (list of float): interpolated list of poloidal angles
            (deg).
        num_theta_list (list of int): number of poloidal angles interpolated
            between user-supplied list.
    
    Returns:
        torus (object): stellarator torus CadQuery solid object.
        cutter (object): updated cutting volume CadQuery solid object.
    """
    # Determine toroidal extent of each period in degrees
    period_ext = 360.0/num_periods
    
    # Define initial angles defining each period needed
    initial_angles = np.linspace(
        period_ext, period_ext*(gen_periods - 1), num = gen_periods - 1
    )

    # Convert toroidal extent of period to radians
    period_ext = np.deg2rad(period_ext)

    # Linearly interpolate offsets according to toroidal angle to expand offset
    # matrix
    offset_mat_exp = expand_offset(
        offset_mat, phi_list, phi_list_exp, num_phi_list
    )

    # Initialize construction
    period = cq.Workplane("XY")

    # Generate poloidal profiles
    for phi, offset_list in zip(phi_list_exp, offset_mat_exp):
        # Initialize points in poloidal profile
        pts = []

        # Convert toroidal (phi) angle from degrees to radians
        phi = np.deg2rad(phi)
        
        # Linearly interpolate offsets according to poloidal angle to expand
        # list of offsets
        offset_list_exp = expand_offset(
            offset_list, theta_list, theta_list_exp, num_theta_list
        )

        # Compute array of points along poloidal profile
        for theta, offset in zip(theta_list_exp[:-1], offset_list_exp[:-1]):
            # Convert poloidal (theta) angle from degrees to radians
            theta = np.deg2rad(theta)

            # Compute offset surface point
            x, y, z = offset_surface(vmec, theta, phi, offset, period_ext)
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
        num_ang_list (list of int): number of angles interpolated between
            user-supplied list.
    """
    # Initialize interpolated list of angles
    ang_list_exp = []
    # Initialize list of number of angles interpolated between user-supplied
    # list
    num_ang_list = []

    # Compute total angular extent of supplied list
    ang_ext = ang_list[-1] - ang_list[0]
    
    # Loop over supplied angles
    for i, ang in enumerate(ang_list[:-1]):
        # Compute difference between current and next angles
        ang_diff = ang_list[i + 1] - ang
        # Compute ratio of angular difference and total angular extent
        ratio = ang_diff/ang_ext
        
        # Compute number of angles to interpolate
        n_ang = round(ratio*num_ang)
        # Append number of angles to interpolate to storage list
        num_ang_list.append(n_ang)
        
        # Compute angular difference between consecutive interpolated angles
        delta_ang = ang_diff/n_ang
        
        # Interpolate angles
        for j in range(n_ang):
            # Compute interpolated angle
            ang_int = ang + j*delta_ang
            # Append interpolated angle to storage list
            ang_list_exp.append(ang_int)

    # Append final specified angle to storage list
    ang_list_exp.append(ang_list[-1])

    return ang_list_exp, num_ang_list


# Define default export dictionary
export_def = {
    'exclude': [],
    'graveyard': False,
    'step_export': True,
    'h5m_export': None,
    'plas_h5m_tag': None,
    'facet_tol': None,
    'len_tol': None,
    'norm_tol': None,
    'min_mesh_size': 5.0,
    'max_mesh_size': 20.0,
    'volume_atol': 0.00001,
    'center_atol': 0.00001,
    'bounding_box_atol': 0.00001
}


def parametric_stellarator(
    plas_eq, num_periods, build, gen_periods, num_phi = 60,
    num_theta = 100, magnets = None, source = None, export = export_def,
    logger = None):
    """Generates CadQuery workplane objects for components of a
    parametrically-defined stellarator, based on user-supplied plasma
    equilibrium VMEC data and a user-defined radial build. Each component is
    of uniform thickness, concentric about the plasma edge. The user may
    export STEP files for each reactor component and/or a DAGMC-compatible
    H5M file using Cubit or Gmsh for use in neutronics simulations.

    Arguments:
        plas_eq (str): path to plasma equilibrium NetCDF file.
        num_periods (int): number of periods in stellarator geometry.
        build (dict): dictionary of list of toroidal and poloidal angles, as
            well as dictionary of component names with corresponding thickness
            matrix and optional material tag to use in H5M neutronics model.
            The thickness matrix specifies component thickness at specified
            (polidal angle, toroidal angle) pairs. This dictionary takes the
            form
            {
                'phi_list': list of float (deg),
                'theta_list': list of float (deg),
                'radial_build': {
                    'component': {
                        'thickness_matrix': list of list of float (cm),
                        'h5m_tag': h5m_tag (str)
                    }
                }
            }
            If no alternate material tag is supplied for the H5M file, the
            given component name will be used.
        gen_periods (int): number of stellarator periods to build in model.
        num_phi (int): number of phi geometric cross-sections to make for each
            period (defaults to 60).
        num_theta (int): number of points defining the geometric cross-section
            (defaults to 100).
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

    # Check if total toroidal extent exceeds 360 degrees
    try:
        assert gen_periods <= num_periods, \
            'Number of requested periods to generate exceeds number in ' \
            'stellarator geometry'
    except AssertionError as e:
        logger.error(e.args[0])
        raise e
    
    # Load plasma equilibrium data
    vmec = read_vmec.vmec_data(plas_eq)

    # Initialize component storage dictionary
    components = {}
    
    # Extract toroidal (phi) and poloidal (theta) arrays
    phi_list = build['phi_list']
    theta_list = build['theta_list']
    # Extract radial build
    radial_build = build['radial_build']

    # Extract array dimensions
    n_phi = len(phi_list)
    n_theta = len(theta_list)

    # Prepend plasma to radial build
    plas_thickness_mat = np.zeros((n_phi, n_theta))
    full_radial_build = {
        'plasma': {'thickness_matrix': plas_thickness_mat}, **radial_build
    }

    # Initialize volume used to cut periods
    cutter = None

    # Linearly interpolate angles to expand phi and theta lists
    phi_list_exp, num_phi_list = expand_ang(phi_list, num_phi)
    theta_list_exp, num_theta_list = expand_ang(theta_list, num_theta)

    # Initialize offset matrix
    offset_mat = np.zeros((n_phi, n_theta))
    
    # Generate components in radial build
    for name, layer_data in full_radial_build.items():
        # Notify which component is being generated
        logger.info(f'Building {name}...')
        
        # Conditionally assign plasma h5m tag
        if name == 'plasma':
            if export_dict['plas_h5m_tag'] is not None:
                layer_data['h5m_tag'] = export_dict['plas_h5m_tag']
        
        # Conditionally populate h5m tag for layer
        if 'h5m_tag' not in layer_data:
            layer_data['h5m_tag'] = name

        # Extract layer thickness matrix
        thickness_mat = layer_data['thickness_matrix']
        # Compute offset list, converting from cm to m
        offset_mat += np.array(thickness_mat)/100

        # Generate component
        try:
            torus, cutter = stellarator_torus(
                vmec, num_periods,
                offset_mat, cutter, gen_periods,
                phi_list, phi_list_exp, num_phi_list,
                theta_list, theta_list_exp, num_theta_list
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
        offset = 2*max(max(offset_mat))
        # Build graveyard
        components = graveyard(vmec, offset, components, logger)

    # Conditionally initialize Cubit
    if export_dict['h5m_export'] == 'Cubit' or magnets is not None:
        cubit.init([
            'cubit',
            '-nojournal',
            '-nographics',
            '-information', 'off',
            '-warning', 'off'
        ])

    # Conditionally build magnet coils and store volume indices
    if magnets is not None:
        magnets['vol_id'] = magnet_coils.magnet_coils(magnets, logger = logger)

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
