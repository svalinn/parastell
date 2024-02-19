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
from sklearn.preprocessing import normalize
import os
import inspect
from pathlib import Path


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
                'dir': directory to which to export output files (str, defaults
                    to empty string). Note that directory must end in '/', if
                    using Linux or MacOS, or '\' if using Windows.
                'h5m_filename': name of DAGMC-compatible neutronics H5M file
                    (str, defaults to 'dagmc'),
                'plas_h5m_tag': optional alternate material tag to use for
                    plasma. If none is supplied and the plasma is not excluded,
                    'plasma' will be used (str, defaults to None),
                'sol_h5m_tag': optional alternate material tag to use for 
                    scrape-off layer. If none is supplied and the scrape-off
                    layer is not excluded, 'sol' will be used (str, defaults to
                    None),
                'native_meshing': choose native or legacy faceting for DAGMC
                    export (bool, defaults to False),
                'facet_tol': maximum distance a facet may be from surface of
                    CAD representation for Cubit export (float, defaults to
                    None),
                'len_tol': maximum length of facet edge for Cubit export
                    (float, defaults to None),
                'norm_tol': maximum change in angle between normal vector of
                    adjacent facets (float, defaults to None),
                'anisotropic_ratio': controls edge length ratio of elements
                    (float, defaults to 100.0),
                'deviation_angle': controls deviation angle of facet from
                    surface, i.e. lower deviation angle => more elements in
                    areas with higher curvature (float, defaults to 5.0),
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

    def legacy_export():
        """Exports neutronics H5M file via legacy plug-in faceting method.
        """
        # Conditionally assign magnet material group
        if magnets is not None:
            magnet_h5m_tag = magnets['h5m_tag']
            cubit.cmd(
                f'group "mat:{magnet_h5m_tag}" add volume '
                + " ".join(str(i) for i in magnets['vol_id'])
            )
        
        # Assign material groups
        for comp in components.values():
            cubit.cmd(
                f'group "mat:{comp["h5m_tag"]}" add volume {comp["vol_id"]}'
            )

        # Extract Cubit export parameters
        facet_tol = export['facet_tol']
        len_tol = export['len_tol']
        norm_tol = export['norm_tol']

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
        export_path = dir / filename.with_suffix('.h5m')
        cubit.cmd(
            f'export dagmc "{export_path}" {facet_tol_str} {len_tol_str} '
            f'{norm_tol_str} make_watertight'
        )

    def native_export():
        """Exports neutronics H5M file via native Cubit faceting method.
        """
        # Extract Cubit export parameters
        anisotropic_ratio = export['anisotropic_ratio']
        deviation_angle = export['deviation_angle']

        # Create materials for native cubit meshing
        for comp in components.values():
            cubit.cmd(
                f'create material "{comp["h5m_tag"]}" property_group '
                + '"CUBIT-ABAQUS"'
            )

        # Assign components to blocks
        for comp in components.values():
            cubit.cmd('set duplicate block elements off')
            cubit.cmd(
                "block " + str(comp['vol_id']) + " add volume "
                + str(comp['vol_id'])
            )
        
        # Assign materials to blocks
        for comp in components.values():
            cubit.cmd(
                "block " + str(comp['vol_id']) + " material "
                + ''.join(("\'",comp['h5m_tag'],"\'"))
            )
            
        if magnets is not None:
            magnet_h5m_tag = magnets['h5m_tag']
            
            # Create magnet material
            cubit.cmd(
                f'create material "{magnet_h5m_tag}" property_group '
                + '"CUBIT-ABAQUS"'
            )

            # Assign magnets to block
            block_number = min(magnets['vol_id'])
            for vol in magnets['vol_id']:
                cubit.cmd('set duplicate block elements off')
                cubit.cmd(
                    "block " + str(block_number) + " add volume " + str(vol)
                )
            
            # Assign magnet material to block
            cubit.cmd(
                "block " + str(block_number) + " material "
                + ''.join(("\'",magnet_h5m_tag,"\'"))
            )

        # Mesh the model
        cubit.cmd(
            "set trimesher coarse on ratio " + str(anisotropic_ratio)
            + " angle " + str(deviation_angle)
        )
        cubit.cmd("surface all scheme trimesh")
        cubit.cmd("mesh surface all")

        # Export DAGMC file
        export_path = dir / filename.with_suffix('.h5m')
        cubit.cmd(
            f'export cf_dagmc "{export_path}" overwrite'
        )

    dir = Path(export['dir'])
    filename = Path(export['h5m_filename'])

    def orient_spline_surfaces(volume_id):
        """Return the inner and outer surface ids for a given volume id
        """
        surfaces = cubit.get_relatives('volume', volume_id, 'surface')
            
        splineSurfaces = []

        for surf in surfaces:
            if cubit.get_surface_type(surf) == 'spline surface':
                splineSurfaces.append(surf)

        # check if this is the plasma
        if len(splineSurfaces) == 1:
            outerSurface = splineSurfaces[0]
            innerSurface = None
        
        else:
            # the outer surface bounding box will have the larger max xy value
            if cubit.get_bounding_box('surface', splineSurfaces[1])[4] > cubit.get_bounding_box('surface',splineSurfaces[0])[4]:
                outerSurface = splineSurfaces[1]
                innerSurface = splineSurfaces[0]

            else:
                outerSurface = splineSurfaces[0]
                innerSurface = splineSurfaces[1]
            
        return innerSurface, outerSurface
    
    def merge_layer_surfaces(): # assumes that components dict is ordered from inside to out
        """Merge surfaces based on surface ids rather than imprinting/merging all
        """

        # tracks the surface id of the outer surface of the previous layer
        lastOuterSurface = None
        
        for name in components.keys():

            # get volume ID for layer

            vol_id = components[name]['vol_id']

            # get the inner and outer surface IDs of the current layer
            innerSurface, outerSurface = orient_spline_surfaces(vol_id)

            # wait to merge until the next layer if the plasma is included
            # store surface to be merged for next loop
            if name == 'plasma':
                
                lastOuterSurface = outerSurface

            # check if we are in the first layer in a build with plasma excluded
            # store outer surface to be merged in next loop
            elif lastOuterSurface is None:
                
                lastOuterSurface = outerSurface

            # merge inner surface with outer surface of previous layer
            else:

                cubit.cmd('merge surface '+ str(innerSurface) + ' ' + str(lastOuterSurface))
                
                lastOuterSurface = outerSurface
        

    for name in components.keys():
        import_path = dir / Path(name).with_suffix('.step')
        cubit.cmd(f'import step "{import_path}" heal')
        components[name]['vol_id'] = cubit.get_last_id("volume")

    if export['skip_imprinting']:
        merge_layer_surfaces()

    else:
        cubit.cmd('imprint volume all')
        cubit.cmd('merge volume all')
            
    if export['native_meshing']:
        native_export()

    else:
        legacy_export()
    


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
                'dir': directory to which to export output files (str, defaults
                    to empty string). Note that directory must end in '/', if
                    using Linux or MacOS, or '\' if using Windows.
                'h5m_filename': name of DAGMC-compatible neutronics H5M file
                    (str, defaults to 'dagmc'),
                'plas_h5m_tag': optional alternate material tag to use for
                    plasma. If none is supplied and the plasma is not excluded,
                    'plasma' will be used (str, defaults to None),
                'sol_h5m_tag': optional alternate material tag to use for 
                    scrape-off layer. If none is supplied and the scrape-off
                    layer is not excluded, 'sol' will be used (str, defaults to
                    None),
                'native_meshing': choose native or legacy faceting for DAGMC
                    export (bool, defaults to False),
                'facet_tol': maximum distance a facet may be from surface of
                    CAD representation for Cubit export (float, defaults to
                    None),
                'len_tol': maximum length of facet edge for Cubit export
                    (float, defaults to None),
                'norm_tol': maximum change in angle between normal vector of
                    adjacent facets (float, defaults to None),
                'anisotropic_ratio': controls edge length ratio of elements
                    (float, defaults to 100.0),
                'deviation_angle': controls deviation angle of facet from
                    surface, i.e. lower deviation angle => more elements in
                    areas with higher curvature (float, defaults to 5.0),
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
    if export['h5m_export'] not in [None, 'Cubit', 'Gmsh']:
        raise ValueError(
            'h5m_export must be None or have a string value of \'Cubit\' or '
            '\'Gmsh\''
        )
    if export['h5m_export'] == 'Cubit' and not export['step_export']:
        raise ValueError('H5M export via Cubit requires STEP files')
    if export['h5m_export'] == 'Gmsh' and magnets is not None:
        raise ValueError(
            'Inclusion of magnets in H5M model requires Cubit export'
        )
    
    dir = Path(export['dir'])
    filename = Path(export['h5m_filename'])

    if export['step_export']:
        logger.info('Exporting STEP files...')
        for name, comp in components.items():
            export_path = dir / Path(name).with_suffix('.step')
            cq.exporters.export(
                comp['solid'],
                str(export_path)
            )
    
    if export['h5m_export'] == 'Cubit':
        logger.info('Exporting neutronics H5M file via Cubit...')
        cubit_export(components, export, magnets)
    
    if export['h5m_export'] == 'Gmsh':
        logger.info('Exporting neutronics H5M file via Gmsh...')
        model = cad_to_dagmc.CadToDagmc()
        for comp in components.values():
            model.add_cadquery_object(
                comp['solid'],
                material_tags = [comp['h5m_tag']]
            )
        model.export_dagmc_h5m_file(
            filename = dir / filename.with_suffix('.h5m')
        )


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
    logger.info('Building graveyard...')
    
    # Define constant to convert from m to cm
    m2cm = 100

    # Determine maximum plasma edge radial position
    R = vmec.vmec2rpz(1.0, 0.0, 0.0)[0]

    # Define length of graveyard and convert from m to cm
    # Double to cover full geometry
    # Mutiply by factor of 2 to be conservative
    L = 2*2*(R + offset)*m2cm

    # Create graveyard volume
    graveyard = cq.Workplane("XY").box(L, L, L).shell(5.0,
        kind = 'intersection')

    # Define name for graveyard component
    name = 'Graveyard'

    # Append graveyard to storage lists
    components[name]['solid'] = graveyard
    components[name]['h5m_tag'] = name

    return components


def surf_norm(vmec, s, phi, theta, ref_pt, plane_norm):
    """Stellarator offset surface root-finding problem.

    Arguments:
        vmec (object): plasma equilibrium object.
        s (float): normalized magnetic closed flux surface label.
        phi (float): toroidal angle being solved for (rad).
        theta (float): poloidal angle of interest (rad).
        ref_pt (array of float): Cartesian coordinates of plasma edge or
            scrape-off-layer location (m).
        plane_norm (array of float): normal direction of toroidal plane.
    
    Returns:
        norm (array of float): surface normal direction (m).
    """
    # Vary poloidal angle by small amount
    eps = 0.000001
    next_pt = np.array(vmec.vmec2xyz(s, theta + eps, phi))
    
    tangent = next_pt - ref_pt

    norm = np.cross(plane_norm, tangent)
    norm = normalize(norm.reshape(-1, 1), axis = 0).ravel()
    norm = np.array(norm)

    return norm


def offset_point(vmec, s, theta, phi, offset, plane_norm):
    """Computes offset surface point.

    Arguments:
        vmec (object): plasma equilibrium object.
        s (float): normalized magnetic closed flux surface label.
        theta (float): poloidal angle of interest (rad).
        phi (float): toroidal angle of interest (rad).
        offset (float): total offset of layer from plamsa (m).
        plane_norm (array of float): normal direction of toroidal plane.

    Returns:
        r (array): offset suface point (m).
    """
    if offset == 0:
        r = np.array(vmec.vmec2xyz(s, theta, phi))
    
    elif offset > 0:
        ref_pt = np.array(vmec.vmec2xyz(s, theta, phi))
        norm = surf_norm(vmec, s, phi, theta, ref_pt, plane_norm)
        r = ref_pt + offset*norm
    
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
        s (float): normalized magnetic closed flux surface label.
        tor_ext (float): toroidal extent of build (rad).
        repeat (int): number of times to repeat build.
        phi_list_exp (list of float): interpolated list of toroidal angles
            (rad).
        theta_list_exp (list of float): interpolated list of poloidal angles
            (rad).
        interpolator (object): scipy.interpolate.RegularGridInterpolator object.
        cutter (object): CadQuery solid object used to cut each segment of
            stellarator torus.
    
    Returns:
        torus (object): stellarator torus CadQuery solid object.
        cutter (object): updated cutting volume CadQuery solid object.
    """
    # Define initial angles defining segments of build
    initial_angles = np.linspace(
        np.rad2deg(tor_ext), np.rad2deg(repeat * tor_ext), num = repeat
    )

    # Initialize construction
    segment = cq.Workplane("XY")

    # Define constant to convert from m to cm
    m2cm = 100

    for phi in phi_list_exp:
        # Initialize points in poloidal profile
        pts = []

        # Compute norm of poloidal profile
        plane_norm = np.array([-np.sin(phi), np.cos(phi), 0])

        for theta in theta_list_exp[:-1]:
            offset = interpolator([phi, theta])[0]
            x, y, z = m2cm*offset_point(vmec, s, theta, phi, offset, plane_norm)
            pt = (x, y, z)
            pts += [pt]
        
        # Ensure final point is same as initial
        pts += [pts[0]]

        # Generate poloidal profile
        segment = segment.spline(pts).close()
    
    # Loft along poloidal profiles to generate segment
    segment = segment.loft()

    # Conditionally cut segment if not plasma volume
    if cutter is not None:
        segment_cut = segment - cutter
    else:
        segment_cut = segment

    # Update cutting volume
    cutter = segment

    # Initialize torus with segment
    torus = segment_cut

    # Generate additional profiles
    for angle in initial_angles:
        segment = segment_cut.rotate((0, 0, 0), (0, 0, 1), angle)
        torus = torus.union(segment)

    return torus, cutter


def expand_ang(ang_list, num_ang):
    """Expands list of angles by linearly interpolating according to specified
    number to include in stellarator build.

    Arguments:
        ang_list (list of float): user-supplied list of toroidal or poloidal
            angles (rad).
        num_ang (int): number of angles to include in stellarator build.
    
    Returns:
        ang_list_exp (list of float): interpolated list of angles (rad).
    """
    # Initialize interpolated list of angles
    ang_list_exp = []

    init_ang = ang_list[0]
    final_ang = ang_list[-1]
    ang_extent = final_ang - init_ang

    # Compute average distance between angles to include in stellarator build
    ang_diff_avg = ang_extent/(num_ang - 1)
    
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
    'dir': '',
    'h5m_filename': 'dagmc',
    'native_meshing': False,
    'plas_h5m_tag': None,
    'sol_h5m_tag': None,
    'facet_tol': None,
    'len_tol': None,
    'norm_tol': None,
    'skip_imprinting': False,
    'anisotropic_ratio': 100,
    'deviation_angle': 5,
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
                'phi_list': toroidal angles at which radial build is specified.
                    This list should always begin at 0.0 and it is advised not
                    to extend past one stellarator period. To build a geometry
                    that extends beyond one period, make use of the 'repeat'
                    parameter (list of float, deg).
                'theta_list': poloidal angles at which radial build is
                    specified. This list should always span 360 degrees (list
                    of float, deg).
                'wall_s': closed flux surface label extrapolation at wall
                    (float),
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
            (defaults to 61).
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
                'dir': directory to which to export output files (str, defaults
                    to empty string). Note that directory must end in '/', if
                    using Linux or MacOS, or '\' if using Windows.
                'h5m_filename': name of DAGMC-compatible neutronics H5M file
                    (str, defaults to 'dagmc'),
                'plas_h5m_tag': optional alternate material tag to use for
                    plasma. If none is supplied and the plasma is not excluded,
                    'plasma' will be used (str, defaults to None),
                'sol_h5m_tag': optional alternate material tag to use for 
                    scrape-off layer. If none is supplied and the scrape-off
                    layer is not excluded, 'sol' will be used (str, defaults to
                    None),
                'native_meshing': choose native or legacy faceting for DAGMC
                    export (bool, defaults to False),
                'facet_tol': maximum distance a facet may be from surface of
                    CAD representation for Cubit export (float, defaults to
                    None),
                'len_tol': maximum length of facet edge for Cubit export
                    (float, defaults to None),
                'norm_tol': maximum change in angle between normal vector of
                    adjacent facets (float, defaults to None),
                'skip_imprinting': choose whether to imprint and merge all in
                    cubit or to merge surfaces based on import order and
                    geometry information.
                'anisotropic_ratio': controls edge length ratio of elements
                    (float, defaults to 100.0),
                'deviation_angle': controls deviation angle of facet from
                    surface, i.e. lower deviation angle => more elements in
                    areas with higher curvature (float, defaults to 5.0),
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
    export_dict = export_def.copy()
    export_dict.update(export)
    
    if export_dict['h5m_export'] == 'Cubit' or magnets is not None:
        cubit_dir = os.path.dirname(inspect.getfile(cubit))
        cubit_dir = Path(cubit_dir) / Path('plugins')
        cubit.init([
            'cubit',
            '-nojournal',
            '-nographics',
            '-information', 'off',
            '-warning', 'off',
            '-commandplugindir',
            str(cubit_dir)
        ])
    
    if logger == None or not logger.hasHandlers():
        logger = log.init()
    
    logger.info('New stellarator build')
    
    # Load plasma equilibrium data
    vmec = read_vmec.vmec_data(plas_eq)

    components = {}
    
    phi_list = build['phi_list']
    phi_list = np.deg2rad(phi_list)

    try:
        assert phi_list[0] == 0.0, \
            'Initial toroidal angle not equal to 0. Please redefine ' \
            'phi_list, beginning at 0.'
    except AssertionError as e:
        logger.error(e.args[0])
        raise e
    
    theta_list = build['theta_list']
    theta_list = np.deg2rad(theta_list)

    try:
        assert theta_list[-1] - theta_list[0] == 2*np.pi, \
            'Poloidal extent is not 360 degrees. Please ensure poloidal ' \
            'angles are specified for one full revolution.'
    except AssertionError as e:
        logger.error(e.args[0])
        raise e

    wall_s = build['wall_s']
    radial_build = build['radial_build']
    
    n_phi = len(phi_list)
    n_theta = len(theta_list)

    # Extract toroidal extent of build
    tor_ext = phi_list[-1] - phi_list[0]

    try:
        assert (repeat + 1)*tor_ext <= 2*np.pi, \
            'Total toroidal extent requested with repeated geometry exceeds ' \
            '360 degrees. Please examine phi_list and the repeat parameter.'
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
        logger.info(f'Building {name}...')
        
        if name == 'plasma':
            if export_dict['plas_h5m_tag'] is not None:
                layer_data['h5m_tag'] = export_dict['plas_h5m_tag']
            s = 1.0
        else:
            s = wall_s

        if name == 'sol':
            if export_dict['sol_h5m_tag'] is not None:
                layer_data['h5m_tag'] = export_dict['sol_h5m_tag']
        
        if 'h5m_tag' not in layer_data:
            layer_data['h5m_tag'] = name

        thickness_mat = layer_data['thickness_matrix']
        
        # Compute total offset matrix, converting from cm to m
        offset_mat += np.array(thickness_mat)/100

        interp = RegularGridInterpolator(
            (phi_list, theta_list), offset_mat, method = 'cubic'
        )
        
        # Generate component
        try:
            torus, cutter = stellarator_torus(
                vmec, s, tor_ext, repeat, phi_list_exp, theta_list_exp, interp,
                cutter
            )

        except ValueError as e:
            logger.error(e.args[0])
            raise e

        if name not in export_dict['exclude']:
            components[name] = {}
            components[name]['solid'] = torus
            components[name]['h5m_tag'] = layer_data['h5m_tag']

    if export_dict['graveyard']:
        max_offset = np.max(offset_mat)
        components = graveyard(vmec, max_offset, components, logger)

    if magnets is not None:
        magnets['vol_id'] = magnet_coils.magnet_coils(
            magnets, (repeat + 1)*tor_ext, export_dict['dir'], logger = logger
        )

    try:
        exports(export_dict, components, magnets, logger)
    except ValueError as e:
        logger.error(e.args[0])
        raise e
    
    if source is not None:
        strengths = source_mesh.source_mesh(
            vmec, source, export_dict['dir'], logger = logger
        )
        return strengths
    
    # reset cubit to avoid issues when looping parastell
    if export_dict['h5m_export'] == 'Cubit':
        cubit.cmd('reset')


