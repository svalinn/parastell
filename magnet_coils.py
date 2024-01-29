import log
import cubit
import numpy as np


def mesh_magnets(vol_ids):
    """Creates tetrahedral mesh of magnet volumes.

    Arguments:
        vol_ids (list of int): indices for magnet volumes.
    """
    # Loop over magnet indices
    for vol in vol_ids:
        # Create scheme and meshes
        cubit.cmd(f'volume {vol} scheme tetmesh')
        cubit.cmd(f'mesh volume {vol}')


def cut_mags(tor_ext, vol_ids):
    """Cuts magnet volumes such that only the specified toroidal extent is
    included.
    
    Arguments:
        tor_ext (float): toroidal extent to model (deg).
        vol_ids (list of int): indices for magnet volumes.
    
    Returns:
        vol_ids (list of int): updated indices for magnet volumes.
    """
    # Create surface
    cubit.cmd('create surface rectangle width 4000 yplane')
    # Extract surface index
    surf_id = cubit.get_last_id("surface")
    # Move surface
    cubit.cmd(f'move Surface {surf_id} x 2000')
    # Sweep surface
    cubit.cmd(f'sweep surface {surf_id} zaxis angle {tor_ext}')
    # Extract volume index
    sweep_id = cubit.get_last_id("volume")

    # Intersect magnet volumes with sweep volume
    cubit.cmd(
        'intersect volume ' + ' '.join(str(i) for i in vol_ids)
        + f' {sweep_id}'
    )

    # Extract index of most recently created volume
    last_id = cubit.get_last_id("volume")
    # Compute magnet volume indices
    vol_ids = range(sweep_id + 1, last_id + 1)

    return vol_ids


def unit_vector(vec):
    """Normalizes given vector.

    Arguments:
        vec (array of [float, float, float]): input vector to be normalized.
    
    Returns:
        vec/np.linalg.norm(vec) (array of [float, float, float]): normalized
            input vector.
    """
    return vec/np.linalg.norm(vec)


def orient_rectangle(path_origin, surf_id, t_vec, norm, rot_axis, rot_ang_norm):
    """Orients rectangular cross-section in the normal plane such that its
    thickness direction faces the origin.
    
    Arguments:
        path_origin (int): index of initial point in filament path.
        surf_id (int): index of cross-section surface.
        t_vec (list of float): cross-section thickness vector.
        norm (list of float): cross-section normal vector.
        rot_axis (list of float): axis about which to rotate the cross-section.
        rot_ang_norm (float): angle by which cross-section was rotated to align
            its normal with the initial point tangent (deg).
    """
    # Determine orientation of thickness vector after cross-section was
    # oriented along filament origin tangent

    # Compute part of thickness vector parallel to rotation axis
    t_vec_par = unit_vector(np.inner(t_vec, rot_axis)*rot_axis)
    # Compute part of thickness vector orthogonal to rotation axis
    t_vec_perp = unit_vector(t_vec - t_vec_par)

    # Compute vector othogonal to both rotation axis and orthogonal
    # part of thickness vector
    orth = unit_vector(np.cross(rot_axis, t_vec_perp))
    
    # Determine part of rotated vector parallel to original
    rot_par = np.cos(rot_ang_norm)
    # Determine part of rotated vector orthogonal to original
    rot_perp = np.sin(rot_ang_norm)

    # Compute orthogonal part of thickness vector after rotation
    t_vec_perp_rot = rot_par*t_vec_perp + rot_perp*orth
    # Compute thickness vector after rotation
    t_vec_rot = unit_vector(t_vec_perp_rot + t_vec_par)

    # Orient cross-section in its plane such that it faces the global origin

    # Extract initial path point
    pos = cubit.vertex(path_origin).coordinates()

    # Project position vector onto cross-section
    pos_proj = unit_vector(pos - np.inner(pos, norm)*norm)

    # Compute angle by which to rotate cross-section such that it faces the
    # origin
    rot_ang_orig = np.arccos(np.inner(pos_proj, t_vec_rot))
    
    # Re-orient rotated cross-section such that thickness vector faces origin
    cubit.cmd(
        f'rotate Surface {surf_id} angle {np.rad2deg(rot_ang_orig)} about '
        'origin 0 0 0 direction ' + ' '.join(str(i) for i in norm)
    )


def extract_cs(cross_section, logger):
    """Extract coil cross-section parameters

    Arguments:
        cross_section (list or tuple of str, float, float): coil cross-section
            definition. Note that the cross-section shape must be either a
            circle or rectangle.
            For a circular cross-section, the list format is
            ['circle' (str), radius (float, cm)]
            For a rectangular cross-section, the list format is
            ['rectangle' (str), width (float, cm), thickness (float, cm)]
        logger (object): logger object.

    Returns:
        shape_str (str): string to pass to Cubit for cross-section generation.
            For a circular cross-section, the string format is
            '{shape} radius {radius}'
            For a rectangular cross-section, the string format is
            '{shape} width {thickness} height {width}'
    """
    # Extract coil cross-section shape
    shape = cross_section[0]
    
    # Conditionally extract parameters for circular cross-section
    if shape == 'circle':
        # Check that list format is correct
        if len(cross_section) == 1:
            raise ValueError(
                'Format of list defining circular cross-section must be\n'
                '[\'circle\' (str), radius (float, cm)]'
            )
        elif len(cross_section) > 2:
            logger.warning(
                'More than one length dimension has been defined for '
                'cross_section. Interpreting the first as the circle\'s radius;'
                ' did you mean to use \'rectangle\'?'
            )
        # Extract parameters
        radius = cross_section[1]
        # Define string to pass to Cubit for cross-section generation
        shape_str = f'{shape} radius {radius}'
    # Conditinally extract parameters for rectangular cross-section
    elif shape == 'rectangle':
        # Check that list format is correct
        if len(cross_section) != 3:
            raise ValueError(
                'Format of list defining rectangular cross-section must be\n'
                '[\'rectangle\' (str), width (float, cm), thickness '
                '(float, cm)]'
            )
        # Extract parameters
        width = cross_section[1]
        thickness = cross_section[2]
        # Define string to pass to Cubit for cross-section generation
        shape_str = f'{shape} width {thickness} height {width}'
    # Otherwise, if input string is neither 'circle' nor 'rectangle', raise an
    # exception
    else:
        raise ValueError(
            'Magnet cross-section must be either a circle or rectangle. The '
            'first entry of the list defining the cross-section must be the '
            'shape, with the following entries defining the shape parameters.\n'
            '\n'
            'For a circular cross-section, the list format is\n'
            '[\'circle\' (str), radius (float, cm)]\n'
            '\n'
            'For a rectangular cross-section, the list format is\n'
            '[\'rectangle\' (str), width (float, cm), thickness (float, cm)]'
        )

    return shape_str


def create_magnets(filaments, cross_section, meshing, logger):
    """Creates magnet coil solids.
    
    Arguments:
        filaments (list of list of list of float): list of filament coordinates.
            Each filament is a list of coordinates.
        cross_section (list or tuple of str, float, float): coil cross-section
            definition. Note that the cross-section shape must be either a
            circle or rectangle.
            For a circular cross-section, the list format is
            ['circle' (str), radius (float, cm)]
            For a rectangular cross-section, the list format is
            ['rectangle' (str), width (float, cm), thickness (float, cm)]
        meshing (bool): setting for tetrahedral mesh generation
        logger (object): logger object.

    Returns:
        vol_ids (list of int): indices for magnet volumes.
    """
    # Extract cross-section parameters
    try:
        shape_str = extract_cs(cross_section, logger)
    except ValueError as e:
        logger.error(e.args[0])
        raise e

    # Cross-section inititally populated with thickness vector
    # oriented along x axis
    t_vec = np.array([1, 0, 0])
    
    # Create cross-section for sweep
    cubit.cmd(f'create surface ' + shape_str + ' zplane')

    # Store cross-section index
    cs_id = cubit.get_last_id("surface")
    # Cross-section initially populated with normal oriented along z
    # axis
    cs_axis = np.array([0, 0, 1])

    # Initialize volume index storage list
    vol_ids = []

    # Initialize path list
    path = []

    # Extract filament data
    for filament in filaments:
        
        # Create vertices in filament path
        for x, y, z in filament:
            # Create vertex in path
            cubit.cmd(f'create vertex {x} {y} {z}')
            # Append vertex to path list
            path += [cubit.get_last_id("vertex")]
        
        # Ensure final vertex in path is the same as the first
        path += [path[0]]
        
        # Create spline for path
        cubit.cmd(
            f'create curve spline location vertex ' +
            ' '.join(str(i) for i in path)
        )
        # Store curve index
        curve_id = cubit.get_last_id("curve")

        # Define new surface normal vector as that pointing between path
        # points adjacent to initial point

        # Extract path points adjacent to initial point
        next_pt = np.array(cubit.vertex(path[1]).coordinates())
        last_pt = np.array(cubit.vertex(path[-2]).coordinates())
        # Compute tangent at path origin (direction in which to align surface
        # normal)
        tang = unit_vector(np.subtract(next_pt, last_pt))
        
        # Define axis and angle of rotation to orient cross-section along
        # defined normal

        # Define axis of rotation as orthogonal to both z axis and surface
        # normal
        rot_axis = unit_vector(np.cross(cs_axis, tang))
        # Compute angle by which to rotate cross-section to orient along
        # defined surface normal
        rot_ang_norm = np.arccos(np.inner(cs_axis, tang))

        # Copy cross-section for sweep
        cubit.cmd(f'surface {cs_id} copy')
        # Store surface index
        surf_id = cubit.get_last_id("surface")
        
        # Orient cross-section along defined normal
        cubit.cmd(
            f'rotate Surface {surf_id} angle {np.rad2deg(rot_ang_norm)} about '
            'origin 0 0 0 direction ' + ' '.join(str(i) for i in rot_axis)
        )

        # Conditionally orients rectangular cross-section
        if cross_section[0] == 'rectangle':
            orient_rectangle(
                path[0], surf_id, t_vec, tang, rot_axis, rot_ang_norm
            )
            
        # Move cross-section to initial path point
        cubit.cmd(f'move Surface {surf_id} location vertex {path[0]}')

        # Sweep cross-section to create magnet coil
        cubit.cmd(
            f'sweep surface {surf_id} along curve {curve_id} '
            f'individual'
        )
        # Store volume index
        volume_id = cubit.get_last_id("volume")
        vol_ids.append(volume_id)
        # Delete extraneous curves and vertices
        cubit.cmd(f'delete curve {curve_id}')
        cubit.cmd('delete vertex all')

        # Reinitialize path list
        path = []
    
    # Delete original cross-section
    cubit.cmd(f'delete surface {cs_id}')

    return vol_ids


def clean_mag_data(filaments, tor_ext):
    """Cleans filament data such that only filaments within the toroidal extent
    of the model are included and filaments are sorted by toroidal angle.

    Arguments:
        filaments (list of list of list of float): list of filament coordinates.
            Each filament is a list of coordinates.
        tor_ext (float): toroidal extent to model (deg).
    
    Returns:
        sorted_fils (list of list of list of float): sorted list of filament
            coordinates.
    """
    # Initialize data for filaments within toroidal extent of model
    reduced_fils = []
    # Initialize list of toroidal angles of filament centers of mass
    phi_list = []

    # Define tolerance of toroidal extent (accounts for width of coils)
    tol = np.deg2rad(5)

    # Loop over filaments
    for fil in filaments:
        # Initialize counter of number of points within toroidal extent
        counter = 0
        # Initialize center of mass
        com = np.array([0, 0, 0])
        
        # Loop over filament points
        for coords in fil:
            # Extract x coordinate
            x = coords[0]
            # Extract y coordinate
            y = coords[1]
            # Compute toroidal angle of point
            phi = np.arctan2(y, x)
            # Conditionally iterate counter
            if (
                phi < np.deg2rad(tor_ext) + tol and
                phi > 0 - tol
            ):
                counter += 1
            
            # Cumulatively sum center of mass
            com = com + np.array(coords)

        # Average center of mass coordinates
        com = com/len(fil)

        # If no points within toroidal extent, exclude filament by continuing
        # to next iteration of loop
        if counter == 0 and tor_ext < 360.0:
            continue
        # Otherwise, append filament to storage list
        reduced_fils.append(fil)

        # Extract x coordinate of center of mass
        x = com[0]
        # Extract y coordinate of center of mass
        y = com[1]
        # Compute toroidal angle of center of mass
        phi = np.arctan2(y, x)
        # Conditionally rotate toroidal angle such that it is positive
        if phi < 0:
            phi = phi + 2*np.pi
        # Append toroidal angle to storage list
        phi_list.append(phi)

    # Sort filaments by toroidal angle
    sorted_fils = [x for _,x in sorted(zip(phi_list, reduced_fils))]

    return sorted_fils


def extract_filaments(file, start, stop, sample):
    """Extracts filament data from magnet coil data file.
    
    Arguments:
        file (str): path to magnet coil data file.
        start (int): index for line in data file where coil data begins.
        stop (int): index for line in data file where coil data ends (defaults
            to None).
        sample (int): sampling modifier for filament points.

    Returns:
        filaments (list of list of list of float): list of filament coordinates.
            Each filament is a list of coordinates.
    """
    # Initialize list of filaments
    filaments = []

    # Load magnet coil data
    file_obj = open(file, 'r')
    data = file_obj.readlines()[start:stop]

    # Initialize list of coordinates for each filament
    coords = []

    # Extract magnet coil data
    for i, line in enumerate(data):
        # Parse line in magnet coil data
        columns = line.strip().split()

        # Break at indicator signaling end of data
        if columns[0] == 'end':
            break

        # Extract coordinates for vertex in path
        x = float(columns[0])*100
        y = float(columns[1])*100
        z = float(columns[2])*100
        # Extract coil current
        s = float(columns[3])

        # Coil current of zero signals end of filament
        # If current is not zero, conditionally store coordinate in filament
        # list
        if s != 0:
            # Only store every five points
            if i % sample == 0:
                # Append coordinates to list
                coords.append([x, y, z])
        # Otherwise, store filament coordinates but do not append final
        # filament point. In Cubit, continuous curves are created by setting
        # the initial and final vertex indices equal. This is handled in the
        # create_magnets method
        else:
            # Append list of coordinates to list of filaments
            filaments.append(coords)
            # Reinitialize list of coordinates
            coords = []

    return filaments


def magnet_coils(magnets, tor_ext, logger = None):
    """Generates STEP file using Cubit for stellarator magnet coils based on
    user-supplied coil data. The coils have rectangular cross-section.

    Arguments:
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
            }
            For the list defining the coil cross-section, the cross-section
            shape must be either a circle or rectangle. For a circular
            cross-section, the list format is
            ['circle' (str), radius (float, cm)]
            For a rectangular cross-section, the list format is
            ['rectangle' (str), width (float, cm), thickness (float, cm)]
        tor_ext (float): toroidal extent to model (deg).
        logger (object): logger object (defaults to None). If no logger is
            supplied, a default logger will be instantiated.

    Returns:
        vol_ids (list of int): indices for magnet volumes.
    """
    # Conditionally instantiate logger
    if logger == None or not logger.hasHandlers():
        logger = log.init()
    
    # Signal magnet generation
    logger.info(f'Building {magnets["name"]}...')
    
    # Extract filament data
    filaments = extract_filaments(
        magnets['file'], magnets['start'], magnets['stop'], magnets['sample']
    )

    # Clean magnet data
    sorted_fils = clean_mag_data(filaments, tor_ext)

    # Generate magnet coil solids
    vol_ids = create_magnets(
        sorted_fils, magnets['cross_section'], magnets['meshing'], logger
    )

    # Conditionally cut magnets according to toroidal extent
    if tor_ext < 360.0:
        vol_ids = cut_mags(tor_ext, vol_ids)
    
    # Export magnet coils
    cubit.cmd(f'export step "{magnets["name"]}.step"  overwrite')

    # Optional tetrahedral mesh functionality
    if magnets['meshing']:
        mesh_magnets(vol_ids)

    return vol_ids
