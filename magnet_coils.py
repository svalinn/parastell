import cubit
import numpy as np


def orient_rectangle(path_origin, surf_id, norm, rot_axis, rot_ang_norm):
    """Orients rectangular cross-section in the normal plane such that its thickness direction faces the origin.
    
    Arguments:
        path_origin (int): index of initial point in filament path.
        surf_id (int): index of cross-section surface.
        norm (list of float): cross-section normal vector.
        rot_axis (list of float): axis about which to rotate the cross-section.
        rot_ang_norm (float): angle by which cross-section was rotated to align
            its normal with the initial point tangent (deg).
    """
    # Extract initial path point
    pos = cubit.vertex(path_origin).coordinates()

    # Cross-section inititally populated with thickness vector
    # oriented along x axis
    t_vec = np.array([1, 0, 0])
    # Compute part of thickness vector parallel to rotation axis
    t_vec_par = (np.inner(t_vec, rot_axis)/np.inner(rot_axis,
        rot_axis))*rot_axis
    # Compute part of thickness vector orthogonal to rotation axis
    t_vec_perp = t_vec - t_vec_par
    # Compute magnitude of orthogonal part of thickness vector
    t_vec_perp_mag = np.linalg.norm(t_vec_perp)

    # Compute vector othogonal to both rotation axis and orthogonal
    # part of thickness vector
    orth = np.cross(rot_axis, t_vec_perp)
    # Compute magnitude of orthgonal vector
    orth_mag = np.linalg.norm(orth)

    # Convert angle from degress to radians
    rot_ang_norm = np.deg2rad(rot_ang_norm)
    
    # Determine part of rotated vector parallel to original
    rot_par = np.cos(rot_ang_norm)/t_vec_perp_mag
    # Determine part of rotated vector orthogonal to original
    rot_perp = np.sin(rot_ang_norm)/orth_mag

    # Compute orthogonal part of thickness vector after rotation
    t_vec_perp_rot = t_vec_perp_mag*(rot_par*t_vec_perp
        + rot_perp*orth)
    # Compute thickness vector after rotation
    t_vec_rot = t_vec_perp_rot + t_vec_par
    # Compute magnitude of thickness vector
    t_vec_rot_mag = np.linalg.norm(t_vec_rot)

    # Project position vector onto cross-section
    pos_proj = pos - (np.inner(pos, norm)/np.inner(pos, pos))*norm
    # Compute magnitude of projection of position vector
    pos_proj_mag = np.linalg.norm(pos_proj)

    # Compute angle by which to rotate cross-section such that it
    # faces the origin
    dot_prod = np.inner(pos_proj, t_vec_rot)
    rot_ang_orig = np.arccos(dot_prod/t_vec_rot_mag/pos_proj_mag)

    # Convert angle from radians to degrees
    rot_ang_orig = np.rad2deg(rot_ang_orig)
    
    # Re-orient rotated cross-section such that thickness vector
    # faces origin
    cubit.cmd(
        f'rotate Surface {surf_id} angle {rot_ang_orig} about  origin 0 0 0 '
        f'direction {" ".join(str(i) for i in norm)} include_merged'
    )


def create_magnets(
    filaments, shape, radius = None, width = None, thickness = None):
    """Creates magnet coil solids.
    
    Arguments:
        filaments (list of list of list of float): list filament coordinates.
            Each filament is a list of coordinates.
        shape (str): shape of cross-section (must be 'circle' or 'rectangle').
        radius (float): radius of circular cross-section (defaults to None).
        width (float): width of rectangular cross-seciton (defaults to None).
        thickness (float): thickness of rectangular cross-section (defaults to
            None).
    """
    # Create cross-section for sweep
    if shape == 'circle':
        cubit.cmd(
            f'create surface circle radius {radius} zplane'
        )
    else:
        cubit.cmd(
            f'create surface rectangle width {thickness} height {width} zplane'
        )
    # Store cross-section index
    cs_id = cubit.get_last_id("surface")

    # Initialize path list
    path = []

    # Extract filament data
    for filament in filaments:
        
        # Create vertices in filament path
        for coords in filament:
            # Extract Cartesian coordinates in filament
            x, y, z = coords
            # Create vertex in path
            cubit.cmd(f'create vertex {x} {y} {z}')
            # Append vertex to path list
            path += [cubit.get_last_id("vertex")]
        
        # Ensure final vertex in path is the same as the first
        path += [path[0]]
        
        # Create spline for path
        cubit.cmd(
            f'create curve spline location vertex '
            f'{" ".join(str(i) for i in path)}'
        )
        # Store curve index
        curve_id = cubit.get_last_id("curve")
        
        # Copy cross-section for sweep
        cubit.cmd(f'surface {cs_id} copy')
        # Store surface index
        surf_id = cubit.get_last_id("surface")

        # Define new surface normal vector as that pointing between path
        # points adjacent to initial point

        # Extract path points adjacent to initial point
        next_pt = np.array(cubit.vertex(path[1]).coordinates())
        last_pt = np.array(cubit.vertex(path[-2]).coordinates())
        # Compute surface normal
        norm = np.subtract(next_pt, last_pt)
        # Compute magnitude of surface normal
        norm_mag = np.linalg.norm(norm)
        
        # Define axis and angle of rotation to orient cross-section along
        # defined normal

        # Cross-section initially populated with normal oriented along z
        # axis
        z_axis = np.array([0, 0, 1])
        # Define axis of rotation as orthogonal to both z axis and surface
        # normal
        rot_axis = np.cross(z_axis, norm)
        # Compute angle by which to rotate cross-section to orient along
        # defined surface normal
        dot_prod = np.inner(z_axis, norm)
        rot_ang_norm = np.arccos(dot_prod/norm_mag)

        # Covert angle from radians to degrees
        rot_ang_norm = np.rad2deg(rot_ang_norm)

        # Orient cross-section along defined normal
        cubit.cmd(
            f'rotate Surface {surf_id} angle {rot_ang_norm} about '
            f'origin 0 0 0 direction {" ".join(str(i) for i in rot_axis)} '
            f'include_merged'
        )

        # Conditionally orients rectangular cross-section
        if shape == 'rectangle':
            orient_rectangle(path[0], surf_id, norm, rot_axis, rot_ang_norm)
            
        # Move cross-section to initial path point
        cubit.cmd(
            f'move Surface {surf_id} location vertex {path[0]} '
            f'include_merged'
        )

        # Sweep cross-section to create magnet coil
        cubit.cmd(
            f'sweep surface {surf_id} along curve {curve_id} '
            f'individual'
        )
        # Delete extraneous curves and vertices
        cubit.cmd(f'delete curve {curve_id}')
        cubit.cmd('delete vertex all')

        # Reinitialize path list
        path = []
    
    # Delete original cross-section
    cubit.cmd(f'delete surface {cs_id}')


def extract_filaments(file, start, stop):
    """Extracts filament data from magnet coil data file.
    
    Arguments:
        file (str): path to magnet coil data file.
        start (int): index for line in data file where coil data begins.
        stop (int): index for line in data file where coil data ends (defaults
            to None).

    Returns:
        filaments (list of list of list of float): list filament coordinates.
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
    for line in data:
        # Parse line in magnet coil data
        line = line.strip()
        columns = line.split()

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
        if s == 0:
            # Append list of coordinates to list of filaments
            filaments.append(coords)
            # Reinitialize list of coordinates
            coords = []
            # Do not include final coordinate as it is not guaranteed to be 
            # exactly equal to initial
            continue

        # Append coordinates to list
        coords.append([x, y, z])

    return filaments


def magnet_coils(magnets):
    """Generates STEP file using Cubit for stellarator magnet coils based on
    user-supplied coil data. The coils have rectangular cross-section.

    Arguments:
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
    # Extract coil data
    file = magnets['file']
    cross_section = magnets['cross_section']
    start = magnets['start']
    stop = magnets['stop']
    name = magnets['name']
    
    # Extract coil cross-section shape
    shape = cross_section[0]
    # Conditionally extract parameters for circular cross-section
    if shape == 'circle':
        if len(cross_section) != 2:
            raise ValueError(
                'Format of list defining circular cross-section must be\n'
                '[\'circle\' (str), radius (float, cm)]'
            )
        radius = cross_section[1]
    # Conditinally extract parameters for rectangular cross-section
    elif shape == 'rectangle':
        if len(cross_section) != 3:
            raise ValueError(
                'Format of list defining rectangular cross-section must be\n'
                '[\'rectangle\' (str), width (float, cm), thickness '
                '(float, cm)]'
            )
        width = cross_section[1]
        thickness = cross_section[2]
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
    
    # Extract filament data
    filaments = extract_filaments(file, start, stop)

    # Generate magnet coil solids
    if shape == 'circle':
        create_magnets(filaments, shape, radius = radius)
    else:
        create_magnets(filaments, shape, width = width, thickness = thickness)
    
    # Export magnet coils
    cubit.cmd(f'export step "{name}.step"  overwrite')
