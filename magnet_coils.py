import cubit
import numpy as np


def magnet_coils(
    file, cross_section, start, stop = None, name = 'magnet_coils'):
    """Generates STEP file using Cubit for stellarator magnet coils based on
    user-supplied coil data. The coils have rectangular cross-section.

    Arguments:
        coils (str): path to magnet coil data file.
        cross_section (list): coil cross-section definiton. Cross-section shape
            must be either a circle or rectangle. For a circular cross-section,
            the list format is
            ['circle' (str), radius (float, cm)]
            For a rectangular cross-section, the list format is
            ['rectangle' (str), width (float, cm), thickness (float, cm)]
        start (int): index for line in data file where coil data begins.
        stop (int): index for line in data file where coil data ends (defaults
            to None).
        name (str): name used for exported STEP file.
    """
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
    
    # Load magnet coil data
    file_obj = open(file, 'r')
    data = file_obj.readlines()[start:stop]

    # Initialize Cubit
    cubit.init([''])
    cubit.cmd('undo on')

    # Initialize vertex, curve, and surface indices
    vert_id = 0
    curve_id = 0
    surf_id = 0

    # Initialize path list
    path = []

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
        
        # Create vertex in path
        cubit.cmd('create vertex ' + str(x) + ' ' + str(y) + ' ' + str(z))
        # Update vertex index
        vert_id += 1
        # Append vertex to path list
        path += [vert_id]
        
        # Current equal to zero signifies final point in coil
        # If final point reached, create path and sweep coil cross-section to
        # create solid
        if s == 0:
            path[-1] = path[0]
            
            # Create path
            cubit.cmd(
                'create curve spline location vertex ' + ' '.join(str(i) for i
                in path)
            )
            # Update curve index
            curve_id += 1
            
            # Create cross-section for sweep
            if shape == 'circle':
                cubit.cmd(
                    'create surface circle radius ' + str(radius) + ' zplane'
                )
            else:
                cubit.cmd(
                    'create surface rectangle width ' + str(thickness)
                    + ' height ' + str(width) + ' zplane'
                )

            # Update surface index
            surf_id += 1

            # Define new surface normal vector as that pointing between path
            # points adjacent to initial point

            # Extract path points adjacent to initial point
            next_pt = cubit.vertex(path[1]).coordinates()
            next_pt = np.array(next_pt)
            last_pt = cubit.vertex(path[-2]).coordinates()
            last_pt = np.array(last_pt)
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
                'rotate Surface ' + str(surf_id) + ' angle ' + str(rot_ang_norm)
                + ' about origin 0 0 0 direction ' + ' '.join(str(i) for i in
                rot_axis) + ' include_merged'
            )

            # If cross-section shape is rectangular, orient cross-section in
            # normal plane such that its vector pointing in the direction of its
            # thickness faces the origin
            if shape == 'rectangle':
                # Extract initial path point
                pos = cubit.vertex(path[0]).coordinates()

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
                    'rotate Surface ' + str(surf_id) + ' angle '
                    + str(rot_ang_orig) + ' about origin 0 0 0 direction '
                    + ' '.join(str(i) for i in norm) + ' include_merged'
                )
            
            # Move cross-section to initial path point
            cubit.cmd(
                'move Surface ' + str(surf_id) + ' location vertex '
                + str(path[0]) + ' include_merged'
            )

            # Sweep cross-section to create magnet coil
            cubit.cmd('undo group begin')
            cubit.cmd(
                'sweep surface ' + str(surf_id) + ' along curve '
                + str(curve_id) + ' individual'
            )
            cubit.cmd('delete curve ' + str(curve_id))
            cubit.cmd('undo group end')
            cubit.cmd('delete vertex all')
            
            # Update vertex, curve, and surface indices
            vert_id = cubit.get_last_id("vertex")
            curve_id = cubit.get_last_id("curve")
            surf_id = cubit.get_last_id("surface")

            # Reinitialize path list
            path = []

    # Export magnet coils
    cubit.cmd('export step "' + name + '.step"  overwrite')
