import read_vmec
import cadquery as cq
import cubit
import magnet_coils
import source_mesh
from paramak.utils import export_solids_to_dagmc_h5m
import numpy as np
from scipy.optimize import root_scalar
import os


def cubit_export(solids, h5m_tags, facet_tol, len_tol, norm_tol,
    include_magnets, magnet_h5m_tag):
    """Export H5M neutronics model via Cubit.

    Arguments:
        solids (list): list of component names used for STEP files.
        h5m_tags (list): list of material tags to be used in H5M neutronics
            model.
        facet_tol (float): maximum distance a facet may be from surface of CAD
            representation for Cubit export.
        len_tol (float): maximum length of facet edge for Cubit export.
        norm_tol (float): maximum change in angle between normal vector of
            adjacent facets.
        include_magnets (bool): generate magnets based on user-supplied coil
            data.
        magnet_h5m_tag (str): string to use for material tag in H5M model.
    """
    # Get current working directory
    cwd = os.getcwd()

    # Initialize import command
    impt_cmd = 'import step "' + cwd + '/'

    # If not already initialized by generating magnets, initialize Cubit and
    # volume index
    if include_magnets == False:
        cubit.init([''])
        vol_id = 1
    # If magnets generated, retrieve index of last volume created
    else:
        vol_id = cubit.get_last_id("volume")

    # Import solids
    for solid in solids:
        cubit.cmd(impt_cmd + solid + '.step" heal')

    # Imprint and merge all volumes
    cubit.cmd('imprint volume all')
    cubit.cmd('merge volume all')

    # Conditionally assign magnet material group
    if include_magnets == True:
        cubit.cmd(
            'group "mat:' + magnet_h5m_tag + '" add volume ' + ' '.join(str(i)
            for i in np.linspace(1, vol_id, num = vol_id))
        )
    
    # Assign material groups
    for i, tag in enumerate(h5m_tags, start = vol_id + 1):
        cubit.cmd('group "mat:' + tag + '" add volume ' + str(i))

    # Initialize tolerance strings for export statement as empty strings
    facet_tol_str = ''
    len_tol_str = ''
    norm_tol_str = ''

    # Conditionally fill tolerance strings
    if facet_tol is not None:
        facet_tol_str = 'faceting_tolerance ' + str(facet_tol) + ' '
    if len_tol is not None:
        len_tol_str = 'length_tolerance ' + str(len_tol) + ' '
    if norm_tol is not None:
        norm_tol_str = 'normal_tolerance ' + str(norm_tol) + ' '
    
    # DAGMC export
    cubit.cmd(
        'export dagmc "dagmc.h5m" ' + facet_tol_str + len_tol_str +
        norm_tol_str + 'make_watertight'
    )


def export(
    components, step_export, h5m_export, include_magnets, magnet_h5m_tag, 
    facet_tol, len_tol, norm_tol, min_mesh_size, max_mesh_size, volume_atol,
    center_atol, bounding_box_atol):
    """Export components.

    Arguments:
        components (dict of dicts): dictionary of component names, each with a
            corresponding dictionary of CadQuery solid and material tag to be
            used in H5M neutronics model in the form
            {'component': {'solid': CadQuery solid (object), 'h5m_tag':
            h5m_tag (string)}}
        step_export (bool): export component STEP files
        h5m_export (str): export DAGMC-compatible neutronics H5M file using
            Cubit or Gmsh. Acceptable values are None or a string value of
            'Cubit' or 'Gmsh'. The string is case-sensitive.
        include_magnets (bool): generate magnets based on user-supplied coil
            data.
        magnet_h5m_tag (str): string to use for material tag in H5M model.
        facet_tol (float): maximum distance a facet may be from surface of CAD
            representation for Cubit export.
        len_tol (float): maximum length of facet edge for Cubit export.
        norm_tol (float): maximum change in angle between normal vector of
            adjacent facets.
        min_mesh_size (float): minimum mesh element size for Gmsh export.
        max_mesh_size (float): maximum mesh element size for Gmsh export.
        volume_atol (float): absolute volume tolerance to allow when matching
            parts in intermediate BREP file with CadQuery parts for Gmsh export.
        center_atol (float): absolute center coordinates tolerance to allow
            when matching parts in intermediate BREP file with CadQuery parts
            for Gmsh export.
        bounding_box_atol (float): absolute bounding box tolerance  to allow
            when matching parts in intermediate BREP file with CadQuery parts
            for Gmsh export.
    """
    # Initialize component name, solid, and tag lists
    names = []
    solids = []
    h5m_tags = []
    
    # Populate name, solid, and tag lists
    for name, comp_data in components.items():
        names.append(name)
        solids.append(comp_data['solid'])
        h5m_tags.append(comp_data['h5m_tag'])

    # Conditionally export STEP files
    if step_export == True:
        for name, solid in zip(names, solids):
            cq.exporters.export(solid, name + '.step')
    
    # Conditinally export h5m file via Cubit
    if h5m_export == 'Cubit':
        cubit_export(
            names, h5m_tags, facet_tol, len_tol, norm_tol, include_magnets,
            magnet_h5m_tag)
    
    # Conditionally export h5m file via Gmsh
    if h5m_export == 'Gmsh':
        export_solids_to_dagmc_h5m(
            solids = solids, tags = h5m_tags, filename = 'dagmc.h5m',
            min_mesh_size = min_mesh_size, max_mesh_size = max_mesh_size,
            volume_atol = volume_atol, center_atol = center_atol,
            bounding_box_atol = bounding_box_atol, verbose = False
        )


def graveyard(vmec, offset, components):
    """Creates graveyard component.

    Arguments:
        vmec (object): plasma equilibrium object.
        offset (float): total offset of layer from plamsa (m).
        components (dict of dicts): dictionary of component names, each with a
            corresponding dictionary of CadQuery solid and material tag to be
            used in H5M neutronics model in the form
            {'component': {'solid': CadQuery solid (object), 'h5m_tag':
            h5m_tag (string)}}
    
    Returns:
        components (dict of dicts): dictionary of component names, each with a
            corresponding dictionary of CadQuery solid and material tag to be
            used in H5M neutronics model in the form
            {'component': {'solid': CadQuery solid (object), 'h5m_tag':
            h5m_tag (string)}}
    """
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
    if offset == 0:
        # Compute plasma edge point
        r = vmec.vmec2xyz(1.0, theta, phi)
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
    elif offset < 0:
        raise ValueError('Offset must be greater than or equal to 0')

    return r


def stellarator_torus(
    vmec, num_periods, offset, cutter, gen_periods, num_phi, num_theta):
    """Creates a stellarator helical torus as a CadQuery object.
    
    Arguments:
        vmec (object): plasma equilibrium object.
        num_periods (int): number of periods in stellarator geometry.
        offset (float): total offset of layer from plamsa (cm).
        cutter (object): CadQuery solid object used to cut period of
            stellarator torus.
        gen_periods (int): number of stellarator periods to build in model.
        num_phi (int): number of phi geometric cross-sections to make.
        num_theta (int): number of points defining the geometric cross-section.
    
    Returns:
        torus (object): stellarator torus CadQuery solid object.
        cutter (object): updated cutting volume CadQuery solid object.
    """
    # Determine toroidal extent of each period in radians
    period_ext = 2*np.pi/num_periods
    
    # Define toroidal (phi) and poloidal (theta) arrays
    phi_list = np.linspace(0, period_ext, num = num_phi + 1)
    theta_list = np.linspace(0, 2*np.pi, num = num_theta + 1)[:-1]

    # Convert toroidal extent of period to degrees
    period_ext = np.rad2deg(period_ext)

    # Define initial angles defining each period needed
    initial_angles = np.linspace(
        period_ext, period_ext*(gen_periods - 1), num = gen_periods - 1
    )

    # Convert toroidal extent of period to radians
    period_ext = np.deg2rad(period_ext)

    # Initialize construction
    period = cq.Workplane("XY")

    # Generate poloidal profiles
    for phi in phi_list:
        # Initialize points in poloidal profile
        pts = []
        
        # Compute array of points along poloidal profile
        for theta in theta_list:
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


def parametric_stellarator(
    plas_eq, num_periods, radial_build, gen_periods, num_phi = 60,
    num_theta = 100, exclude = [], step_export = True, h5m_export = None,
    plas_h5m_tag = None, include_graveyard = False, include_magnets = False,
    magnets = None, magnet_data = None, include_source_mesh = False,
    source_mesh_params = None, facet_tol = None, len_tol = None,
    norm_tol = None, min_mesh_size = 5.0, max_mesh_size = 20.0,
    volume_atol = 0.00001, center_atol = 0.00001, bounding_box_atol = 0.00001):
    """Generates CadQuery workplane objects for components of a
    parametrically-defined stellarator, based on user-supplied plasma
    equilibrium VMEC data and a user-defined radial build. Each component is
    of uniform thickness, concentric about the plasma edge. The user may
    export STEP files for each reactor component and/or a DAGMC-compatible
    H5M file using Cubit or Gmsh for use in neutronics simulations.

    Arguments:
        plas_eq (str): path to plasma equilibrium NetCDF file.
        num_periods (int): number of periods in stellarator geometry.
        radial_build (dict of dicts): dictionary of component names, each with
            a corresponding dictionary of layer thickness and optional material
            tag to use in H5M neutronics model in the form
            {'component': {'thickness': thickness (float, cm), 'h5m_tag':
            h5m_tag (str)}}
            Concentric layers will be built in the order given. If no alternate
            material tag is supplied for the H5M file, the given component name
            will be used.
        gen_periods (int): number of stellarator periods to build in model.
        num_phi (int): number of phi geometric cross-sections to make for each
            period (defaults to 60).
        num_theta (int): number of points defining the geometric cross-section
            (defaults to 100).
        exclude (list of str): names of components not to export.
        step_export (bool): export component STEP files (defaults to True).
        h5m_export (str): export DAGMC-compatible neutronics H5M file using
            Cubit or Gmsh. Acceptable values are None or a string value of
            'Cubit' or 'Gmsh' (defaults to None). The string is case-sensitive.
            Note that if magnets are included, 'Cubit' must be used.
        plas_h5m_tag (str): optional alternate material tag to use for plasma.
            If none is supplied and the plasma is not excluded, 'plasma' will
            be used (defaults to None).
        include_graveyard (bool): generate graveyard volume as additional
            component (defaults to False).
        include_magnets (bool): generate magnets based on user-supplied coil
            data (defaults to False).
        magnets (str): path to magnet coil data file (defaults to None).
        magnet_data (dict of dict): dictionary for magnet data including magnet
            name to use for STEP export, width, thickness, material tag to use
            in H5M neutronics model, starting index for data in data file, and
            stopping index for data in data file in the form
            {'name': {'width': width (cm), 'thickness': thickness (cm),
            'h5m_tag': h5m_tag (str), 'start': start (int), 'stop': stop (int)}}
            (defaults to None). All values must be provided.
        include_source_mesh (bool): generate H5M volumetric source mesh based on
            user-supplied plasma equilibrium VMEC data.
        source_mesh_params (dict): parameters to use defining source mesh
            including numbers of closed flux surfaces, poloidal angles, and
            toroidal angles in the form
            {'num_s': num_s (int), 'num_theta': num_theta (int),
            'num_phi': num_phi (int)}
        facet_tol (float): maximum distance a facet may be from surface of CAD
            representation for Cubit export.
        len_tol (float): maximum length of facet edge for Cubit export
            (defaults to None).
        norm_tol (float): maximum change in angle between normal vector of
            adjacent facets (defaults to None).
        min_mesh_size (float): minimum mesh element size for Gmsh export
            (defaults to 5.0).
        max_mesh_size (float): maximum mesh element size for Gmsh export
            (defaults to 20.0).
        volume_atol (float): absolute volume tolerance to allow when matching
            parts in intermediate BREP file with CadQuery parts for Gmsh export
            (defaults to 0.00001).
        center_atol (float): absolute center coordinates tolerance to allow
            when matching parts in intermediate BREP file with CadQuery parts
            for Gmsh export (defaults to 0.00001).
        bounding_box_atol (float): absolute bounding box tolerance  to allow
            when matching parts in intermediate BREP file with CadQuery parts
            for Gmsh export (defaults to 0.00001).
    """
    # Check if total toroidal extent exceeds 360 degrees
    if gen_periods > num_periods:
        raise ValueError(
            'Number of requested periods to generate exceeds number in '
            'stellarator geometry'
        )

    # Check that h5m_export has an appropriate value
    if h5m_export != None and h5m_export != 'Cubit' and h5m_export != 'Gmsh':
        raise ValueError(
            'h5m_export must be None or have a string value of \'Cubit\' or '
            '\'Gmsh\''
        )

    # Check that Cubit export has STEP files to use
    if step_export == False and h5m_export == 'Cubit':
        raise ValueError('H5M export via Cubit requires STEP files')

    # Check that H5M export of magnets uses Cubit
    if h5m_export == 'Gmsh' and include_magnets == True:
        raise ValueError(
            'Inclusion of magnets in H5M model requires Cubit export'
        )
    
    # Load plasma equilibrium data
    vmec = read_vmec.vmec_data(plas_eq)

    # Initialize component storage dictionary
    components = {}
    
    # Prepend plasma to radial build
    full_radial_build = {'plasma': {'thickness': 0}, **radial_build}
    
    # Initialize offset value
    offset = 0.0

    # Initialize volume used to cut periods
    cutter = None
    
    # Generate components in radial build
    for name, layer_data in full_radial_build.items():
        
        # Notify which component is being generated
        print('Generating ' + str(name) + '...')
        
        # Conditionally assign plasma h5m tag
        if name == 'plasma':
            if plas_h5m_tag is not None:
                layer_data['h5m_tag'] = plas_h5m_tag
        
        # Conditionally populate h5m tag for layer
        if 'h5m_tag' not in layer_data:
            layer_data['h5m_tag'] = name

        # Extract layer thickness
        thickness = layer_data['thickness']
        # Compute offset, converting from cm to m
        offset += thickness/100

        # Generate component
        torus, cutter = stellarator_torus(
            vmec, num_periods, offset, cutter, gen_periods, num_phi, num_theta
        )

        # Store solid and name tag
        if name not in exclude:
            components[name] = {}
            components[name]['solid'] = torus
            components[name]['h5m_tag'] = layer_data['h5m_tag']

    # Conditionally build graveyard volume
    if include_graveyard == True:
        components = graveyard(vmec, offset, components)

    # Conditionally build magnet coils
    if include_magnets == True:
        # Extract coil data
        magnet_name = list(magnet_data.keys())[0]
        coil_width = list(magnet_data.values())[0]['width']
        coil_thickness = list(magnet_data.values())[0]['thickness']
        magnet_h5m_tag = list(magnet_data.values())[0]['h5m_tag']
        start = list(magnet_data.values())[0]['start']
        stop = list(magnet_data.values())[0]['stop']
        # Generate magnets
        magnet_coils.magnet_coils(
            magnets, coil_width, coil_thickness, start, stop, magnet_name)

    # Export components
    export(
        components, step_export, h5m_export, include_magnets, magnet_h5m_tag,
        facet_tol, len_tol, norm_tol, min_mesh_size, max_mesh_size,
        volume_atol, center_atol, bounding_box_atol
    )

    # Conditionally create source mesh
    if include_source_mesh == True:
        num_s = source_mesh_params['num_s']
        num_theta = source_mesh_params['num_theta']
        num_phi = source_mesh_params['num_phi']
        source_mesh.source_mesh(plas_eq, num_s, num_theta, num_phi)
