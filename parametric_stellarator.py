import cadquery as cq
from scipy.optimize import root_scalar
import numpy as np
import read_vmec
from paramak.utils import export_solids_to_dagmc_h5m


def offset_point(zeta, theta, offset):
    """Stellarator offset surface root-finding problem.

    Arguments:
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
    
    # Vary poloidal and toroidal angles by small amount
    r_phi = np.array(vmec.vmec2xyz(1.0, theta, zeta + 0.000001))
    r_theta = np.array(vmec.vmec2xyz(1.0, theta + 0.000001, zeta))
    
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


def root_problem(zeta, theta, phi, offset):
    """Stellarator offset surface root-finding problem.

    Arguments:
        zeta (float): toroidal angle being solved for (rad).
        theta (float): poloidal angle of interest (rad).
        phi (float): toroidal angle of interest (rad).
        offset (float): total offset of layer from plamsa (m).
    
    Returns:
        diff (float): difference between computed toroidal angle and toroidal
            angle of interest (root-finding problem definition).
    """
    # Define offset surface
    x, y, z = offset_point(zeta, theta, offset)

    offset_phi = np.arctan2(y, x)

    if offset_phi < phi - np.pi:
        offset_phi += 2*np.pi

    diff = offset_phi - phi

    return diff


def offset_surface(theta, phi, offset):
    """Computes offset surface point.

    Arguments:
        theta (float): poloidal angle of interest (rad).
        phi (float): toroidal angle of interest (rad).
        offset (float): total offset of layer from plamsa (m).

    Returns:
        r (array): offset suface point (m).
    """
    # Root-solve for the toroidal angle at which the plasma
    # surface normal points to the toroidal angle of interest
    zeta = root_scalar(
        root_problem, args = (theta, phi, offset), x0 = phi, method = 'bisect',
        bracket = [phi - np.pi/4, phi + np.pi/4])
    zeta = zeta.root

    # Compute offset surface point
    r = offset_point(zeta, theta, offset)

    return r


def stellarator_torus(offset, cutter, num_periods):
    """Creates a stellarator helical torus as a CadQuery object.
    
    Arguments:
        offset (float): total offset of layer from plamsa (cm).
        cutter (object): CadQuery solid object used to cut period of
            stellarator torus.
        num_periods (int): number of stellarator periods to build in model.
    
    Returns:
        torus (object): stellarator torus CadQuery solid object.
        cutter(object): updated cutting volume CadQuery solid object.
    """
    # Define the number of phi geometric cross-sections to make
    num_phi = 60
    # Define the number of points defining the geometric cross-section
    num_theta = 100

    # Define toroidal (phi) and poloidal (theta) arrays
    phi = np.linspace(0, np.pi/2, num = num_phi + 1)
    theta = np.linspace(0, 2*np.pi, num = num_theta + 1)

    # Initialize construction
    period = cq.Workplane("XY")

    # Generate poloidal profiles
    for i in phi:
        
        # Initialize points in poloidal profile
        pts = []
        
        # Compute array of points along poloidal profile
        for j in theta[:-1]:
            
            # Conditionally offset poloidal profile
            if offset == 0:
                # Compute plasma edge point
                x, y, z = vmec.vmec2xyz(1.0, j, i)
            
            else:
                # Compute offset surface point
                x, y, z = offset_surface(j, i, offset)
            
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

    # Define torus as conditionally cut period
    torus = period_cut

    # Conditionally generate additional periods
    if num_periods > 1:

        # Define initial angles defining each period needed
        initial_angles = np.linspace(
            90.0, 90.0*(num_periods - 1), num = num_periods - 1
            )

        # Generate additional profiles
        for angle in initial_angles:
            period = period_cut.rotate((0, 0, 0), (0, 0, 1), angle)
            torus = torus.union(period)

    return torus, cutter


def parametric_stellarator(
    plas_eq, radial_build, num_periods, exclude = [], step_export = True,
    h5m_export = True, h5m_tags = None, include_graveyard = False, 
    min_mesh_size = 5.0, max_mesh_size = 20.0, volume_atol = 0.00001,
    center_atol = 0.00001, bounding_box_atol = 0.00001):
    """Generates CadQuery workplane objects for components of a
    parametrically-defined stellarator, based on user-supplied plasma
    equilibrium VMEC data and a user-defined radial build. Each component is
    of uniform thickness, concentric about the plasma edge. The user may
    export .step files for each reactor component and/or a DAGMC-compatible
    .h5m file using Gmsh for use in neutronics simulations.

    Arguments:
        plas_eq (str): path to plasma equilibrium NetCDF file
        radial_build (dict): lists component names and thicknesses in the
            form {'component': thickness (cm)}. Concentric layers will be built
            in the order given.
        num_periods (int): number of stellarator periods to build in model.
        exclude (list of str): names of components not to export.
        step_export (bool): export component .step fiiles
        h5m_export (bool): export DAGMC-compatible neutronics .h5m file using
            Gmsh.
        h5m_tags (list of str): alternative tags to apply to neutronics .h5m
            model. Tags must be in same order as included radial build
            components. Useful when applying vacuum material definition.
        include_graveyard (bool): include automatically generated graveyard
            volume in neutronics .h5m model.
        min_mesh_size (float): minimum mesh element size for Gmsh.
        max_mesh_size (float): maximum mesh element size for Gmsh.
        volume_atol (float): absolute volume tolerance to allow when matching
            parts in intermediate BREP file with CadQuery parts.
        center_atol (float): absolute center coordinates tolerance to allow
            when matching parts in intermediate BREP file with CadQuery parts.
        bounding_box_atol (float): absolute bounding box tolerance  to allow
            when matching parts in intermediate BREP file with CadQuery parts.
    """
    # Load plasma equilibrium data
    global vmec
    vmec = read_vmec.vmec_data(plas_eq)

    # Initialize offset value
    offset = 0.0

    # Initialize solid and name tag storage lists
    solids = []
    tags = []

    # Initialize volume used to cut periods
    cutter = None

    # Generate plasma STEP file
    plasma, cutter = stellarator_torus(offset, cutter, num_periods)

    # Optionally store and export plasma
    if 'plasma' not in exclude:
        solids += [plasma]
        tags += ['plasma']
        if step_export == True:
            cq.exporters.export(plasma, 'plasma.step')
    
    # Initialize offset
    offset = 0.0
    
    # Generate components in radial build
    for layer in radial_build:
        
        # Extract component data
        name = layer
        thickness = radial_build[layer]

        # Compute offset, converting from cm to m
        offset += thickness/100

        # Generate component
        torus, cutter = stellarator_torus(offset, cutter, num_periods)

        # Store solid and name tag, export component
        if name not in exclude:
            solids += [torus]
            tags += [name]
            if step_export == True:
                cq.exporters.export(torus, name + '.step')    

    # Optionally build graveyard volume
    if include_graveyard == True:

        # Determine maximum plasma edge radial position
        R = vmec.vmec2rpz(1.0, 0.0, 0.0)[0]

        # Define length of graveyard and convert from m to cm
        L = 2*(R + offset)*1.25*100

        # Create graveyard volume
        graveyard = cq.Workplane("XY").box(L, L, L).shell(5.0,
            kind = 'intersection')

        # Append graveyard to storage lists
        solids += [graveyard]
        tags += ['Graveyard']

    # Optional DAGMC export
    if h5m_export == True:

        # Optionally apply alternative component tags
        if h5m_tags is not None:
            tags = h5m_tags

        # Export neutronics .h5m file
        export_solids_to_dagmc_h5m(
            solids = solids, tags = tags, filename = 'dagmc.h5m',
            min_mesh_size = min_mesh_size, max_mesh_size = max_mesh_size,
            volume_atol = volume_atol, center_atol = center_atol,
            bounding_box_atol = bounding_box_atol, verbose = True)
