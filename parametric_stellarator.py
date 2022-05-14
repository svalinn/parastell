import cadquery as cq
import numpy as np
import read_vmec
from scipy.optimize import fsolve
from paramak.utils import export_solids_to_dagmc_h5m


def root_problem(zeta):
    """Stellarator offset surface root-finding problem.

    Arguments:
        zeta (float): toroidal angle being solved for (rad)
    
    Returns:
        Root-finding problem definition
    """

    # Convert data type from list to float
    zeta = zeta[0]
    # Define plasma edge
    r = np.array(vmec.vmec2xyz(1.0, psi, zeta))
    
    # Vary poloidal and toroidal angles by small amount
    r_phi = np.array(vmec.vmec2xyz(1.0, psi, zeta + 0.001))
    r_theta = np.array(vmec.vmec2xyz(1.0, psi + 0.001, zeta))
    
    # Take difference from plasma edge point
    delta_phi = r_phi - r
    delta_theta = r_theta - r

    # Compute surface normal
    norm = np.cross(delta_phi, delta_theta)
    norm_mag = np.sqrt(sum(k**2 for k in norm))
    n = norm/norm_mag

    # Define offset surface
    x, y, z = r + offset*n

    return np.arctan(y/x) - alpha


def offset_surface(theta, phi):
    """Computes offset surface point.

    Arguments:
        theta (float): poloidal angle of interest (rad)
        phi (float): toroidal angle of interest (rad)

    Returns:
        r (array): offset suface point (m)
    """

    # Globalize poloidal angle under name psi for root finder
    global psi
    psi = theta
    # Globalize toroidal angle under name alpha for root finder
    global alpha
    alpha = phi

    # Root-solve for the toroidal angle at which the plasma
    # surface normal points to the toroidal angle of interest
    zeta = fsolve(root_problem, phi)
    zeta = zeta[0]

    # Compute plasma edge at root toroidal angle
    r_root = np.array(vmec.vmec2xyz(1.0, psi, zeta))
    
    # Vary poloidal and toroidal angles by small amount
    r_phi = np.array(vmec.vmec2xyz(1.0, psi, zeta + 0.001))
    r_theta = np.array(vmec.vmec2xyz(1.0, psi + 0.001, zeta))
    
    # Take difference from plasma edge point
    delta_phi = r_phi - r_root
    delta_theta = r_theta - r_root

    # Compute surface normal
    norm = np.cross(delta_phi, delta_theta)
    norm_mag = np.sqrt(sum(k**2 for k in norm))
    n = norm/norm_mag

    # Compute offset surface point
    r = r_root + offset*n

    return r


def stellarator_torus():
    """Creates a stellarator helical torus as a CadQuery object.

    Returns:
        torus (object): stellarator torus CadQuery object
    """

    # Define the number of phi geometric cross-sections to make
    num_phi = 60
    # Define the number of points defining the geometric cross-section
    num_theta = 30

    # Define toroidal (phi) and poloidal (theta) arrays
    phi = np.linspace(0, np.pi/2, num = num_phi + 1)
    theta = np.linspace(0, 2*np.pi, num = num_theta + 1)

    # Generate poloidal profiles
    for i in phi:

        # Define rotation vector to orient new workplane at the toroidal
        # angle
        rotation = cq.Vector(0.0, np.rad2deg(i), 0.0)

        # Define new workplane oriented along magnetic axis
        if i == phi[0]:
            period = cq.Workplane("XZ").transformed(rotate = rotation)
        else:
            period = period.copyWorkplane(cq.Workplane("XZ")
                ).transformed(rotate = rotation)

        # Initialize array of points along poloidal profile
        pts = []
        
        # Compute array of points along poloidal profile
        for j in theta:

            # Conditionally offset poloidal profile
            if offset == 0:
                # Compute plasma edge point
                r, p, z = vmec.vmec2rpz(1.0, j, i)
            
            else:
                # Compute offset surface point
                x, y, z = offset_surface(j, i)
                r = np.sqrt(x**2 + y**2)
            
            # Define point on poloidal profile, converting from m to cm
            pt = (r*100, z*100)

            # Append point to poloidal profile
            pts += [pt]

        # Generate poloidal profile
        period = period.spline(pts).close()

    # Loft along poloidal profiles to generate period
    period = period.loft()

    # Construct first period
    period1 = period

    # Construct second period
    period2 = period.rotate((0, 0, 0), (0, 0, 1), 90)

    # Construct third period
    period3 = period.rotate((0, 0, 0), (0, 0, 1), 180)

    # Construct fourth period
    period4 = period.rotate((0, 0, 0), (0, 0, 1), 270)

    # Union periods to create torus
    torus = period1 + period2 + period3 + period4

    return torus


def parametric_stellarator(
    plas_eq, radial_build, exclude = [], step_export = True,
    h5m_export = True, min_mesh_size = 5.0, max_mesh_size = 20.0,
    volume_atol = 0.000001, center_atol = 0.000001,
    bounding_box_atol = 0.000001):
    """Generates CadQuery workplane objects for components of a
    parametrically-defined stellarator, based on user-supplied plasma
    equilibrium VMEC data and a user-defined radial build. Each region is of
    uniform thickness, concentric about the plasma edge. The user may export
    .step files for each reactor component and/or a DAGMC-compatible .h5m file
    for use in neutronics simulations.

    Arguments:
        plas_eq (str): path to plasma equilibrium NetCDF file
        radial_build (dict): dictionary listing region name and region
            thickness in the form {'region': thickness (cm)}. Concentric
            layers will be built in the order given.
        exclude (list of str): names of components not to export.
        step_export (bool): export component .step fiiles
        dagmc_export (bool): export DAGMC compatible .h5m file.
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
    global offset
    offset = 0.0

    # Initialize solid and name tag storage lists
    solids = []
    tags = []

    # Generate plasma STEP file
    plasma = stellarator_torus()
    
    # Optionally store plasma
    if 'plasma' not in exclude:
        solids += [plasma]
        tags += ['plasma']

    # Initialize volume with which layers are cut
    cutter = plasma

    # Generate regions in radial build
    for layer in radial_build:

        # Extract region data
        name = layer
        thickness = radial_build[layer]

        # Compute offset, converting from cm to m
        offset += thickness/100

        # Generate region
        torus_uncut = stellarator_torus()
        torus = torus_uncut - cutter

        # Update cutting volume
        cutter = torus_uncut

        # Store solid and name tag
        if name not in exclude:
            solids += [torus]
            tags += [name]

    if step_export == True:
        for (solid, tag) in zip(solids, tags):
            cq.exporters.export(solid, tag + '.step')

    # Optional DAGMC export
    if h5m_export == True:

        # Determine maximum plasma edge radial position
        R = vmec.vmec2rpz(1.0, 0.0, 0.0)[0]

        # Define length of graveyard and convert from m to cm
        L = 2*(R + offset)*1.25*100

        # Create graveyard volume
        graveyard = cq.Workplane("XY").box(L, L, L).shell(5.0)

        # Append graveyard to storage lists
        solids += [graveyard]
        tags += ['Graveyard']

        export_solids_to_dagmc_h5m(
            solids = solids, tags = tags, filename = 'dagmc.h5m',
            min_mesh_size = min_mesh_size, max_mesh_size = max_mesh_size,
            volume_atol = volume_atol, center_atol = center_atol,
            bounding_box_atol = bounding_box_atol)
