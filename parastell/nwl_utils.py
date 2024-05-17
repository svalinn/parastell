import h5py
import numpy as np
from scipy.optimize import direct
import openmc
import src.pystell.read_vmec as read_vmec
import matplotlib.pyplot as plt


def extract_ss(ss_file):
    """Extracts list of source strengths for each tetrahedron from input file.

    Arguments:
        ss_file (str): path to source strength input file.

    Returns:
        strengths (list): list of source strengths for each tetrahedron (1/s).
            Returned only if source mesh is generated.
    """
    strengths = []
    
    file_obj = open(ss_file, 'r')
    data = file_obj.readlines()
    for line in data:
        strengths.append(float(line))

    return strengths


def nwl_transport(dagmc_geom, source_mesh, tor_ext, ss_file, num_parts):
    """Performs neutron transport on first wall geometry via OpenMC.

    Arguments:
        dagmc_geom (str): path to DAGMC geometry file.
        source_mesh (str): path to source mesh file.
        tor_ext (float): toroidal extent of model (deg).
        ss_file (str): source strength input file.
        num_parts (int): number of source particles to simulate.
    """
    tor_ext = np.deg2rad(tor_ext)
    
    strengths = extract_ss(ss_file)

    # Initialize OpenMC model
    model = openmc.model.Model()

    dag_univ = openmc.DAGMCUniverse(dagmc_geom, auto_geom_ids = False)

    # Define problem boundaries
    vac_surf = openmc.Sphere(
        r = 10000, surface_id = 9999, boundary_type = 'vacuum'
    )
    per_init = openmc.YPlane(
        boundary_type = 'periodic',
        surface_id = 9991
    )
    per_fin = openmc.Plane(
        a = np.sin(tor_ext),
        b = -np.cos(tor_ext),
        c = 0,
        d = 0,
        boundary_type = 'periodic',
        surface_id = 9990
    )

    # Define first period of geometry
    region  = -vac_surf & +per_init & +per_fin
    period = openmc.Cell(cell_id = 9996, region = region, fill = dag_univ)
    geometry = openmc.Geometry([period])
    model.geometry = geometry

    # Define run settings
    settings = openmc.Settings()
    settings.run_mode = 'fixed source'
    settings.particles = num_parts
    settings.batches = 1

    # Define neutron source
    mesh = openmc.UnstructuredMesh(source_mesh, 'moab')
    src = openmc.IndependentSource()
    src.space = openmc.stats.MeshSpatial(
        mesh, strengths = strengths, volume_normalized = False
    )
    src.angle = openmc.stats.Isotropic()
    src.energy = openmc.stats.Discrete([14.1e6], [1.0])
    settings.source = [src]

    # Track surface crossings
    settings.surf_source_write = {
        'surface_ids': [1],
        'max_particles': num_parts
    }

    model.settings = settings

    model.run()


def min_problem(theta, vmec, wall_s, phi, pt):
    """Minimization problem to solve for the poloidal angle.

    Arguments:
        theta (float): poloidal angle (rad).
        vmec (object): plasma equilibrium object.
        wall_s (float): closed flux surface label extrapolation at wall.
        phi (float): toroidal angle (rad).
        pt (array of float): Cartesian coordinates of interest (cm).

    Returns:
        diff (float): L2 norm of difference between coordinates of interest and
            computed point (cm).
    """
    # Compute first wall point
    fw_pt = np.array(vmec.vmec2xyz(wall_s, theta, phi))
    m2cm = 100
    fw_pt = fw_pt*m2cm
    
    diff = np.linalg.norm(pt - fw_pt)

    return diff


def find_coords(vmec, wall_s, phi, pt):
    """Solves for poloidal angle of plasma equilibrium corresponding to
    specified Cartesian coordinates.

    Arguments:
        vmec (object): plasma equilibrium object.
        wall_s (float): closed flux surface label extrapolation at wall.
        phi (float): toroidal angle (rad).
        pt (array of float): Cartesian coordinates of interest (cm).

    Returns:
        theta (float): poloidal angle (rad).
    """
    # Solve for the poloidal angle via minimization
    theta = direct(
        min_problem,
        bounds = [(-np.pi, np.pi)],
        args = (vmec, wall_s, phi, pt)
    )
    # Extract angle
    theta = theta.x[0]

    return theta


def flux_coords(vmec, wall_s, coords):
    """Computes flux-coordinate toroidal and poloidal angles corresponding to
    specified Cartesian coordinates.
    
    Arguments:
        vmec (object): plasma equilibrium object.
        wall_s (float): closed flux surface label extrapolation at wall.
        coords (array of array of float): Cartesian coordinates of all particle
            surface crossings (cm).

    Returns:
        phi_coords (array of float): toroidal angles of surface crossings (rad).
        theta_coords (array of float): poloidal angles of surface crossings
            (rad).
    """
    phi_coords = np.arctan2(coords[:,1], coords[:,0])
    theta_coords = []
    
    for pt, phi in zip(coords, phi_coords):
        theta = find_coords(vmec, wall_s, phi, pt)
        theta_coords.append(theta)

    return phi_coords, theta_coords


def extract_coords(source_file):
    """Extracts Cartesian coordinates of particle surface crossings given an
    OpenMC surface source file.

    Arguments:
        source_file (str): path to OpenMC surface source file.
    
    Returns:
        coords (array of array of float): Cartesian coordinates of all particle
            surface crossings.
    """
    # Load source file
    file = h5py.File(source_file, 'r')
    # Extract source information
    dataset = file['source_bank']['r']
    # Construct matrix of particle crossing coordinates
    coords = np.empty((len(dataset), 3))
    coords[:,0] = dataset['x']
    coords[:,1] = dataset['y']
    coords[:,2] = dataset['z']

    return coords


def plot(nwl_mat, phi_pts, theta_pts, num_levels):
    """Generates contour plot of NWL.

    Arguments:
        nwl_mat (array of array of float): NWL solutions at centroids of
            (phi, theta) bins (MW).
        phi_pts (array of float): centroids of toroidal angle bins (rad).
        theta_bins (array of float): centroids of poloidal angle bins (rad).
        num_levels (int): number of contour regions.
    """
    phi_pts = np.rad2deg(phi_pts)
    theta_pts = np.rad2deg(theta_pts)

    levels = np.linspace(np.min(nwl_mat), np.max(nwl_mat), num = num_levels)
    fig, ax = plt.subplots()
    CF = ax.contourf(phi_pts, theta_pts, nwl_mat, levels = levels)
    cbar = plt.colorbar(CF)
    cbar.ax.set_ylabel('NWL (MW)')
    plt.xlabel('Toroidal Angle (degrees)')
    plt.ylabel('Poloidal Angle (degrees)')
    fig.savefig('NWL.png')


def nwl_plot(
    source_file, ss_file, plas_eq, tor_ext, pol_ext, wall_s, num_phi = 101,
    num_theta = 101, num_levels = 10
    ):
    """Computes and plots NWL.

    Arguments:
        source_file (str): path to OpenMC surface source file.
        ss_file (str): source strength input file.
        plas_eq (str): path to plasma equilibrium NetCDF file.
        tor_ext (float): toroidal extent of model (deg).
        pol_ext (float): poloidal extent of model (deg).
        wall_s (float): closed flux surface label extrapolation at wall.
        num_phi (int): number of toroidal angle bins (defaults to 101).
        num_theta (int): number of poloidal angle bins (defaults to 101).
        num_levels (int): number of contour regions (defaults to 10).
    """
    tor_ext = np.deg2rad(tor_ext)
    pol_ext = np.deg2rad(pol_ext)
    
    coords = extract_coords(source_file)
    
    # Load plasma equilibrium data
    vmec = read_vmec.VMECData(plas_eq)

    phi_coords, theta_coords = flux_coords(vmec, wall_s, coords)

    # Define minimum and maximum bin edges for each dimension
    phi_min = 0 - tor_ext/num_phi/2
    phi_max = tor_ext + tor_ext/num_phi/2
    theta_min = -pol_ext/2 - pol_ext/num_theta/2
    theta_max = pol_ext/2 + pol_ext/num_theta/2
    
    # Bin particle crossings
    count_mat, phi_bins, theta_bins = np.histogram2d(
        phi_coords,
        theta_coords,
        bins = [num_phi, num_theta],
        range = [[phi_min, phi_max], [theta_min, theta_max]]
    )

    # Compute centroids of bin dimensions
    phi_pts = np.linspace(0, tor_ext, num = num_phi)
    theta_pts = np.linspace(-pol_ext/2, pol_ext/2, num = num_theta)

    # Define fusion neutron energy (eV)
    n_energy = 14.1e6
    # Define eV to joules constant
    eV2J = 1.60218e-19
    # Compute total neutron source strength (n/s)
    strengths = extract_ss(ss_file)
    SS = sum(strengths)
    # Define joules to megajoules constant
    J2MJ = 1e-6
    # Define number of source particles
    num_parts = len(coords)

    nwl_mat = count_mat*n_energy*eV2J*SS*J2MJ/num_parts

    plot(nwl_mat, phi_pts, theta_pts, num_levels)
