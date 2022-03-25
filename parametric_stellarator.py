import cadquery as cq
import numpy as np
import read_vmec


def stellarator_torus(vmec, offset = 0.0):
    """Creates a stellarator helical torus as a CadQuery object based on
    VMEC data.

    Arguments:
        vmec (object): read_vmec.vmec_data class object
        offset (float): offset of the torus relative to the plasma edge
            (defaults to 0.0) (cm)

    Returns:
        torus (object): stellarator torus CadQuery object
    """

    # Define the number of phi geometric cross-sections to make
    num_phi = 60
    # Define the number of points defining the geometric cross-section
    num_theta = 20

    # Define toroidal (phi) and poloidal (theta) arrays
    phi = np.linspace(0, np.pi/2, num = num_phi + 1)
    theta = np.linspace(0, 2*np.pi, num = num_theta + 1)

    # Initialize storage list of toroidal periods
    periods = []

    # Generate four toroidal periods
    for _ in range(4):

        # Generate toroidal profiles
        for i in phi:

            # Determine global coordinates of local origin on magnetic axis
            X, Y, Z = vmec.vmec2xyz(0.0, 0.0, i)
            # Multiply by 100 to convert from m to cm
            X, Y, Z = X*100, Y*100, Z*100
            origin = (X, Y, Z)

            # Define rotation vector to orient new workplane at the toroidal
            # angle
            rotation = cq.Vector(0.0, np.rad2deg(i), 0.0)

            # Define new workplane oriented along magnetic axis
            if i == phi[0]:
                period = cq.Workplane("XZ", origin = origin)\
                    .transformed(rotate = rotation)
            else:
                period = period.copyWorkplane(
                    cq.Workplane("XZ", origin = origin)
                    ).transformed(rotate = rotation)

            # Compute radial distance between global and local origins
            R = np.sqrt(X**2 + Y**2)

            # Initialize array of points along toroidal profile
            pts = []
            
            # Compute array of points along toroidal profile
            for j in theta:
                r, p, z = vmec.vmec2rpz(1.0, j, i)
                # Multiply by 100 to convert from m to cm
                r, z = r*100, z*100
                # Transform r, z global coordinates to local coordinates
                r = r - R
                z = z - Z
                # Offset r, z point
                pt = (r + offset*np.cos(j), z + offset*np.sin(j))
                pts += [pt]

            # Generate toroidal profile
            period = period.spline(pts).close()

        # Loft along toroidal profiles to generate period
        period = period.loft()

        # Store toroidal period
        periods += [period]

        # Update the toroidal array to the next toroidal period
        phi = phi + np.pi/2

    # Union all toroidal periods
    torus = periods[0] + periods[1] + periods[2] + periods[3]

    return torus


def parametric_stellarator(
        plas_eq, sol_thickness, fw_thickness, breeder_thickness, bw_thickness,
        shield_thickness, cm_thickness, gap_thickness, vv_thickness):
    """Generates STEP files for components of a parametrically-defined
    stellarator based on user-supplied plasma equilibrium VMEC data
    using CadQuery. Currently, this package generates STEP files for the
    stellarator plasma, scrape-off layer, first wall, breeder, back
    wall, shield, coolant manifold, gap, and vacuum vessel.

    Arguments:
        plas_eq (str): path to plasma equilibrium NetCDF file
        sol_thickness (float): radial thickness of scrape-off layer (cm)
        fw_thickness (float): radial thickness of first wall (cm)
        breeder_thickness (float): radial thickness of breeder (cm)
        bw_thickness (float): radial thickness of back wall (cm)
        shield_thickness (float): radial thickness of shield (cm)
        cm_thickness (float): radial thickness of coolant manifold (cm)
        gap_thickness (float): radial thickness of gap between coolant
            manifold and vacuum vessel (cm)
        vv_thickness (float): radial thickness of vacuum vessel (cm)
    """
    # Load plasma equilibrium data
    vmec = read_vmec.vmec_data(plas_eq)

    # Generate plasma STEP file
    plasma = stellarator_torus(vmec)
    cq.exporters.export(plasma, 'plasma.step')

    # Generate SOL STEP file
    sol_uncut = stellarator_torus(vmec, offset = sol_thickness)
    sol = sol_uncut - plasma
    cq.exporters.export(sol, 'sol.step')

    # Generate first wall STEP file
    fw_offset = sol_thickness + fw_thickness
    fw_uncut = stellarator_torus(vmec, offset = fw_offset)
    first_wall = fw_uncut - sol_uncut
    cq.exporters.export(first_wall, 'first_wall.step')

    # Generate breeder STEP file
    breeder_offset = fw_offset + breeder_thickness
    breeder_uncut = stellarator_torus(vmec, offset = breeder_offset)
    breeder = breeder_uncut - fw_uncut
    cq.exporters.export(breeder, 'breeder.step')

    # Generate back wall STEP file
    bw_offset = breeder_offset + bw_thickness
    bw_uncut = stellarator_torus(vmec, offset = bw_offset)
    back_wall = bw_uncut - breeder_uncut
    cq.exporters.export(back_wall, 'back_wall.step')

    # Generate shield STEP file
    shield_offset = bw_offset + shield_thickness
    shield_uncut = stellarator_torus(vmec, offset = shield_offset)
    shield = shield_uncut - bw_uncut
    cq.exporters.export(shield, 'shield.step')

    # Generate coolant manifold STEP file
    cm_offset = shield_offset + cm_thickness
    cm_uncut = stellarator_torus(vmec, offset = cm_offset)
    coolant_manifold = cm_uncut - shield_uncut
    cq.exporters.export(coolant_manifold, 'coolant_manifold.step')

    # Generate gap STEP file
    gap_offset = cm_offset + gap_thickness
    gap_uncut = stellarator_torus(vmec, offset = gap_offset)
    gap = gap_uncut - cm_uncut
    cq.exporters.export(gap, 'gap.step')

    # Generate vacuum vessel STEP file
    vv_offset = gap_offset + vv_thickness
    vv_uncut = stellarator_torus(vmec, offset = vv_offset)
    vacuum_vessel = vv_uncut - gap_uncut
    cq.exporters.export(vacuum_vessel, 'vacuum_vessel.step')
