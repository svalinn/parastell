from pymoab import core, types
import read_vmec
import numpy as np


def rxn_rate(s):
    """Calculates fusion reaction rate in plasma.

    Arguments:
        s (float): closed magnetic flux surface index in range of 0 (magnetic
            axis) to 1 (plasma edge).

    Returns:
        rr (float): fusion reaction rate (1/cm^3/s). Equates to neutron source
            density.
    """
    # Temperature
    T = 11.5*(1 - s)
    # Ion density
    n = 4.8e20*(1 - s**5)
    # Reaction rate
    rr = 3.68e-18*(n**2)/4*T**(-2/3)*np.exp(-19.94*T**(-1/3))

    return rr


def create_tet(moab_core, mesh_set, verts, id0, id1, id2, id3):
    """Creates tetrahedron and adds to moab core.
    
    Arguments:
        moab_core (object): pymoab core instance.
        mesh_set (EntityHandle): pymoab mesh set.
        verts (list of EntityHandle): list of mesh vertices.
        id0 (int): first tetrahedron index in MOAB canonical numbering system.
        id1 (int): second tetrahedron index in MOAB canonical numbering system.
        id2 (int): third tetrahedron index in MOAB canonical numbering system.
        id3 (int): fourth tetrahedron index in MOAB canonical numbering system.
    """
    # Define vertices for tetrahedron
    tet_verts = [
            verts[id0],
            verts[id1],
            verts[id2],
            verts[id3]
        ]
    # Create tetrahedron in pymoab
    tet = moab_core.create_element(types.MBTET, tet_verts)
    # Add tetrahedron to pymoab core instance
    moab_core.add_entity(mesh_set, tet)


def source_mesh(plas_eq, num_s, num_theta, num_phi):
    """Creates H5M volumetric mesh defining fusion source using pymoab and
    user-supplied plasma equilibrium VMEC data.

    Arguments:
        plas_eq (str): path to plasma equilibrium NetCDF file.
        num_s (int): number of closed magnetic flux surfaces defining mesh.
        num_theta (int): number of poloidal angles defining mesh.
        num_phi (int): number of toroidal angles defining mesh.
    """
    # Load plasma equilibrium data
    vmec = read_vmec.vmec_data(plas_eq)

    # Generate list for closed magnetic flux surface indices in idealized space
    # to be included in mesh
    s_list = np.linspace(0, 1, num = num_s)
    # Generate list for poloidal angles in idealized space to be included in
    # mesh
    theta_list = np.linspace(0, 2*np.pi, num = num_theta)
    # Generate list for toroidal angles in idealized space to be included in
    # mesh
    phi_list = np.linspace(0, 2*np.pi, num = num_phi)

    # Create pymoab core instance
    mbc = core.Core()
    # Create pymoab mesh set for tetrahedra
    tet_set = mbc.create_meshset()

    # Initialize list of vertices in mesh
    verts = []

    # Compute vertices in Cartesian space
    for phi in phi_list:
        # Determine vertex at magnetic axis
        verts += [list(vmec.vmec2xyz(s_list[0], theta_list[0], phi))]
        # Detemine vertices beyond magnetic axis in same toroidal angle
        for s in s_list[1:]:
            for theta in theta_list:
                verts += [list(vmec.vmec2xyz(s, theta, phi))]

    # Create vertices in pymoab
    mbc_verts = mbc.create_vertices(verts)
    # Add vertices to tetrahedron mesh set
    mbc.add_entity(tet_set, mbc_verts)

    # Create tetrahedra, looping through vertices
    for i, phi in enumerate(phi_list[:-1]):
        # Define index for magnetic axis at phi
        ma_id = i*((num_s - 1)*num_theta + 1)
        # Define index for magnetic axis at next phi
        next_ma_id = ma_id + (num_s - 1)*num_theta + 1

        # Create tetrahedra for wedges at center of plasma
        for k, theta in enumerate(theta_list[:-1], 1):
            # Define indices for wedges at center of plasma
            wedge_id0 = ma_id
            wedge_id1 = ma_id + k
            wedge_id2 = ma_id + k + 1
            wedge_id3 = next_ma_id
            wedge_id4 = next_ma_id + k
            wedge_id5 = next_ma_id + k + 1
            
            # Define three tetrahedra for wedge
            create_tet(
                mbc, tet_set, mbc_verts, wedge_id1, wedge_id2, wedge_id4,
                wedge_id0
            )
            create_tet(
                mbc, tet_set, mbc_verts, wedge_id5, wedge_id4, wedge_id2,
                wedge_id3
            )
            create_tet(
                mbc, tet_set, mbc_verts, wedge_id0, wedge_id2, wedge_id4,
                wedge_id3
            )

        # Create tetrahedra for hexahedra beyond center of plasma
        for j, s in enumerate(s_list[1:-1]):
            # Define index at current closed magnetic flux surface
            s_offset = j*num_theta
            # Define index at next closed magnetic flux surface
            next_s_offset = s_offset + num_theta
            
            # Create tetrahedra for current hexahedron
            for k, theta in enumerate(theta_list[:-1], 1):
                # Define indices for hexahedron beyond center of plasma
                tet_id0 = ma_id + s_offset + k
                tet_id1 = ma_id + next_s_offset + k
                tet_id2 = ma_id + next_s_offset + k + 1
                tet_id3 = ma_id + s_offset + k + 1
                tet_id4 = next_ma_id + s_offset + k
                tet_id5 = next_ma_id + next_s_offset + k
                tet_id6 = next_ma_id + next_s_offset + k + 1
                tet_id7 = next_ma_id + s_offset + k + 1
                
                # Define five tetrahedra for hexahedron
                create_tet(
                    mbc, tet_set, mbc_verts, tet_id0, tet_id3, tet_id1, tet_id4)
                create_tet(
                    mbc, tet_set, mbc_verts, tet_id7, tet_id4, tet_id6, tet_id3)
                create_tet(
                    mbc, tet_set, mbc_verts, tet_id2, tet_id1, tet_id3, tet_id6)
                create_tet(
                    mbc, tet_set, mbc_verts, tet_id5, tet_id6, tet_id4, tet_id1)
                create_tet(
                    mbc, tet_set, mbc_verts, tet_id3, tet_id1, tet_id4, tet_id6)

    # Export mesh
    mbc.write_file("SourceMesh.h5m")
