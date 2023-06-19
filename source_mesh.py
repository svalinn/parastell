import log
from pymoab import core, types
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
    if s == 1:
        rr = 0
    else:
        # Temperature
        T = 11.5*(1 - s)
        # Ion density
        n = 4.8e20*(1 - s**5)
        # Reaction rate
        rr = 3.68e-18*(n**2)/4*T**(-2/3)*np.exp(-19.94*T**(-1/3))

    return rr


def source_strength(verts, verts_s, id0, id1, id2, id3):
    """Computes neutron source strength for a tetrahedron using five-node
    Gaussian quadrature.

    Arguments:
        verts (list of list of float): list of 3D Cartesian coordinates of each
            vertex in form [x (cm), y (cm), z (cm)].
        verts_s (list of float): list of closed flux surface indices for each
            vertex.
        id0 (int): first tetrahedron index in MOAB canonical numbering system.
        id1 (int): second tetrahedron index in MOAB canonical numbering system.
        id2 (int): third tetrahedron index in MOAB canonical numbering system.
        id3 (int): fourth tetrahedron index in MOAB canonical numbering system.

    Returns:
        ss (float): integrated source strength for tetrahedron.
    """
    # Define vertices for tetrahedron
    verts0 = verts[id0]
    verts1 = verts[id1]
    verts2 = verts[id2]
    verts3 = verts[id3]

    # Compute fusion source density at each vertex
    ss0 = rxn_rate(verts_s[id0])
    ss1 = rxn_rate(verts_s[id1])
    ss2 = rxn_rate(verts_s[id2])
    ss3 = rxn_rate(verts_s[id3])

    # Define barycentric coordinates for integration points
    bary_coords = [
        [0.25, 0.25, 0.25, 0.25],
        [0.5, 1/6, 1/6, 1/6],
        [1/6, 0.5, 1/6, 1/6],
        [1/6, 1/6, 0.5, 1/6],
        [1/6, 1/6, 1/6, 0.5]
    ]

    # Define weights for integration points
    int_w = [-0.8, 0.45, 0.45, 0.45, 0.45, 0.45]
    
    # Interpolate source strength at integration points
    ss_int_pts = []
    for pt in bary_coords:
        ss_int = pt[0]*ss0 + pt[1]*ss1 + pt[2]*ss2 + pt[3]*ss3
        ss_int_pts.append(ss_int)
    
    # Compute graph of tetrahedral vertices
    T = [
        [
            verts0[0] - verts3[0],
            verts1[0] - verts3[0],
            verts2[0] - verts3[0]
        ],
        [
            verts0[1] - verts3[1],
            verts1[1] - verts3[1],
            verts2[1] - verts3[1]
        ],
        [
            verts0[2] - verts3[2],
            verts1[2] - verts3[2],
            verts2[2] - verts3[2]
        ]
    ]
    
    # Compute Jacobian of graph
    Jac = np.linalg.det(T)
    # Compute volume of tetrahedron
    vol = np.abs(Jac)/6
    # Compute source strength of tetrahedron
    ss = vol*sum(i*j for i, j in zip(int_w, ss_int_pts))

    return ss


def create_tet(
    mbc, tag_handle, mesh_set, mbc_verts, verts, verts_s, id0, id1, id2,
    id3, strengths):
    """Creates tetrahedron and adds to moab core.

    Arguments:
        mbc (object): PyMOAB core instance.
        tag_handle (TagHandle): PyMOAB source strength tag.
        mesh_set (EntityHandle): PyMOAB mesh set.
        mbc_verts (list of EntityHandle): list of mesh vertices.
        verts (list of list of float): list of 3D Cartesian coordinates of each
            vertex in form [x (cm), y (cm), z (cm)].
        verts_s (list of float): list of closed flux surface indices for each
            vertex.
        id0 (int): first tetrahedron index in MOAB canonical numbering system.
        id1 (int): second tetrahedron index in MOAB canonical numbering system.
        id2 (int): third tetrahedron index in MOAB canonical numbering system.
        id3 (int): fourth tetrahedron index in MOAB canonical numbering system.
    """
    # Define vertices for tetrahedron
    tet_verts = [
            mbc_verts[id0],
            mbc_verts[id1],
            mbc_verts[id2],
            mbc_verts[id3]
        ]
    # Create tetrahedron in PyMOAB
    tet = mbc.create_element(types.MBTET, tet_verts)
    # Add tetrahedron to PyMOAB core instance
    mbc.add_entity(mesh_set, tet)
    # Compute source strength for tetrahedron
    ss = source_strength(verts, verts_s, id0, id1, id2, id3)
    # Tag tetrahedra with source strength data
    mbc.tag_set_data(tag_handle, tet, [ss])
    # Append source strength to list
    strengths.append(ss)


def create_mesh(
    mbc, tag_handle, num_s, num_theta, s_list, theta_list, phi_list, mbc_verts,
    verts, verts_s):
    """Creates volumetric source mesh in real space.

    Arguments:
        mbc (object): PyMOAB core instance.
        s_list (list of float): list of closed flux surfaces defining mesh in
            confinement space.
        theta_list (list of float): list of poloidal angles defining mesh in
            confinement space (rad).
        phi_list (list of float): list of toroidal angles defining mesh in
            confinement space (rad).
        mbc_verts (list of EntityHandle): list of mesh vertices.
        verts (list of list of float): list of 3D Cartesian coordinates of each
            vertex in form [x (cm), y (cm), z (cm)].
        verts_s (list of float): list of closed flux surface indices for each
            vertex.
            
    Returns:
        strengths (list of float): list of source strengths for each
            tetrahedron (1/s).
    """
    # Instantiate tetrahedra mesh set in PyMOAB
    tet_set = mbc.create_meshset()
    # Add vertices to mesh set
    mbc.add_entity(tet_set, mbc_verts)

    # Initialize list of tetrahedra source strengths
    strengths = []
    
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
                mbc, tag_handle, tet_set, mbc_verts, verts, verts_s, wedge_id1,
                wedge_id2, wedge_id4, wedge_id0, strengths
            )
            create_tet(
                mbc, tag_handle, tet_set, mbc_verts, verts, verts_s, wedge_id5,
                wedge_id4, wedge_id2, wedge_id3, strengths
            )
            create_tet(
                mbc, tag_handle, tet_set, mbc_verts, verts, verts_s, wedge_id0,
                wedge_id2, wedge_id4, wedge_id3, strengths
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
                    mbc, tag_handle, tet_set, mbc_verts, verts, verts_s,
                    tet_id0, tet_id3, tet_id1, tet_id4, strengths
                )
                create_tet(
                    mbc, tag_handle, tet_set, mbc_verts, verts, verts_s,
                    tet_id7, tet_id4, tet_id6, tet_id3, strengths
                )
                create_tet(
                    mbc, tag_handle, tet_set, mbc_verts, verts, verts_s,
                    tet_id2, tet_id1, tet_id3, tet_id6, strengths
                )
                create_tet(
                    mbc, tag_handle, tet_set, mbc_verts, verts, verts_s,
                    tet_id5, tet_id6, tet_id4, tet_id1, strengths
                )
                create_tet(
                    mbc, tag_handle, tet_set, mbc_verts, verts, verts_s,
                    tet_id3, tet_id1, tet_id4, tet_id6, strengths
                )

    return strengths


def create_vertices(vmec, mbc, s_list, theta_list, phi_list):
    """Creates mesh vertices and adds them to PyMOAB core.

    Arguments:
        vmec (object): plasma equilibrium VMEC object.
        mbc (object): PyMOAB core instance.
        s_list (list of float): list of closed flux surfaces defining mesh in
            confinement space.
        theta_list (list of float): list of poloidal angles defining mesh in
            confinement space (rad).
        phi_list (list of float): list of toroidal angles defining mesh in
            confinement space (rad).

    Returns:
        mbc_verts (list of EntityHandle): list of mesh vertices.
        verts (list of list of float): list of 3D Cartesian coordinates of each
            vertex in form [x (cm), y (cm), z (cm)].
        verts_s (list of float): list of closed flux surface indices for each
            vertex.
    """
    # Initialize list of vertices in mesh
    verts = []
    # Initialize list of closed flux surface indices for each vertex
    verts_s = []

    # Compute vertices in Cartesian space
    for phi in phi_list:
        # Determine vertex at magnetic axis, converting to cm
        vertex = np.array(vmec.vmec2xyz(s_list[0], theta_list[0], phi))*100
        # Append vertex to list
        verts += [list(vertex)]
        # Store s for vertex
        verts_s += [s_list[0]]
        for s in s_list[1:]:
            for theta in theta_list:
                # Detemine vertices beyond magnetic axis in same toroidal angle
                vertex = np.array(vmec.vmec2xyz(s, theta, phi))*100
                verts += [list(vertex)]
                # Store s for vertex
                verts_s += [s]

    # Create vertices in PyMOAB
    mbc_verts = mbc.create_vertices(verts)

    return mbc_verts, verts, verts_s


def create_mbc():
    """Creates PyMOAB core instance with source strength tag.

    Returns:
        mbc (object): PyMOAB core instance.
        tag_handle (TagHandle): PyMOAB source strength tag.
    """
    # Create PyMOAB core instance
    mbc = core.Core()

    # Define data type for source strength tag
    tag_type = types.MB_TYPE_DOUBLE
    # Define tag size for source strength tag (1 double value)
    tag_size = 1
    # Define storage type for source strength
    storage_type = types.MB_TAG_DENSE
    # Define tag handle for source strength
    tag_handle = mbc.tag_get_handle(
        "SourceStrength", tag_size, tag_type, storage_type,
        create_if_missing = True
    )

    return mbc, tag_handle


def source_mesh(vmec, source, logger = None):
    """Creates H5M volumetric mesh defining fusion source using PyMOAB and
    user-supplied plasma equilibrium VMEC data.

    Arguments:
        vmec (object): plasma equilibrium VMEC object.
        source (dict): dictionary of source mesh parameters including
            {
                'num_s': number of closed magnetic flux surfaces defining mesh
                    (int),
                'num_theta': number of poloidal angles defining mesh (int),
                'num_phi': number of toroidal angles defining mesh (int)
            }
        logger (object): logger object (defaults to None). If no logger is
            supplied, a default logger will be instantiated.

    Returns:
        strengths (list of float): list of source strengths for each
            tetrahedron (1/s).
    """
    # Conditionally instantiate logger
    if logger == None or not logger.hasHandlers():
        logger = log.init()

    # Signal source mesh generation
    logger.info(f'Building SourceMesh.h5m...')

    # Extract source mesh parameters
    num_s = source['num_s']
    num_theta = source['num_theta']
    num_phi = source['num_phi']

    # Generate list for closed magnetic flux surface indices in confinement
    # space to be included in mesh
    s_list = np.linspace(0, 1, num = num_s)
    # Generate list for poloidal angles in confinement space to be included in
    # mesh
    theta_list = np.linspace(0, 2*np.pi, num = num_theta)
    # Generate list for toroidal angles in confinement space to be included in
    # mesh
    phi_list = np.linspace(0, 2*np.pi, num = num_phi)

    # Instantiate PyMOAB core instance and define source strength tag
    mbc, tag_handle = create_mbc()

    # Define mesh vertices in real and confinement space and add to PyMOAB core
    mbc_verts, verts, verts_s = create_vertices(
        vmec, mbc, s_list, theta_list, phi_list
    )

    # Create source mesh
    strengths = create_mesh(
        mbc, tag_handle, num_s, num_theta, s_list, theta_list, phi_list,
        mbc_verts, verts, verts_s
    )

    # Export mesh
    mbc.write_file("SourceMesh.h5m")

    return strengths
