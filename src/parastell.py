import argparse
import magnet_coils
import source_mesh
import log
import read_vmec
import cadquery as cq
import cubit
import cad_to_dagmc
import numpy as np
import math
from scipy.interpolate import RegularGridInterpolator
import os
import inspect
from pathlib import Path
import yaml

m2cm = 100

# Define default export dictionary
export_def = {
    'exclude': [],
    'graveyard': False,
    'step_export': True,
    'h5m_export': None,
    'dir': '',
    'h5m_filename': 'dagmc',
    'native_meshing': False,
    'plas_h5m_tag': None,
    'sol_h5m_tag': None,
    'facet_tol': None,
    'len_tol': None,
    'norm_tol': None,
    'skip_imprinting': False,
    'anisotropic_ratio': 100,
    'deviation_angle': 5,
    'min_mesh_size': 5.0,
    'max_mesh_size': 20.0,
    'volume_atol': 0.00001,
    'center_atol': 0.00001,
    'bounding_box_atol': 0.00001
}


class Stellarator(object):
    '''Parametrically generates a fusion stellarator model using plasma
    equilibrium data and user-defined parameters. In-vessel component
    geometries are determined by a user-defined radial build, in which
    thickness values are supplied in a grid of toroidal and poloidal angles,
    and plasma equilibrium VMEC data. Magnets are defined by a user-defined
    cross-section and coil filament point-locus data.

    Arguments:
        plas_eq (str): path to plasma equilibrium VMEC file.
        build (dict): dictionary of list of toroidal and poloidal angles, as
            well as dictionary of component names with corresponding thickness
            matrix and optional material tag to use in H5M neutronics model.
            The thickness matrix specifies component thickness at specified
            (polidal angle, toroidal angle) pairs. This dictionary takes the
            form
            {
                'phi_list': toroidal angles at which radial build is specified.
                    This list should always begin at 0.0 and it is advised not
                    to extend past one stellarator period. To build a geometry
                    that extends beyond one period, make use of the 'repeat'
                    parameter (list of double, deg).
                'theta_list': poloidal angles at which radial build is
                    specified. This list should always span 360 degrees (list
                    of double, deg).
                'wall_s': closed flux surface label extrapolation at wall
                    (double),
                'radial_build': {
                    'component': {
                        'thickness_matrix': list of list of double (cm),
                        'h5m_tag': h5m_tag (str)
                    }
                }
            }
            If no alternate material tag is supplied for the H5M file, the
            given component name will be used.
        repeat (int): number of times to repeat build segment.
        num_phi (int): number of phi geometric cross-sections to make for each
            build segment (defaults to 61).
        num_theta (int): number of points defining the geometric cross-section
            (defaults to 61).
        scale (double): a scaling factor between the units of VMEC and [cm]
            (defaults to m2cm = 100).
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
                'meshing': setting for tetrahedral mesh generation (bool)
            }
            For the list defining the coil cross-section, the cross-section
            shape must be either a circle or rectangle. For a circular
            cross-section, the list format is
            ['circle' (str), radius (double, cm)]
            For a rectangular cross-section, the list format is
            ['rectangle' (str), width (double, cm), thickness (double, cm)]
        source (dict): dictionary of source mesh parameters including
            {
                'num_s': number of closed magnetic flux surfaces defining mesh
                    (int),
                'num_theta': number of poloidal angles defining mesh (int),
                'num_phi': number of toroidal angles defining mesh (int)
            }
        export (dict): dictionary of export parameters including
            {
                'exclude': names of components not to export (list of str,
                    defaults to empty),
                'graveyard': generate graveyard volume as additional component
                    (bool, defaults to False),
                'step_export': export component STEP files (bool, defaults to
                    True),
                'h5m_export': export DAGMC-compatible neutronics H5M file using
                    Coreform Cubit or CAD-to-DAGMC. Acceptable values are None
                    or a string value of 'cubit' or 'cad_to_damgc'
                    (str, defaults to None). The string is case-sensitive. Note
                    that if magnets are included, 'cubit' must be used,
                'dir': directory to which to export output files (str, defaults
                    to empty string). Note that directory must end in '/', if
                    using Linux or MacOS, or '\' if using Windows.
                'h5m_filename': name of DAGMC-compatible neutronics H5M file
                    (str, defaults to 'dagmc'),
                'plas_h5m_tag': optional alternate material tag to use for
                    plasma. If none is supplied and the plasma is not excluded,
                    'plasma' will be used (str, defaults to None),
                'sol_h5m_tag': optional alternate material tag to use for 
                    scrape-off layer. If none is supplied and the scrape-off
                    layer is not excluded, 'sol' will be used (str, defaults to
                    None),
                'native_meshing': choose native or legacy faceting for DAGMC
                    export (bool, defaults to False),
                'facet_tol': maximum distance a facet may be from surface of
                    CAD representation for Cubit export (double, defaults to
                    None),
                'len_tol': maximum length of facet edge for Cubit export
                    (double, defaults to None),
                'norm_tol': maximum change in angle between normal vector of
                    adjacent facets (double, defaults to None),
                'skip_imprinting': choose whether to imprint and merge all in
                    cubit or to merge surfaces based on import order and
                    geometry information.
                'anisotropic_ratio': controls edge length ratio of elements
                    (double, defaults to 100.0),
                'deviation_angle': controls deviation angle of facet from
                    surface, i.e. lower deviation angle => more elements in
                    areas with higher curvature (double, defaults to 5.0),
                'min_mesh_size': minimum mesh element size for Gmsh export
                    (double, defaults to 5.0),
                'max_mesh_size': maximum mesh element size for Gmsh export
                    (double, defaults to 20.0),
                'volume_atol': absolute volume tolerance to allow when matching
                    parts in intermediate BREP file with CadQuery parts for
                    Gmsh export(double, defaults to 0.00001),
                'center_atol': absolute center coordinates tolerance to allow
                    when matching parts in intermediate BREP file with CadQuery
                    parts for Gmsh export (double, defaults to 0.00001),
                'bounding_box_atol': absolute bounding box tolerance  to allow
                    when matching parts in intermediate BREP file with CadQuery
                    parts for Gmsh export (double, defaults to 0.00001).
            }
        logger (object): logger object (defaults to None). If no logger is
            supplied, a default logger will be instantiated.
    '''

    def __init__(
            self,
            plas_eq,
            build,
            repeat,
            num_phi=61,
            num_theta=61,
            scale=m2cm,
            magnets=None,
            source=None,
            export=export_def,
            logger=None
    ):
        self.plas_eq = plas_eq
        self.build = build
        self.repeat = repeat
        self.num_phi = num_phi
        self.num_theta = num_theta
        self.scale = scale
        self.magnets = magnets
        self.source = source

        self.export_dict = export_def.copy()
        self.export_dict.update(export)

        self.logger = logger
        if self.logger == None or not self.logger.hasHandlers():
            self.logger = log.init()

        self.data = Data(
            self.plas_eq, self.build, self.repeat, self.num_phi,
            self.num_theta, self.scale, self.export_dict, self.logger
        )

        self.cad_geometry = CADGeometry(
            self.data, self.repeat, self.magnets, self.export_dict,
            self.logger
        )

        self.components = self.cad_geometry.components
        self.magnets = self.cad_geometry.magnets

        if self.source is not None:
            self.source_mesh = SourceMesh(self.data.vmec, self.source)

    def export_CAD_geometry(self):
        '''Exports stellarator CAD geometry STEP and/or DAGMC neutronics H5M
        files according to user-specification.
        '''
        if self.export_dict['h5m_export'] not in [None, 'cubit',
                                                  'cad_to_dagmc']:
            raise ValueError(
                'h5m_export must be None or have a string value of \'cubit\' '
                'or \'cad_to_dagmc\''
            )

        if (
            self.export_dict['h5m_export'] == 'cubit' and
            not self.export_dict['step_export']
        ):
            raise ValueError('H5M export via Cubit requires STEP files')

        if (self.export_dict['h5m_export'] == 'cad_to_dagmc' and
                self.magnets is not None):
            raise ValueError(
                'Inclusion of magnets in H5M model requires Cubit export'
            )

        self.dir = Path(self.export_dict['dir'])
        self.h5m_filename = Path(self.export_dict['h5m_filename'])

        if self.export_dict['step_export']:
            self.logger.info('Exporting STEP files...')
            for name, comp in self.components.items():
                export_path = self.dir / Path(name).with_suffix('.step')
                cq.exporters.export(
                    comp['solid'],
                    str(export_path)
                )

        if self.export_dict['h5m_export'] == 'cubit':
            self.logger.info(
                'Exporting DAGMC neutronics H5M file via Coreform Cubit...'
            )
            self.cubit_export()

        if self.export_dict['h5m_export'] == 'cad_to_dagmc':
            self.logger.info(
                'Exporting DAGMC neutronics H5M file via CAD-to-DAGMC...'
            )
            self.gmsh_export()

    def cubit_export(self):
        '''Exports DAGMC neutronics H5M file via Coreform Cubit.
        '''

        def legacy_export():
            """Exports DAGMC neutronics H5M file via legacy plug-in faceting
            method for Coreform Cubit.
            """
            if self.magnets is not None:
                cubit.cmd(
                    f'group "mat:{self.magnets["h5m_tag"]}" add volume '
                    + " ".join(str(i) for i in self.magnets['vol_id'])
                )

            for comp in self.components.values():
                cubit.cmd(
                    f'group "mat:{comp["h5m_tag"]}" add volume '
                    f'{comp["vol_id"]}'
                )

            facet_tol = self.export_dict['facet_tol']
            len_tol = self.export_dict['len_tol']
            norm_tol = self.export_dict['norm_tol']

            facet_tol_str = ''
            len_tol_str = ''
            norm_tol_str = ''

            if facet_tol is not None:
                facet_tol_str = f'faceting_tolerance {facet_tol}'
            if len_tol is not None:
                len_tol_str = f'length_tolerance {len_tol}'
            if norm_tol is not None:
                norm_tol_str = f'normal_tolerance {norm_tol}'

            export_path = self.dir / self.h5m_filename.with_suffix('.h5m')
            cubit.cmd(
                f'export dagmc "{export_path}" {facet_tol_str} {len_tol_str} '
                f'{norm_tol_str} make_watertight'
            )

        def native_export():
            """Exports DAGMC neutronics H5M file via native Coreform Cubit
            faceting method.
            """
            anisotropic_ratio = self.export_dict['anisotropic_ratio']
            deviation_angle = self.export_dict['deviation_angle']

            for comp in self.components.values():
                cubit.cmd(
                    f'create material "{comp["h5m_tag"]}" property_group '
                    + '"CUBIT-ABAQUS"'
                )

            for comp in self.components.values():
                cubit.cmd('set duplicate block elements off')
                cubit.cmd(
                    "block " + str(comp['vol_id']) + " add volume "
                    + str(comp['vol_id'])
                )

            for comp in self.components.values():
                cubit.cmd(
                    "block " + str(comp['vol_id']) + " material "
                    + ''.join(("\'", comp['h5m_tag'], "\'"))
                )

            if self.magnets is not None:
                cubit.cmd(
                    f'create material "{self.magnets["h5m_tag"]}" '
                    + 'property_group "CUBIT-ABAQUS"'
                )

                block_number = min(self.magnets['vol_id'])
                for vol in self.magnets['vol_id']:
                    cubit.cmd('set duplicate block elements off')
                    cubit.cmd(
                        f'block {block_number} add volume {vol}'
                    )

                cubit.cmd(
                    f'block {block_number} material '
                    + ''.join(("\'", self.magnets['h5m_tag'], "\'"))
                )

            cubit.cmd(
                'set trimesher coarse on ratio '
                f'{anisotropic_ratio} angle {deviation_angle}'
            )
            cubit.cmd("surface all scheme trimesh")
            cubit.cmd("mesh surface all")

            export_path = self.dir / self.h5m_filename.with_suffix('.h5m')
            cubit.cmd(f'export cf_dagmc "{export_path}" overwrite')

        def merge_layer_surfaces():
            """Merges model surfaces in Coreform Cubit based on surface IDs
            rather than imprinting and merging all. Assumes that the components
            dictionary is ordered radially outward.
            """

            # Tracks the surface id of the outer surface of the previous layer
            last_outer_surface = None

            for name in self.components.keys():
                # Get volume ID for layer
                vol_id = self.components[name]['vol_id']

                # Get the inner and outer surface IDs of the current layer
                inner_surface, outer_surface = orient_spline_surfaces(vol_id)

                # Wait to merge until the next layer if the plasma is included
                # Store surface to be merged for next loop
                if name == 'plasma':
                    last_outer_surface = outer_surface

                # First layer if plasma is excluded
                elif last_outer_surface is None:
                    last_outer_surface = outer_surface

                # Merge inner surface with outer surface of previous layer
                else:

                    cubit.cmd(
                        f'merge surface {inner_surface} {last_outer_surface}'
                    )

                    last_outer_surface = outer_surface

        def orient_spline_surfaces(volume_id):
            """Return the inner and outer surface IDs for a given volume ID in
            Coreform Cubit.
            """

            surfaces = cubit.get_relatives('volume', volume_id, 'surface')

            spline_surfaces = []

            for surf in surfaces:
                if cubit.get_surface_type(surf) == 'spline surface':
                    spline_surfaces.append(surf)

            if len(spline_surfaces) == 1:
                outer_surface = spline_surfaces[0]
                inner_surface = None
            else:
                # The outer surface bounding box will have the larger max xy
                # value
                if (
                    cubit.get_bounding_box('surface', spline_surfaces[1])[4] >
                    cubit.get_bounding_box('surface', spline_surfaces[0])[4]
                ):
                    outer_surface = spline_surfaces[1]
                    inner_surface = spline_surfaces[0]
                else:
                    outer_surface = spline_surfaces[0]
                    inner_surface = spline_surfaces[1]

            return inner_surface, outer_surface

        for name in self.components.keys():
            import_path = self.dir / Path(name).with_suffix('.step')
            cubit.cmd(f'import step "{import_path}" heal')
            self.components[name]['vol_id'] = cubit.get_last_id("volume")

        if self.export_dict['skip_imprinting']:
            merge_layer_surfaces()

        else:
            cubit.cmd('imprint volume all')
            cubit.cmd('merge volume all')

        if self.export_dict['native_meshing']:
            native_export()
        else:
            legacy_export()

    def gmsh_export(self):
        '''Exports DAGMC neutronics H5M file via CAD-to-DAGMC.
        '''
        model = cad_to_dagmc.CadToDagmc()
        for comp in self.components.values():
            model.add_cadquery_object(
                comp['solid'],
                material_tags=[comp['h5m_tag']]
            )

        model.export_dagmc_h5m_file(
            filename=self.dir / self.h5m_filename.with_suffix('.h5m')
        )


class Data(object):
    '''Generates all stellarator in-vessel component data necessary for CAD
    generation according to user-specification. All user-specified data, the
    actual toroidal and poloidal angles used to build the geometry, and segment
    and total model toroidal angular extent are contained in the Data object.
    In-vessel component outer surface Cartesian point-loci data is contained
    within the Data.radial_build dictionary.

    Arguments:
        plas_eq (str): path to plasma equilibrium VMEC file.
        build (dict): dictionary defining stellarator build. See Stellarator
            class docstring for more detail.
        repeat (int): number of times to repeat build segment.
        num_phi (int): number of phi geometric cross-sections to make for each
            build segment (defaults to 61).
        num_theta (int): number of points defining the geometric cross-section
            (defaults to 61).
        scale (double): a scaling factor between the units of VMEC and [cm].
        export_dict (dict): dictionary defining model export parameters. See
            Stellarator class docstring for more detail.
        logger (object): logger object.
    '''

    def __init__(
            self,
            plas_eq,
            build,
            repeat,
            num_phi,
            num_theta,
            scale,
            export_dict,
            logger
    ):
        self.plas_eq = plas_eq
        self.build = build
        self.repeat = repeat
        self.num_phi = num_phi
        self.num_theta = num_theta
        self.scale = scale
        self.export_dict = export_dict
        self.logger = logger

        self.vmec = read_vmec.vmec_data(self.plas_eq)

        self.phi_list = np.deg2rad(self.build['phi_list'])
        try:
            assert self.phi_list[0] == 0.0, \
                'Initial toroidal angle not equal to 0. Please redefine ' \
                'phi_list, beginning at 0.'
        except AssertionError as e:
            self.logger.error(e.args[0])
            raise e

        self.theta_list = np.deg2rad(self.build['theta_list'])
        try:
            assert self.theta_list[-1] - self.theta_list[0] == 2*np.pi, \
                'Poloidal extent is not 360 degrees. Please ensure poloidal ' \
                'angles are specified for one full revolution.'
        except AssertionError as e:
            self.logger.error(e.args[0])
            raise e

        self.wall_s = self.build['wall_s']
        self.radial_build = self.build['radial_build']

        self.generate_data()

    def generate_data(self):
        '''Generates data defining stellarator in-vessel component geometry.
        '''
        n_phi = len(self.phi_list)
        n_theta = len(self.theta_list)

        # Extract toroidal extent of build
        self.seg_tor_ext = self.phi_list[-1]
        self.tot_tor_ext = (self.repeat + 1)*self.seg_tor_ext

        try:
            assert self.tot_tor_ext <= 2*np.pi, (
                'Total toroidal extent requested with repeated geometry '
                'exceeds 360 degrees. Please examine phi_list and the repeat '
                'parameter.'
            )
        except AssertionError as e:
            self.logger.error(e.args[0])
            raise e

        if self.wall_s != 1.0:
            self.prepend_component_to_radial_build(
                'sol', np.zeros((n_phi, n_theta))
            )

        self.prepend_component_to_radial_build(
            'plasma', np.zeros((n_phi, n_theta))
        )

        self.phi_list_exp = self.expand_ang(self.phi_list, self.num_phi)
        self.theta_list_exp = self.expand_ang(self.theta_list, self.num_theta)

        offset_mat = np.zeros((n_phi, n_theta))

        for name, layer_data in self.radial_build.items():
            self.logger.info(f'Populating {name} data...')

            if name == 'plasma':
                if self.export_dict['plas_h5m_tag'] is not None:
                    layer_data['h5m_tag'] = self.export_dict['plas_h5m_tag']
                s = 1.0
            else:
                s = self.wall_s

            if name == 'sol':
                if self.export_dict['sol_h5m_tag'] is not None:
                    layer_data['h5m_tag'] = self.export_dict['sol_h5m_tag']

            if 'h5m_tag' not in layer_data:
                layer_data['h5m_tag'] = name

            thickness_mat = layer_data['thickness_matrix']

            offset_mat += np.array(thickness_mat)

            offset_mat_exp = self.interpolate_offset_matrix(offset_mat)

            surface_loci = SurfaceLoci(
                self.vmec, s, offset_mat_exp, self.theta_list_exp,
                self.phi_list_exp, self.scale
            )
            layer_data['surface_loci'] = surface_loci.surface_loci

    def interpolate_offset_matrix(self, offset_mat):
        '''Interpolates total offset for expanded angle lists using cubic spline
            interpolation.
        '''
        interpolator = RegularGridInterpolator(
            (self.phi_list, self.theta_list), offset_mat, method='cubic'
        )
        offset_mat_exp = np.zeros(
            (len(self.phi_list_exp), len(self.theta_list_exp)))

        for i, phi in enumerate(self.phi_list_exp):
            offset_mat_exp[i, :] = [interpolator(
                [phi, theta])[0] for theta in self.theta_list_exp]

        return offset_mat_exp

    def prepend_component_to_radial_build(self, comp_name, comp_thickness_mat):
        '''Prepends a component to stellarator radial build.
        '''
        self.radial_build = {
            comp_name: {'thickness_matrix': comp_thickness_mat},
            **self.radial_build
        }

    def expand_ang(self, ang_list, num_ang):
        '''Expands list of angles by linearly interpolating according to
        specified number to include in stellarator build.

        Arguments:
            ang_list (list of double): user-supplied list of toroidal or
                poloidal angles (rad).
            num_ang (int): number of angles to include in stellarator build.

        Returns:
            ang_list_exp (list of double): interpolated list of angles (rad).
        '''
        ang_list_exp = []

        init_ang = ang_list[0]
        final_ang = ang_list[-1]
        ang_extent = final_ang - init_ang

        ang_diff_avg = ang_extent/(num_ang - 1)

        for ang, next_ang in zip(ang_list[:-1], ang_list[1:]):
            n_ang = math.ceil((next_ang - ang)/ang_diff_avg)

            ang_list_exp = np.append(
                ang_list_exp,
                np.linspace(ang, next_ang, num=n_ang + 1)[:-1]
            )

        ang_list_exp = np.append(ang_list_exp, ang_list[-1])

        return ang_list_exp


class SurfaceLoci(object):
    '''Generates stellarator in-vessel component outer surface Cartesian
    point-loci.

    Arguments:
        vmec (object): plasma equilibrium VMEC object from PyStell-UW.
        s (double): the normalized closed flux surface label defining the point
            of reference for offset.
        offset (np.array(double)): for each poloidal and toroidal angle pair,
            an offset from the surface defined by s [cm].
        theta (np.array(double)): the set of poloidal angles specified for
            each rib [rad].
        phi (np.array(double)): the set of toroidal angles defining the plane
            in which each rib is located [rad].
        scale (double): a scaling factor between the units of VMEC and [cm].
    '''

    def __init__(
            self,
            vmec,
            s,
            offset,
            theta,
            phi,
            scale
    ):
        self.vmec = vmec
        self.s = s
        self.offset = offset
        self.theta = theta
        self.phi = phi
        self.scale = scale

        self.surface_loci = self.generate_surface_loci()

    def generate_surface_loci(self):
        '''Generates outer surface point loci.
        '''
        ribs = [
            RibLoci(self.vmec, self.s,
                    self.offset[i, :], self.theta, phi, self.scale)
            for i, phi in enumerate(self.phi)
        ]

        return [rib.r_loci for rib in ribs]


class RibLoci(object):
    '''Generates Cartesian point-loci for stellarator outer surface ribs.

    Arguments:
        vmec (object): plasma equilibrium VMEC object from PyStell-UW.
        s (double): the normalized closed flux surface label defining the point
            of reference for offset.
        offset (np.array(double)): for each poloidal and toroidal angle pair,
            an offset from the surface defined by s [cm].
        theta (np.array(double)): the set of poloidal angles specified for
            each rib [rad].
        phi (np.array(double)): the set of toroidal angles defining the plane
            in which each rib is located [rad].
        scale (double): a scaling factor between the units of VMEC and [cm].
    '''

    def __init__(
            self,
            vmec,
            s,
            offset,
            theta,
            phi,
            scale
    ):
        self.vmec = vmec
        self.s = s
        self.offset = offset
        self.theta = theta
        self.phi = phi
        self.scale = scale

        if not np.all(self.offset >= 0):
            raise ValueError(
                'Offset must be greater than or equal to 0. Check thickness '
                'inputs for negative values'
            )

        self.r_loci = self.generate_loci()

    def generate_loci(self):
        '''Generates Cartesian point-loci for stellarator rib.
        '''
        r_loci = self.vmec2xyz()

        if not np.all(self.offset == 0):
            r_loci += (self.offset.T * self.surf_norm(r_loci).T).T

        return r_loci

    def vmec2xyz(self, poloidal_offset=0):
        '''Return an N x 3 NumPy array containing the Cartesian coordinates of the points at this toroidal angle and N different poloidal angles, each offset slightly.

        Arguments:
            poloidal_offset (double) : some offset to apply to the full set of
                poloidal angles for evaluating the location of the Cartesian
                points (defaults to 0).
        '''
        return self.scale * np.array(
            [self.vmec.vmec2xyz(self.s, theta, self.phi)
             for theta in (self.theta + poloidal_offset)]
        )

    def surf_norm(self, r_loci):
        '''Approximate the normal to the curve at each poloidal angle by first
        approximating the tangent to the curve and then taking the
        cross-product of that tangent with a vector defined as normal to the
        plane at this toroidal angle.

        Arguments:
            r_loci (np.array(double)): Cartesian point-loci of reference
                surface rib [cm].
        '''
        eps = 1e-4
        next_pt_loci = self.vmec2xyz(eps)

        tangent = next_pt_loci - r_loci

        plane_norm = np.array([-np.sin(self.phi), np.cos(self.phi), 0])

        norm = np.cross(plane_norm, tangent)

        return self.normalize(norm)

    def normalize(self, vec_list):
        return np.divide(vec_list.T, np.linalg.norm(vec_list, axis=1).T).T


class CADGeometry(object):
    '''Builds CAD geometry for stellarator in-vessel components using CadQuery
    and calls MagnetSet class when constructing magnet coils. All relevant
    user-defined parameters and segment and total toroidal angular extents are
    contained in the CADGeometry class object. Component parameters, including
    CadQuery solid, H5M material tag, and volume IDs are contained within the
    CADGeometry.components dictionary.

    Arguments:
        data (object): Data class object.
        repeat (int): number of times to repeat build segment.
        magnets (dict): dictionary defining magnets build. See Stellarator
            class docstring for more detail.
        export_dict (dict): dictionary defining model export parameters. See
            Stellarator class docstring for more detail.
        logger (object): logger object.
    '''

    def __init__(
            self,
            data,
            repeat,
            magnets,
            export_dict,
            logger
    ):
        self.data = data
        self.repeat = repeat
        self.magnets = magnets
        self.export_dict = export_dict
        self.logger = logger

        self.seg_tor_ext = self.data.seg_tor_ext
        self.tot_tor_ext = self.data.tot_tor_ext

        self.init_cubit()
        self.components = self.create_geometry()

        if self.magnets is not None:
            self.magnets = MagnetSet(
                self.magnets, self.tot_tor_ext, self.export_dict['dir'],
                self.logger
            )

    def init_cubit(self):
        '''Initializes Coreform Cubit with the DAGMC plugin.
        '''
        if (
            self.export_dict['h5m_export'] == 'cubit' or
            self.magnets is not None
        ):
            cubit_dir = os.path.dirname(inspect.getfile(cubit))
            cubit_dir = Path(cubit_dir) / Path('plugins')
            cubit.init([
                'cubit',
                '-nojournal',
                '-nographics',
                '-information', 'off',
                '-warning', 'off',
                '-commandplugindir',
                str(cubit_dir)
            ])

    def create_geometry(self):
        '''Builds user-specified in-vessel components via InVesselComponent class and constructs components dictionary.
        '''
        radial_build = self.data.radial_build

        components = {}

        # Initialize volume used to cut segments
        cutter = None

        for name, layer_data in radial_build.items():
            self.logger.info(f'Constructing {name} geometry...')

            surface_loci = layer_data['surface_loci']

            components[name] = {}
            components[name]['h5m_tag'] = layer_data['h5m_tag']

            component = InVesselComponent(
                surface_loci, self.seg_tor_ext, self.tot_tor_ext, self.repeat,
                cutter
            )
            components[name]['solid'] = component.component

            cutter = component.cutter

        return components


class InVesselComponent(object):
    '''An object that represents a stellarator in-vessel component CAD
    geometry.

    Arguments:
        surface_loci (np.array(double)): Cartesian coordinates of outer surface
            point cloud.
        seg_tor_ext (double): toroidal angular extent of single build segment
            [rad].
        tot_tor_ext (double): toroidal angular extent of whole build [rad].
        repeat (int): number of times to repeat build segment.
        cutter (object): CadQuery solid used to cut outer surface solid to
            create component layer.
    '''

    def __init__(
            self,
            surface_loci,
            seg_tor_ext,
            tot_tor_ext,
            repeat,
            cutter=None,

    ):
        self.surface_loci = surface_loci
        self.seg_tor_ext = seg_tor_ext
        self.tot_tor_ext = tot_tor_ext
        self.repeat = repeat
        self.cutter = cutter

        self.generate_component()

    def generate_component(self):
        '''Constructs in-vessel component CAD geometry.
        '''
        initial_angles = np.linspace(
            np.rad2deg(self.seg_tor_ext), np.rad2deg(
                self.tot_tor_ext - self.seg_tor_ext),
            num=self.repeat
        )

        surface = OuterSurface(self.surface_loci)

        if self.cutter is not None:
            segment = surface.surface.cut(self.cutter)
        else:
            segment = surface.surface

        self.cutter = surface.surface

        self.component = segment

        for angle in initial_angles:
            rot_segment = segment.rotate((0, 0, 0), (0, 0, 1), angle)
            self.component = self.component.union(rot_segment)


class OuterSurface(object):
    '''An object that represents a surface formed by lofting across a number of
    "ribs" located at different toroidal planes and offset from a reference
    surface.

    Arguments:
        surface_loci (np.array(double)): Cartesian coordinates of outer surface
            point cloud.
    '''

    def __init__(
            self,
            surface_loci
    ):
        self.surface_loci = surface_loci

        self.surface = self.create_surface()

    def create_surface(self):
        '''Constructs component outer surface by lofting across rib splines.
        '''
        rib_objects = [
            RibSpline(r_loci) for r_loci in self.surface_loci
        ]

        ribs = []
        for rib in rib_objects:
            ribs += [rib.rib]

        return cq.Solid.makeLoft(ribs)


class RibSpline(object):
    '''An object that represents a spline curve formed from different poloidal
    points in a single toroidal plane.

    Arguments:
        r_loci (np.array(double)): Cartesian point-loci of component rib spline
            [cm].
    '''

    def __init__(
            self,
            r_loci
    ):
        self.r_loci = r_loci

        self.create_rib()

    def create_rib(self):
        '''Constructs component rib by constructing a spline connecting all
        specified Cartesian point-loci.
        '''
        r_loci = [cq.Vector(tuple(r)) for r in self.r_loci]
        edge = cq.Edge.makeSpline(r_loci).close()
        self.rib = cq.Wire.assembleEdges([edge]).close()


class MagnetSet(object):
    '''Calls magnet_coils Python script to build MagnetSet class object.

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
                'meshing': setting for tetrahedral mesh generation (bool)
            }
            For the list defining the coil cross-section, the cross-section
            shape must be either a circle or rectangle. For a circular
            cross-section, the list format is
            ['circle' (str), radius (double, cm)]
            For a rectangular cross-section, the list format is
            ['rectangle' (str), width (double, cm), thickness (double, cm)]
        tor_ext (double): toroidal extent to model (rad).
        export_dir (str): directory to which to export output files.
        logger (object): logger object (defaults to None). If no logger is
            supplied, a default logger will be instantiated.
    '''

    def __init__(
        self,
        magnets,
        tor_ext,
        export_dir,
        logger
    ):
        self.magnet_geometry = magnet_coils.MagnetSet(
            magnets, tor_ext, export_dir, logger)


class SourceMesh(object):
    '''Calls source_mesh Python script to build SourceMesh class object.

    Arguments:
        vmec (object): plasma equilibrium VMEC object from PyStell-UW.
        source (dict): dictionary of source mesh parameters including
            {
                'num_s': number of closed magnetic flux surfaces defining mesh
                    (int),
                'num_theta': number of poloidal angles defining mesh (int),
                'num_phi': number of toroidal angles defining mesh (int)
            }
    '''

    def __init__(
        self,
        vmec,
        source
    ):
        self.vmec = vmec
        self.source = source

        num_s = self.source['num_s']
        num_theta = self.source['num_theta']
        num_phi = self.source['num_phi']
        tor_ext = self.source['tor_ext']

        self.source_mesh = source_mesh.SourceMesh(
            vmec, num_s, num_theta, num_phi, tor_ext
        )


def parse_args():
    '''Parser for running as a script.
    '''
    parser = argparse.ArgumentParser(prog='sourcemesh')

    parser.add_argument('filename', help='YAML file defining this case')

    return parser.parse_args()


def read_yaml_src(filename):
    '''Read YAML file describing the stellarator build and extract all data.
    '''
    with open(filename) as yaml_file:
        all_data = yaml.safe_load(yaml_file)

    # Extract data to define source mesh
    return (
        all_data['plasma_eq'], all_data['build'], all_data['repeat'],
        all_data['num_phi'], all_data['num_theta'], all_data['magnets'],
        all_data['source'], all_data['export'], all_data['logger']
    )


def parastell():
    '''Main method when run as a command line script.
    '''
    args = parse_args()

    (plasma_eq, build, repeat, num_phi, num_theta, magnets, source, export,
     logger) = read_yaml_src(args.filename)

    stellarator = Stellarator(
        plasma_eq, build, repeat, num_phi, num_theta, magnets, source, export,
        logger
    )


if __name__ == "__main__":
    parastell()
