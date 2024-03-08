import cubit
import src.invessel_build as ivb
import magnet_coils
import source_mesh
import log
import cadquery as cq
import cad_to_dagmc
import read_vmec
import argparse
import yaml
from pathlib import Path
import os
import inspect

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
    'plasma_h5m_tag': None,
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
        vmec_file (str): path to plasma equilibrium VMEC file.
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
            vmec_file,
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
        self.vmec_file = vmec_file
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

        self.vmec = read_vmec.vmec_data(self.vmec_file)

        self.check_input()

        if (
            self.export_dict['h5m_export'] == 'cubit' or
            self.magnets is not None
        ):
            self.init_cubit()

    def check_input(self):
        '''Checks user input for errors.
        '''
        if (
            self.export_dict['h5m_export'] not in
            [None, 'cubit', 'cad_to_dagmc']
        ):
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
    
    def init_cubit(self):
        '''Initializes Coreform Cubit with the DAGMC plugin.
        '''
        cubit_plugin_dir = (
            Path(os.path.dirname(inspect.getfile(cubit))) / Path('plugins')
        )
        cubit.init([
            'cubit',
            '-nojournal',
            '-nographics',
            '-information', 'off',
            '-warning', 'off',
            '-commandplugindir',
            str(cubit_plugin_dir)
        ])

    def construct_invessel_build(self):
        '''Construct InVesselBuild class object.
        '''
        self.invessel_build = ivb.InVesselBuild(
            self.vmec, self.build, self.repeat, self.num_phi, self.num_theta,
            self.scale, self.export_dict['plasma_h5m_tag'],
            self.export_dict['sol_h5m_tag'], self.logger
        )
        self.invessel_build.populate_surfaces()
        self.invessel_build.calculate_loci()
        self.invessel_build.generate_components()

    def construct_source_mesh(self):
        '''Constructs SourceMesh class object.
        '''
        self.source_mesh = source_mesh.SourceMesh(self.vmec, self.source)

    def construct_magnets(self):
        '''Constructs MagnetSet class object.
        '''
        self.magnet_set = magnet_coils.MagnetSet(
            self.magnets, self.ivc_data.tot_tor_ext, self.export_dict['dir'],
            self.logger)

    def export_CAD_geometry(self):
        '''Exports stellarator CAD geometry STEP and/or DAGMC neutronics H5M
        files according to user-specification.
        '''
        self.construct_components_dict()

        if self.export_dict['step_export']:
            self.logger.info('Exporting STEP files...')
            for name, component in self.components.items():
                export_path = (
                    Path(self.export_dict['dir']) /
                    Path(name).with_suffix('.step')
                )
                cq.exporters.export(
                    component['solid'],
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

    def construct_components_dict(self):
        '''Constructs components dictionary for export routine.
        '''
        self.components = {}

        for component, (name, layer_data) in zip(
            self.invessel_build.Components,
            self.build['radial_build'].items()
        ):
            self.components[name] = {}
            self.components[name]['h5m_tag'] = layer_data['h5m_tag']
            self.components[name]['solid'] = component

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

            export_path = (
                Path(self.export_dict['dir']) /
                Path(self.export_dict['h5m_filename']).with_suffix('.h5m')
            )
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

            export_path = (
                Path(self.export_dict['dir']) /
                Path(self.export_dict['h5m_filename']).with_suffix('.h5m')
            )
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
            import_path = (
                Path(self.export_dict['dir']) / Path(name).with_suffix('.step')
            )
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
        export_path = (
            Path(self.export_dict['dir']) /
            Path(self.export_dict['h5m_filename']).with_suffix('.h5m')
        )
        model.export_dagmc_h5m_file(
            filename=export_path
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
        all_data['vmec_file'], all_data['build'], all_data['repeat'],
        all_data['num_phi'], all_data['num_theta'], all_data['magnets'],
        all_data['source'], all_data['export'], all_data['logger']
    )


def parastell():
    '''Main method when run as a command line script.
    '''
    args = parse_args()

    (vmec_file, build, repeat, num_phi, num_theta, magnets, source, export,
     logger) = read_yaml_src(args.filename)

    stellarator = Stellarator(
        vmec_file, build, repeat, num_phi, num_theta, magnets, source, export,
        logger
    )


if __name__ == "__main__":
    parastell()
