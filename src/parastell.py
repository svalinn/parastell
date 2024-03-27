import log
import argparse
import yaml

import cubit
import read_vmec

import src.invessel_build as ivb
import src.magnet_coils as mc
import src.source_mesh as sm
import src.cubit_io as cubit_io
from src.utils import invessel_build_def, magnets_def, source_def,
    dagmc_export_def

class Stellarator(object):
    """Parametrically generates a fusion stellarator reactor core model using
    plasma equilibrium data and user-defined parameters. In-vessel component
    geometries are determined by plasma equilibrium VMEC data and a
    user-defined, three-dimensional radial build, in which thickness values for
    each component are supplied in a grid defined by toroidal and poloidal
    angles. Magnets are defined by coil filament point-locus data and a
    user-defined cross-section. Source meshes are defined on plasma equilibrium
    VMEC data and a structured, uniform grid in magnetic flux space.

    Arguments:
        vmec_file (str): path to plasma equilibrium VMEC file.
        logger (object): logger object (defaults to None). If no logger is
            supplied, a default logger will be instantiated.
    """

    def __init__(
            self,
            vmec_file,
            logger=None
    ):
        self.vmec_file = vmec_file

        self.logger = logger
        if self.logger == None or not self.logger.hasHandlers():
            self.logger = log.init()

        self.vmec = read_vmec.vmec_data(self.vmec_file)

        self.invessel_build = None
        self.magnet_set = None
        self.source_mesh = None

    def construct_invessel_build(self, ivb_dict):
        """Construct InVesselBuild class object.

        Arguments:
            ivb_dict (dict): dictionary of in-vessel component
                parameters, including
                {
                    'toroidal_angles': toroidal angles at which radial build is
                        specified. This list should always begin at 0.0 and it
                        is advised not to extend beyond one stellarator period.
                        To build a geometry that extends beyond one period,
                        make use of the 'repeat' parameter [deg](array of
                        double).
                    'poloidal_angles': poloidal angles at which radial build is
                        specified. This array should always span 360 degrees
                        [deg](array of double).
                    'radial_build': dictionary representing the
                        three-dimensional radial build of in-vessel components,
                        including
                        {
                            'component': {
                                'thickness_matrix': 2-D matrix defining
                                    component thickness at (toroidal angle,
                                    poloidal angle) locations. Rows represent
                                    toroidal angles, columns represent poloidal
                                    angles, and each must be in the same order
                                    provided in toroidal_angles and
                                    poloidal_angles [cm](ndarray(double)).
                                'mat_tag': DAGMC material tag for component in
                                    DAGMC neutronics model (str, defaults to
                                    None). If none is supplied, the 'component'
                                    key will be used.
                            }
                        }.
                    'wall_s': closed flux surface label extrapolation at wall
                        (double).
                    'repeat': number of times to repeat build segment for full
                        model (int, defaults to 0).
                    'num_ribs': total number of ribs over which to loft for each
                        build segment (int, defaults to 61). Ribs are set at
                        toroidal angles interpolated between those specified in
                        'toroidal_angles' if this value is greater than the
                        number of entries in 'toroidal_angles'.
                    'num_rib_pts': total number of points defining each rib
                        spline (int, defaults to 61). Points are set at
                        poloidal angles interpolated between those specified in
                        'poloidal_angles' if this value is greater than the
                        number of entries in 'poloidal_angles'.
                    'scale': a scaling factor between the units of VMEC and [cm]
                        (double, defaults to m2cm = 100).
                    'export_cad_to_dagmc': export DAGMC neutronics H5M file of
                        in-vessel components via CAD-to-DAGMC (bool, defaults
                        to False).
                    'plasma_mat_tag': alternate DAGMC material tag to use for
                        plasma. If none is supplied, 'plasma' will be used
                        (str, defaults to None).
                    'sol_mat_tag': alternate DAGMC material tag to use for
                        scrape-off layer. If none is supplied, 'sol' will be
                        used (str, defaults to None).
                    'dagmc_filename': name of DAGMC output file, excluding
                        '.h5m' extension (str, defaults to 'dagmc').
                    'export_dir': directory to which to export the output files
                        (str, defaults to empty string).
                }
        """

        self.invessel_build = ivb.InVesselBuild(
            self.vmec, ivb_dict['toroidal_angles'],
            ivb_dict['poloidal_angles'], ivb_dict['radial_build'],
            ivb_dict['wall_s'], repeat=ivb_dict['repeat'],
            num_ribs=ivb_dict['num_ribs'], num_rib_pts=ivb_dict['num_rib_pts'],
            scale=ivb_dict['scale'], plasma_mat_tag=ivb_dict['plasma_mat_tag'],
            sol_mat_tag=ivb_dict['sol_mat_tag'], logger=self.logger
        )

        self.invessel_build.populate_surfaces()
        self.invessel_build.calculate_loci()
        self.invessel_build.generate_components()

    def construct_magnets(self, magnets):
        """Constructs MagnetSet class object.

        Arguments:
            magnets (dict): dictionary of magnet parameters, including
                {
                    'coils_file_path': path to coil filament data file (str).
                    'start_line': starting line index for data in file (int).
                    'cross_section': coil cross-section definition; see details
                        below (list).
                    'toroidal_extent': toroidal extent of magnets to model [deg]
                        (double).
                    'sample_mod': sampling modifier for filament points (int,
                        defaults to 1). For a user-supplied value of n, sample
                        every n points in list of points in each filament.
                    'scale': a scaling factor between the units of the filament
                        data and [cm] (double, defaults to m2cm = 100).
                    'step_filename': name of STEP export output file, excluding
                        '.step' extension (str, defaults to 'magnets').
                    'mat_tag': DAGMC material tag for magnets in DAGMC
                        neutronics model (str, defaults to 'magnets').
                    'export_mesh': flag to indicate tetrahedral mesh generation
                        for magnet volumes (bool, defaults to False).
                    'mesh_filename': name of tetrahedral mesh H5M file,
                        excluding '.h5m' extension (str, defaults to
                        'magnet_mesh').
                    'export_dir': directory to which to export output files
                        (str, defaults to empty string).
                }
                For the list defining the coil cross-section, the cross-section
                shape must be either a circle or rectangle. For a circular
                cross-section, the list format is
                ['circle' (str), radius [cm](double)]
                For a rectangular cross-section, the list format is
                ['rectangle' (str), width [cm](double), thickness [cm](double)]
        """

        magnets_dict = magnets_def.copy()
        magnets_dict.update(magnets)

        self.magnet_set = mc.MagnetSet(
            magnets_dict['coils_file_path'], magnets_dict['start_line'],
            magnets_dict['cross_section'], magnets_dict['toroidal_extent'],
            sample_mod=magnets_dict['sample_mod'], scale=magnets_dict['scale'],
            mat_tag=magnets_dict['mat_tag'], logger=self.logger
        )

        self.magnet_set.build_magnet_coils()
        self.magnet_set.export_step(
            filename=magnets_dict['step_filename'],
            export_dir=magnets_dict['export_dir']
        )

        if magnets_dict['export_mesh']:
            self.magnet_set.mesh_magnets()
            self.magnet_set.export_mesh(
                filename=magnets_dict['mesh_filename'],
                export_dir=magnets_dict['export_dir']
            )

    def construct_source_mesh(self, source):
        """Constructs SourceMesh class object.

        Arguments:
            source (dict): dictionary of source mesh parameters including
                {
                    'num_s': number of closed flux surfaces for vertex
                        locations in each toroidal plane (int).
                    'num_theta': number of poloidal angles for vertex locations
                        in each toroidal plane (int).
                    'num_phi': number of toroidal angles for planes of vertices
                        (int).
                    'toroidal_extent': toroidal extent of source to model [deg]
                        (double).
                    'scale': a scaling factor between the units of VMEC and [cm]
                        (double, defaults to m2cm = 100).
                    'filename': name of H5M output file, excluding '.h5m'
                        extension (str, defaults to 'source_mesh').
                    'export_dir': directory to which to export H5M output file
                        (str, defaults to empty string).
                }
        """
        source_dict = source_def.copy()
        source_dict.update(source)

        self.source_mesh = sm.SourceMesh(
            self.vmec, source_dict['num_s'], source_dict['num_theta'],
            source_dict['num_phi'], source_dict['toroidal_extent'],
            scale=source_dict['scale'], logger=self.logger
        )

        self.source_mesh.create_vertices()
        self.source_mesh.create_mesh()
        self.source_mesh.export_mesh(
            filename=source_dict['filename'],
            export_dir=source_dict['export_dir']
        )

    def _import_ivb_step(self):
        """Imports STEP files from in-vessel build into Coreform Cubit.
        (Internal function not intended to be called externally)
        """
        for component, data in (
            self.invessel_build.radial_build.items()
        ):
            vol_id = cubit_io.import_step_cubit(
                component, self.invessel_build.export_dir
            )
            data['vol_id'] = vol_id

    def _construct_components_dict(self):
        """Constructs components dictionary for export routine.
        (Internal function not intended to be called externally)
        """
        self.components = {}

        if self.magnet_set is not None:
            name = self.magnet_set.step_filename
            self.components[name] = {}
            self.components[name]['mat_tag'] = self.magnet_set.mat_tag
            self.components[name]['vol_id'] = list(self.magnet_set.volume_ids)

        if self.invessel_build is not None:
            for component, data in (
                self.invessel_build.radial_build.items()
            ):
                self.components[component] = {}
                self.components[component]['mat_tag'] = data['mat_tag']
                self.components[component]['vol_id'] = data['vol_id']

    def _tag_materials_legacy(self):
        """Applies material tags to corresponding CAD volumes for legacy DAGMC
        neutronics model export.
        (Internal function not intended to be called externally)
        """
        for data in self.components.values():
            if isinstance(data['vol_id'], list):
                vol_id_str = " ".join(str(i) for i in data["vol_id"])
            else:
                vol_id_str = str(data['vol_id'])

            cubit.cmd(
                f'group "mat:{data["mat_tag"]}" add volume {vol_id_str}'
            )

    def _tag_materials_native(self):
        """Applies material tags to corresponding CAD volumes for native DAGMC
        neutronics model export.
        (Internal function not intended to be called externally)
        """
        cubit.cmd('set duplicate block elements off')

        for data in self.components.values():
            if isinstance(data['vol_id'], list):
                block_id = min(data['vol_id'])
                vol_id_str = " ".join(str(i) for i in data["vol_id"])
            else:
                block_id = data['vol_id']
                vol_id_str = str(data['vol_id'])

            cubit.cmd(
                f'create material "{data["mat_tag"]}" property_group '
                '"CUBIT-ABAQUS"'
            )
            cubit.cmd(
                f'block {block_id} add volume {vol_id_str}'
            )
            cubit.cmd(
                f'block {block_id} material \'{data["mat_tag"]}\''
            )

    def export_dagmc(self, dagmc_export=dagmc_export_def):
        """Exports DAGMC neutronics H5M file of ParaStell components via
        Coreform Cubit.

        Arguments:
            dagmc_export (dict): dictionary of DAGMC export parameters including
                {
                    'skip_imprint': choose whether to imprint and merge all in
                        Coreform Cubit or to merge surfaces based on import
                        order and geometry information (bool, defaults to
                        False).
                    'legacy_faceting': choose legacy or native faceting for
                        DAGMC export (bool, defaults to True).
                    'faceting_tolerance': maximum distance a facet may
                        be from surface of CAD representation for DAGMC export
                        (double, defaults to None).
                    'length_tolerance': maximum length of facet edge for DAGMC
                        export (double, defaults to None).
                    'normal_tolerance': maximum change in angle between normal
                        vector of adjacent facets (double, defaults to None).
                    'anisotropic_ratio': controls edge length ratio of elements
                        (double, defaults to 100.0).
                    'deviation_angle': controls deviation angle of facet from
                        surface (i.e., lesser deviation angle results in more
                        elements in areas with greater curvature) (double,
                        defaults to 5.0).
                    'filename': name of DAGMC output file, excluding '.h5m'
                        extension (str, defaults to 'dagmc').
                    'export_dir': directory to which to export DAGMC output file
                        (str, defaults to empty string).
                }
        """
        cubit_io.init_cubit()
        
        self.logger.info(
            'Exporting DAGMC neutronics model...'
        )

        export_dict = dagmc_export_def.copy()
        export_dict.update(dagmc_export)

        if self.invessel_build:
            self._import_ivb_step()

        self._construct_components_dict()

        if export_dict['skip_imprint']:
            ivb.merge_layer_surfaces(self.components)
        else:
            cubit.cmd('imprint volume all')
            cubit.cmd('merge volume all')

        if export_dict['legacy_faceting']:
            self._tag_materials_legacy()
            cubit_io.export_dagmc_cubit_legacy(
                faceting_tolerance=export_dict['faceting_tolerance'],
                length_tolerance=export_dict['length_tolerance'],
                normal_tolerance=export_dict['normal_tolerance'],
                filename=export_dict['filename'],
                export_dir=export_dict['export_dir'],
            )
        else:
            self._tag_materials_native()
            cubit_io.export_dagmc_cubit_native(
                anisotropic_ratio=export_dict['anisotropic_ratio'],
                deviation_angle=export_dict['deviation_angle'],
                filename=export_dict['filename'],
                export_dir=export_dict['export_dir'],
            )


def parse_args():
    """Parser for running as a script.
    """
    parser = argparse.ArgumentParser(prog='stellarator')

    parser.add_argument(
        'filename',
        help='YAML file defining ParaStell stellarator configuration'
    )

    return parser.parse_args()


def read_yaml_config(filename):
    """Read YAML file describing the stellarator configuration and extract all
    data.
    """
    with open(filename) as yaml_file:
        all_data = yaml.safe_load(yaml_file)

    return (
        all_data['vmec_file'], all_data['invessel_build'],
        all_data['magnet_coils'], all_data['source_mesh'],
        all_data['dagmc_export']
    )


def parastell():
    """Main method when run as a command line script.
    """
    args = parse_args()

    (
        vmec_file, invessel_build, magnets, source, dagmc_export
    ) = read_yaml_config(args.filename)

    stellarator = Stellarator(vmec_file)

    ivb_dict = invessel_build_def.copy()
    ivb_dict.update(invessel_build)

    stellarator.construct_invessel_build(ivb_dict)
    stellarator.invessel_build.export_step(export_dir=ivb_dict['export_dir'])

    if ivb_dict['export_cad_to_dagmc']:
        stellarator.invessel_build.export_cad_to_dagmc(
            filename=ivb_dict['dagmc_filename'],
            export_dir=ivb_dict['export_dir']
        )

    stellarator.construct_magnets(magnets)
    stellarator.construct_source_mesh(source)
    stellarator.export_dagmc(dagmc_export)


if __name__ == "__main__":
    parastell()
