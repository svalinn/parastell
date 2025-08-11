import argparse
from pathlib import Path

import cadquery as cq
import cad_to_dagmc
from pymoab import core

from . import log
from . import invessel_build as ivb
from . import magnet_coils as mc
from . import source_mesh as sm
from .cubit_utils import (
    create_new_cubit_instance,
    export_dagmc_cubit,
    export_cub5,
    make_material_block,
)
from .utils import read_yaml_config, filter_kwargs, m2cm, combine_dagmc_models
from .pystell import read_vmec

build_cubit_model_allowed_kwargs = ["skip_imprint"]
export_cubit_dagmc_allowed_kwargs = [
    "filename",
    "export_dir",
    "anisotropic_ratio",
    "deviation_angle",
]
build_cad_to_dagmc_model_allowed_kwargs = []
export_cad_to_dagmc_allowed_kwargs = [
    "filename",
    "export_dir",
    "min_mesh_size",
    "max_mesh_size",
]


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
        vmec_file (str): path to plasma equilibrium VMEC file. Used to define
            the source mesh, and if no other reference surface is provided,
            defines the innermost surface from which in vessel components are
            built.
        ref_surf (ReferenceSurface): ReferenceSurface object. Must have a
            method 'angles_to_xyz(toroidal_angles, poloidal_angles, s)' that
            returns an Nx3 numpy array of cartesian coordinates for any closed
            flux surface label, s, poloidal angle (theta), and toroidal angle
            (phi). Optional. If None, the vmec data will be used as the
            reference surface.
        logger (object): logger object (optional, defaults to None). If no
            logger is supplied, a default logger will be instantiated.
    """

    def __init__(self, vmec_file, ref_surf=None, logger=None):

        self.logger = logger
        self.ref_surf = ref_surf
        self.vmec_file = vmec_file
        self.invessel_build = None
        self.magnet_set = None
        self.source_mesh = None
        self.use_pydagmc = False

    @property
    def ref_surf(self):
        return self._ref_surf

    @ref_surf.setter
    def ref_surf(self, ref_surf_obj):
        self._ref_surf = ref_surf_obj

    @property
    def vmec_file(self):
        return self._vmec_file

    @vmec_file.setter
    def vmec_file(self, file):
        self._vmec_file = file
        try:
            self._vmec_obj = read_vmec.VMECData(self._vmec_file)
            if self._ref_surf is None:
                self.ref_surf = ivb.VMECSurface(self._vmec_obj)

        except Exception as e:
            self._logger.error(e.args[0])
            raise e

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, logger_object):
        self._logger = log.check_init(logger_object)

    def construct_invessel_build(
        self,
        toroidal_angles,
        poloidal_angles,
        wall_s,
        radial_build,
        split_chamber=False,
        **kwargs,
    ):
        """Construct InVesselBuild class object.

        Arguments:
            toroidal_angles (array of float): toroidal angles at which radial
                build is specified. This list should always begin at 0.0 and it
                is advised not to extend beyond one stellarator period. To
                build a geometry that extends beyond one period, make use of
                the 'repeat' parameter [deg].
            poloidal_angles (array of float): poloidal angles at which radial
                build is specified. This array should always span 360 degrees
                [deg].
            wall_s (float): closed flux surface label extrapolation at wall.
            radial_build (dict): dictionary representing the three-dimensional
                radial build of in-vessel components, including
                {
                    'component': {
                        'thickness_matrix': 2-D matrix defining component
                            thickness at (toroidal angle, poloidal angle)
                            locations. Rows represent toroidal angles, columns
                            represent poloidal angles, and each must be in the
                            same order provided in toroidal_angles and
                            poloidal_angles [cm](ndarray(float)).
                        'mat_tag': DAGMC material tag for component in DAGMC
                            neutronics model (str, optional, defaults to None).
                            If none is supplied, the 'component' key will be
                            used.
                    }
                }.
            split_chamber (bool): if wall_s > 1.0, separate interior vacuum
                chamber into plasma and scrape-off layer components (optional,
                defaults to False). If an item with a 'sol' key is present in
                the radial_build dictionary, settting this to False will not
                combine the resultant 'chamber' with 'sol'. To include a custom
                scrape-off layer definition for 'chamber', add an item with a
                'chamber' key and desired 'thickness_matrix' value to the
                radial_build dictionary.

        Optional attributes:
            plasma_mat_tag (str): alternate DAGMC material tag to use for
                plasma. If none is supplied, 'Vacuum' will be used (defaults to
                None).
            sol_mat_tag (str): alternate DAGMC material tag to use for
                scrape-off layer. If none is supplied, 'Vacuum' will be used
                (defaults to None).
            repeat (int): number of times to repeat build segment for full model
                (defaults to 0).
            num_ribs (int): total number of ribs over which to loft for each
                build segment (defaults to 61). Ribs are set at toroidal angles
                interpolated between those specified in 'toroidal_angles' if
                this value is greater than the number of entries in
                'toroidal_angles'.
            num_rib_pts (int): total number of points defining each rib spline
                (defaults to 61). Points are set at poloidal angles interpolated
                between those specified in 'poloidal_angles' if this value is
                greater than the number of entries in 'poloidal_angles'.
            scale (float): a scaling factor between input and output data
                (defaults to m2cm = 100).
            use_pydagmc (bool): if True, generate dagmc model directly with
                pydagmc, bypassing CAD generation. Results in faceted geometry,
                rather than smooth spline surfaces. Defaults to False.
        """
        self.radial_build = ivb.RadialBuild(
            toroidal_angles,
            poloidal_angles,
            wall_s,
            radial_build,
            split_chamber=split_chamber,
            logger=self._logger,
            **kwargs,
        )
        self.invessel_build = ivb.InVesselBuild(
            self._ref_surf, self.radial_build, logger=self._logger, **kwargs
        )
        self.use_pydagmc = self.invessel_build.use_pydagmc
        self.invessel_build.populate_surfaces()
        self.invessel_build.calculate_loci()
        self.invessel_build.generate_components()

    def export_invessel_build_step(self, export_dir=""):
        """Exports InVesselBuild component STEP files.

        Arguments:
            export_dir (str): directory to which to export the output files
                (optional, defaults to empty string).
        """
        self.invessel_build.export_step(export_dir=export_dir)

    def export_invessel_build_mesh_moab(
        self, components, filename, export_dir=""
    ):
        """Creates a tetrahedral mesh of in-vessel component volumes via MOAB
        and exports the mesh as a H5M file. Note that this mesh is created
        using the point cloud of the specified components and as such, each
        component's mesh will be one tetrahedron thick.

        Arguments:
            components (array of str): array containing the names of the
                in-vessel components to be meshed.
            filename (str): name of H5M output file.
            export_dir (str): directory to which to export the h5m output file
                (defaults to empty string).
        """
        self.invessel_build.mesh_components_moab(components)
        self.invessel_build.export_mesh_moab(filename, export_dir=export_dir)

    def export_invessel_build_mesh_gmsh(
        self,
        components,
        filename,
        min_mesh_size=5.0,
        max_mesh_size=20.0,
        algorithm=1,
        export_dir="",
    ):
        """Creates a tetrahedral mesh of in-vessel component volumes via
        Gmsh and exports the mesh as a H5M file.

        Arguments:
            components (array of str): array containing the names of the
                in-vessel components to be meshed.
            filename (str): name of H5M output file.
            min_mesh_size (float): minimum size of mesh elements (defaults to
                5.0).
            max_mesh_size (float): maximum size of mesh elements (defaults to
                20.0).
            algorithm (int): integer identifying the meshing algorithm to use
                for the surface boundary (defaults to 1). Options are as
                follows, refer to Gmsh documentation for explanations of each.
                1: MeshAdapt, 2: automatic, 3: initial mesh only, 4: N/A,
                5: Delaunay, 6: Frontal-Delaunay, 7: BAMG, 8: Frontal-Delaunay
                for Quads, 9: Packing of Parallelograms, 11: Quasi-structured
                Quad.
            export_dir (str): directory to which to export the h5m
                output file (optional, defaults to empty string).
        """
        self.invessel_build.mesh_components_gmsh(
            components,
            min_mesh_size=min_mesh_size,
            max_mesh_size=max_mesh_size,
            algorithm=algorithm,
        )
        self.invessel_build.export_mesh_gmsh(filename, export_dir=export_dir)

    def export_invessel_build_mesh_cubit(
        self,
        components,
        filename,
        mesh_size=5,
        anisotropic_ratio=100.0,
        deviation_angle=5.0,
        export_dir="",
    ):
        """Creates a tetrahedral mesh of in-vessel component volumes via
        Coreform Cubit and exports the mesh as a H5M file.

        Arguments:
            components (array of str): array containing the names of the
                in-vessel components to be meshed.
            filename (str): name of H5M output file.
            mesh_size (int): controls the size of the mesh. Takes values
                between 1 (finer) and 10 (coarser) (optional, defaults to 5).
            anisotropic_ratio (float): controls edge length ratio of elements
                (defaults to 100.0).
            deviation_angle (float): controls deviation angle of facet from
                surface (i.e., lesser deviation angle results in more elements
                in areas with higher curvature) (defaults to 5.0).
            export_dir (str): directory to which to export the h5m
                output file (optional, defaults to empty string).
        """
        self.invessel_build.mesh_components_cubit(
            components,
            mesh_size=mesh_size,
            anisotropic_ratio=anisotropic_ratio,
            deviation_angle=deviation_angle,
            import_dir=self.invessel_build.export_dir,
        )
        self.invessel_build.export_mesh_cubit(filename, export_dir=export_dir)

    def construct_magnets_from_filaments(
        self,
        coils_file,
        width,
        thickness,
        toroidal_extent,
        case_thickness=0.0,
        **kwargs,
    ):
        """Constructs MagnetSetFromFilaments class object.

        Arguments:
            coils_file (str): path to coil filament data file.
            width (float): width of coil cross-section in toroidal direction
                [cm].
            thickness (float): thickness of coil cross-section in radial
                direction [cm].
            toroidal_extent (float): toroidal extent to model [deg].
            case_thickness (float): thickness of outer coil casing (defaults to
                0.0) [cm]. Double this amount will be subtracted from the width
                and thickness parameters to form the inner coil volume.

        Optional attributes:
            start_line (int): starting line index for data in filament data file
                (defaults to 3).
            sample_mod (int): sampling modifier for filament points (defaults to
                1). For a user-defined value n, every nth point will be sampled.
            scale (float): a scaling factor between input and output data
                (defaults to m2cm = 100).
            mat_tag (str or iterable of str): DAGMC material tag(s) to use for
                magnets in DAGMC neutronics model (defaults to 'magnets'). If
                an iterable is given, the first entry will be applied to coil
                casing and the second to the inner volume. If just one is
                given, it will be applied to all magnet volumes.
        """
        self.magnet_set = mc.MagnetSetFromFilaments(
            coils_file,
            width,
            thickness,
            toroidal_extent,
            case_thickness=case_thickness,
            logger=self._logger,
            **kwargs,
        )

        self.magnet_set.populate_magnet_coils()
        self.magnet_set.build_magnet_coils()

    def add_magnets_from_geometry(self, geometry_file, **kwargs):
        """Adds custom geometry via the MagnetSetFromGeometry class
        Arguments:
            geometry_file (str): path to the existing coil geometry. Can be of
                the types supported by cubit_io.import_geom_to_cubit()
            logger (object): logger object (optional, defaults to None). If no
                logger is supplied, a default logger will be instantiated.

        Optional attributes:
            mat_tag (str or iterable of str): DAGMC material tag(s) to use for
                magnets in DAGMC neutronics model (defaults to 'magnets'). If
                an iterable is given, the first entry will be applied to coil
                casing and the second to the inner volume. If just one is
                given, it will be applied to all magnet volumes.
            volume_ids (2-D iterable of int): list of ID pairs for
                (outer, inner) volume pairs, as imported by CadQuery or Cubit,
                beginning from 0.
        """
        self.magnet_set = mc.MagnetSetFromGeometry(
            geometry_file,
            logger=self._logger,
            **kwargs,
        )

    def export_magnets_step(self, filename="magnet_set", export_dir=""):
        """Export STEP file of magnet set.

        Arguments:
            filename (str): name of STEP export output file (optional, defaults
                to 'magnet_set').
            export_dir (str): directory to which to export output files
                (optional, defaults to empty string).
        """
        self.magnet_set.export_step(filename=filename, export_dir=export_dir)

    def export_magnet_mesh_cubit(
        self,
        filename="magnet_mesh",
        mesh_size=5,
        anisotropic_ratio=100.0,
        deviation_angle=5.0,
        volumes_to_mesh="both",
        export_dir="",
    ):
        """Creates a tetrahedral mesh of magnet volumes via Coreform Cubit and
        exports the mesh as a H5M file.

        Arguments:
            filename (str): name of H5M output file (defaults to
                'magnet_mesh').
            mesh_size (int): controls the size of the mesh. Takes values
                between 1 (finer) and 10 (coarser) (optional, defaults to 5).
            anisotropic_ratio (float): controls edge length ratio of elements
                (defaults to 100.0).
            deviation_angle (float): controls deviation angle of facet from
                surface (i.e., lesser deviation angle results in more elements
                in areas with higher curvature) (defaults to 5.0).
            volumes_to_mesh (str): volumes to include in mesh. Acceptable
                values are "inner", "outer", and "both" (defaults to "both").
                If no casing was modeled, all volumes will be meshed.
            export_dir (str): directory to which to export the h5m output file
                (defaults to empty string).
        """
        self.magnet_set.mesh_magnets_cubit(
            mesh_size=mesh_size,
            anisotropic_ratio=anisotropic_ratio,
            deviation_angle=deviation_angle,
            volumes_to_mesh=volumes_to_mesh,
        )
        self.magnet_set.export_mesh_cubit(
            filename=filename, export_dir=export_dir
        )

    def export_magnet_mesh_gmsh(
        self,
        filename="magnet_mesh",
        min_mesh_size=5.0,
        max_mesh_size=20.0,
        algorithm=1,
        volumes_to_mesh="both",
        export_dir="",
    ):
        """Creates a tetrahedral mesh of magnet volumes via Coreform Cubit and
        exports the mesh as a H5M file.

        Arguments:
            filename (str): name of H5M output file (defaults to
                'magnet_mesh').
            min_mesh_size (float): minimum size of mesh elements (defaults to
                5.0).
            max_mesh_size (float): maximum size of mesh elements (defaults to
                20.0).
            algorithm (int): integer identifying the meshing algorithm to use
                for the surface boundary (defaults to 1). Options are as
                follows, refer to Gmsh documentation for explanations of each.
                1: MeshAdapt, 2: automatic, 3: initial mesh only, 4: N/A,
                5: Delaunay, 6: Frontal-Delaunay, 7: BAMG, 8: Frontal-Delaunay
                for Quads, 9: Packing of Parallelograms, 11: Quasi-structured
                Quad.
            volumes_to_mesh (str): volumes to include in mesh. Acceptable
                values are "inner", "outer", and "both" (defaults to "both").
                If no casing was modeled, all volumes will be meshed.
            export_dir (str): directory to which to export the h5m output file
                (defaults to empty string).
        """
        self.magnet_set.mesh_magnets_gmsh(
            min_mesh_size=min_mesh_size,
            max_mesh_size=max_mesh_size,
            algorithm=algorithm,
            volumes_to_mesh=volumes_to_mesh,
        )
        self.magnet_set.export_mesh_gmsh(
            filename=filename, export_dir=export_dir
        )

    def construct_source_mesh(
        self, cfs_values, poloidal_angles, toroidal_angles, **kwargs
    ):
        """Constructs SourceMesh class object.

        Arguments:
            cfs_values (iterable of float): grid points along the closed flux
                surface axis of flux-coordinate space. Must begin at 0.0 and
                end at 1.0.
            poloidal_angles (iterable of float): grid points along the poloidal
                angle axis of flux-coordinate space. Must span 360 degrees.
            toroidal_angles (iterable of float): grid points along the toroidal
                angle axis of flux-coordinate space. Cannot span more than 360
                degrees.

        Optional attributes:
            scale (float): a scaling factor between input and output data
                (defaults to m2cm = 100).
            plasma_conditions (function): function that takes the plasma
                parameter s, and returns temperature and ion density with
                suitable units for the reaction_rate() function. Defaults to
                default_plasma_conditions()
            reaction_rate (function): function that takes the values returned by
                plasma_conditions() and returns a reaction rate in
                reactions/cm3/s
        """
        self.source_mesh = sm.SourceMesh(
            self._vmec_obj,
            cfs_values,
            poloidal_angles,
            toroidal_angles,
            logger=self._logger,
            **kwargs,
        )

        self.source_mesh.create_vertices()
        self.source_mesh.create_mesh()

    def export_source_mesh(self, filename="source_mesh", export_dir=""):
        """Export source mesh

        Arguments:
            filename (str): name of H5M output file, excluding '.h5m'
                extension (optional, defaults to 'source_mesh').
            export_dir (str): directory to which to export H5M output file
                (optional, defaults to empty string).
        """
        self.source_mesh.export_mesh(filename=filename, export_dir=export_dir)

    def _tag_materials(self):
        """Applies material tags to corresponding CAD volumes for DAGMC
        neutronics model export via Coreform Cubit.
        (Internal function not intended to be called externally)
        """
        if self.magnet_set:
            if isinstance(self.magnet_set.mat_tag, (list, tuple)):
                for idx, _ in enumerate(["outer", "inner"]):
                    mat_tag = self.magnet_set.mat_tag[idx]
                    volume_ids = list(self.magnet_set.volume_ids[:, idx])
                    volume_id_str = " ".join(str(i) for i in volume_ids)
                    block_id = min(volume_ids)
                    make_material_block(mat_tag, block_id, volume_id_str)
            else:
                volume_ids = self.magnet_set.volume_ids.flatten()
                volume_id_str = " ".join(str(i) for i in volume_ids)
                block_id = min(volume_ids)
                make_material_block(
                    self.magnet_set.mat_tag, block_id, volume_id_str
                )

        if self.invessel_build and not self.invessel_build.use_pydagmc:
            for data in self.invessel_build.radial_build.radial_build.values():
                block_id = data["vol_id"]
                vol_id_str = str(block_id)
                make_material_block(data["mat_tag"], block_id, vol_id_str)

    def build_cubit_model(self):
        """Build model for DAGMC neutronics H5M file of Parastell components via
        Coreform Cubit.
        """
        self._logger.info(
            "Building DAGMC neutronics model via Coreform Cubit..."
        )

        # Ensure fresh Cubit instance
        create_new_cubit_instance()

        if self.invessel_build and not self.use_pydagmc:
            self.invessel_build.import_step_cubit()
            self.invessel_build.merge_surfaces()

        if self.magnet_set:
            self.magnet_set.import_geom_cubit()
            self.magnet_set.merge_surfaces()

        self._tag_materials()

    def export_cubit_dagmc(
        self,
        filename="dagmc",
        export_dir="",
        anisotropic_ratio=100.0,
        deviation_angle=5.0,
    ):
        """Exports DAGMC neutronics H5M file of ParaStell components via
        Coreform Cubit.

        Arguments:
            filename (str): name of DAGMC output file, excluding '.h5m'
                extension (optional, defaults to 'dagmc').
            export_dir (str): directory to which to export DAGMC output file
                (optional, defaults to empty string).
            anisotropic_ratio (float): controls edge length ratio of elements
                (optional, defaults to 100.0).
            deviation_angle (float): controls deviation angle of facet from
                surface (i.e., lesser deviation angle results in more elements
                in areas with higher curvature) (optional, defaults to 5.0).
        """
        self._logger.info(
            "Exporting DAGMC neutronics model using Coreform Cubit..."
        )

        filename = Path(filename).with_suffix(".h5m")

        export_dagmc_cubit(
            filename=filename,
            export_dir=export_dir,
            anisotropic_ratio=anisotropic_ratio,
            deviation_angle=deviation_angle,
        )

        if self.use_pydagmc:
            self.magnet_model_path = Path(export_dir) / filename

    def export_cub5(self, filename="stellarator", export_dir=""):
        """Export native Coreform Cubit format (cub5) of Parastell model.

        Arguments:
            filename (str): name of DAGMC output file, excluding '.h5m'
                extension (optional, defaults to 'dagmc').
            export_dir (str): directory to which to export DAGMC output file
                (optional, defaults to empty string).
        """
        self._logger.info("Exporting cub5 model...")

        export_cub5(filename=filename, export_dir=export_dir)

    def build_cad_to_dagmc_model(self):
        """Build model for DAGMC neutronics H5M file of Parastell components via
        CAD-to-DAGMC.
        """
        self._logger.info(
            "Building DAGMC neutronics model via CAD-to-DAGMC..."
        )

        solids = []
        self._material_tags = []

        if self.invessel_build:
            ivb_solids, ivb_material_tags = (
                self.invessel_build.extract_solids_and_mat_tags()
            )
            solids.extend(ivb_solids)
            self._material_tags.extend(ivb_material_tags)

        if self.magnet_set:
            magnet_solids = self.magnet_set.all_coil_solids
            solids.extend(magnet_solids)

            if isinstance(self.magnet_set.mat_tag, (list, tuple)):
                magnet_mat_tags = self.magnet_set.mat_tag * len(
                    self.magnet_set.coil_solids
                )
            else:
                magnet_mat_tags = [self.magnet_set.mat_tag] * len(
                    magnet_solids
                )

            self._material_tags.extend(magnet_mat_tags)

        self._geometry = cq.Compound.makeCompound(solids)

    def export_cad_to_dagmc(
        self,
        filename="dagmc",
        export_dir="",
        min_mesh_size=5.0,
        max_mesh_size=20.0,
        algorithm=1,
    ):
        """Exports DAGMC neutronics H5M file of ParaStell components via
        CAD-to-DAGMC.

        Arguments:
            filename (str): name of DAGMC output file, excluding '.h5m'
                extension (optional, defaults to 'dagmc').
            export_dir (str): directory to which to export DAGMC output file
                (optional, defaults to empty string).
            min_mesh_size (float): minimum size of mesh elements (defaults to
                5.0).
            max_mesh_size (float): maximum size of mesh elements (defaults to
                20.0).
            algorithm (int): integer identifying the meshing algorithm to use
                for the surface mesh (defaults to 1). Options are as follows,
                refer to Gmsh documentation for explanations of each.
                1: MeshAdapt, 2: automatic, 3: initial mesh only, 4: N/A,
                5: Delaunay, 6: Frontal-Delaunay, 7: BAMG, 8: Frontal-Delaunay
                for Quads, 9: Packing of Parallelograms, 11: Quasi-structured
                Quad.
        """
        self._logger.info(
            "Exporting DAGMC neutronics model with CAD-to-DAGMC ..."
        )

        export_path = Path(export_dir) / Path(filename).with_suffix(".h5m")

        gmsh_obj = cad_to_dagmc.init_gmsh()

        _, volumes = cad_to_dagmc.get_volumes(
            gmsh_obj, self._geometry, method="in memory"
        )

        cad_to_dagmc.set_sizes_for_mesh(
            gmsh_obj,
            min_mesh_size=min_mesh_size,
            max_mesh_size=max_mesh_size,
            mesh_algorithm=algorithm,
        )

        gmsh_obj.model.mesh.generate(dim=2)

        vertices, triangles_by_solid_and_by_face = (
            cad_to_dagmc.mesh_to_vertices_and_triangles(volumes)
        )

        gmsh_obj.finalize()

        cad_to_dagmc.vertices_to_h5m(
            vertices,
            triangles_by_solid_and_by_face,
            self._material_tags,
            h5m_filename=export_path,
        )
        if self.use_pydagmc:
            self.magnet_model_path = export_path

    def build_pydagmc_model(self, magnet_exporter, **kwargs):
        """Combines the invessel build DAGMC model with a DAGMC model of the
        the magnets, as appropriate.

        Arguments:
            magnet_exporter (str): Software to use to mesh and export a DAGMC
                model of the magnet coils. Options are 'cubit' or
                'cad_to_dagmc'

            Optional Attributes:
            Valid optional attributes depend on which magnet exporter is
            being used.

            For magnet_exporter = 'cubit'
            skip_imprint (bool): choose whether to imprint and merge all in
                Coreform Cubit or to merge surfaces based on import order and
                geometry information (Defaults to False).
            filename (str): name of DAGMC output file, excluding '.h5m'
                extension (optional, defaults to 'dagmc').
            export_dir (str): directory to which to export DAGMC output file
                (optional, defaults to empty string).
            anisotropic_ratio (float): controls edge length ratio of elements
                (optional, defaults to 100.0).
            deviation_angle (float): controls deviation angle of facet from
                surface (i.e., lesser deviation angle results in more elements
                in areas with higher curvature) (optional, defaults to 5.0).

            For 'cad_to_dagmc' the optional arguments are:
            filename (str): name of DAGMC output file, excluding '.h5m'
                extension (optional, defaults to 'dagmc').
            export_dir (str): directory to which to export DAGMC output file
                (optional, defaults to empty string).
            min_mesh_size (float): minimum size of mesh elements (defaults to
                20).
            max_mesh_size (float): maximum size of mesh elements (defaults to
                50).
        """

        if self.magnet_set:
            if magnet_exporter == "cubit":
                self.build_cubit_model(
                    **(filter_kwargs(kwargs, build_cubit_model_allowed_kwargs))
                )
                self.export_cubit_dagmc(
                    **(
                        filter_kwargs(
                            kwargs, export_cubit_dagmc_allowed_kwargs
                        )
                    )
                )
            elif magnet_exporter == "cad_to_dagmc":
                self.build_cad_to_dagmc_model(
                    **(
                        filter_kwargs(
                            kwargs,
                            build_cad_to_dagmc_model_allowed_kwargs,
                        )
                    )
                )
                self.export_cad_to_dagmc(
                    **(
                        filter_kwargs(
                            kwargs, export_cad_to_dagmc_allowed_kwargs
                        )
                    )
                )
            magnet_mbc = core.Core()
            magnet_mbc.load_file(str(self.magnet_model_path))
            self.pydagmc_model = combine_dagmc_models(
                [self.invessel_build.dag_model.mb, magnet_mbc]
            )
        else:
            self.pydagmc_model = self.invessel_build.dag_model

    def export_pydagmc_model(self, filename="dagmc", export_dir=""):
        """Saves the PyDAGMC model to .h5m format.

        Arguments:
            filename (str): name of DAGMC output file, defaults to 'dagmc'
            export_dir (str): directory to save the output file, defaults to
                current directory
        """

        export_path = Path(export_dir) / Path(filename).with_suffix(".h5m")
        self.pydagmc_model.write_file(str(export_path))


def parse_args():
    """Parser for running as a script."""
    parser = argparse.ArgumentParser(prog="parastell")

    parser.add_argument(
        "filename",
        help="YAML file defining ParaStell stellarator configuration",
    )
    parser.add_argument(
        "-e",
        "--export_dir",
        default="",
        help=(
            "directory to which output files are exported (default: working "
            "directory)"
        ),
        metavar="",
    )
    parser.add_argument(
        "-l",
        "--logger",
        action="store_true",
        help=(
            "flag to indicate the instantiation of a logger object (default: "
            "False)"
        ),
    )

    parser.add_argument(
        "-i",
        "--ivb",
        action="store_true",
        help=(
            "flag to indicate the creation of in-vessel component geometry "
            "(default: False)"
        ),
    )

    parser.add_argument(
        "-m",
        "--magnets",
        action="store_true",
        help=(
            "flag to indicate the creation of magnet geometry (default: False)"
        ),
    )

    parser.add_argument(
        "-s",
        "--source",
        action="store_true",
        help=(
            "flag to indicate the creation of a tetrahedral source mesh "
            "(default: False)"
        ),
    )

    parser.add_argument(
        "-n",
        "--nwl",
        action="store_true",
        help=(
            "flag to indicate the creation of a geometry for neutron wall "
            "loading calculations (default: False)"
        ),
    )

    return parser.parse_args()


def check_inputs(invessel_build, magnet_coils, source_mesh, logger):
    """Checks inputs for consistency across ParaStell classes.

    Arguments:
        invessel_build (dict): dictionary of RadialBuild and InVesselBuild
            parameters.
        magnet_coils (dict): dictionary of MagnetSet parameters.
        source_mesh (dict): dictionary of SourceMesh parameters.
        logger (object): logger object.
    """
    if "repeat" in invessel_build:
        repeat = invessel_build["repeat"]
    else:
        repeat = 0

    ivb_tor_ext = (repeat + 1) * invessel_build["toroidal_angles"][-1]
    mag_tor_ext = magnet_coils["toroidal_extent"]
    src_tor_ext = source_mesh["toroidal_extent"]

    if ivb_tor_ext != mag_tor_ext:
        w = Warning(
            f"The total toroidal extent of the in-vessel build, {ivb_tor_ext} "
            "degrees, does not match the toroidal extent of the magnet coils, "
            f"{mag_tor_ext} degrees."
        )
        logger.warning(w.args[0])

    if ivb_tor_ext != src_tor_ext:
        w = Warning(
            f"The total toroidal extent of the in-vessel build, {ivb_tor_ext} "
            "degrees, does not match the toroidal extent of the source mesh, "
            f"{src_tor_ext} degrees."
        )
        logger.warning(w.args[0])

    if mag_tor_ext != src_tor_ext:
        w = Warning(
            f"The toroidal extent of the magnet coils, {mag_tor_ext} degrees, "
            f"does not match that of the source mesh, {src_tor_ext} degrees."
        )
        logger.warning(w.args[0])

    if "scale" in invessel_build:
        ivb_scale = invessel_build["scale"]
    else:
        ivb_scale = m2cm

    if "scale" in source_mesh:
        src_scale = source_mesh["scale"]
    else:
        src_scale = m2cm

    if ivb_scale != src_scale:
        e = ValueError(
            f"The conversion scale of the in-vessel build, {ivb_scale}, does "
            f"not match that of the source mesh, {src_scale}."
        )
        logger.error(e.args[0])
        raise e


def parastell():
    """Main method when run as a command line script."""
    args = parse_args()

    all_data = read_yaml_config(args.filename)

    if args.logger == True:
        logger = log.init()
    else:
        logger = log.NullLogger()

    check_inputs(
        all_data["invessel_build"],
        all_data["magnet_coils"],
        all_data["source_mesh"],
        all_data["dagmc_export"],
        logger,
    )

    vmec_file = all_data["vmec_file"]

    stellarator = Stellarator(vmec_file, logger=logger)

    if args.ivb:
        invessel_build = all_data["invessel_build"]
        stellarator.construct_invessel_build(**invessel_build)
        stellarator.export_invessel_build(export_dir=args.export_dir)

    if args.magnets:
        magnet_coils = all_data["magnet_coils"]
        stellarator.construct_magnets(**magnet_coils)
        stellarator.export_magnets(
            export_dir=args.export_dir,
            **(filter_kwargs(magnet_coils, mc.export_allowed_kwargs)),
        )

    if args.source:
        source_mesh = all_data["source_mesh"]
        stellarator.construct_source_mesh(**source_mesh)
        stellarator.export_source_mesh(
            export_dir=args.export_dir,
            **(filter_kwargs(source_mesh, sm.export_allowed_kwargs)),
        )

    if args.ivb or args.magnets:
        dagmc_export = all_data["dagmc_export"]
        stellarator.build_cubit_model(
            **(filter_kwargs(dagmc_export, build_cubit_model_allowed_kwargs))
        )
        stellarator.export_cubit_dagmc(
            export_dir=args.export_dir,
            **(filter_kwargs(dagmc_export, export_cubit_dagmc_allowed_kwargs)),
        )

        if all_data["cub5_export"]:
            stellarator.export_cub5(export_dir=args.export_dir)

    if args.nwl:
        if not args.ivb:
            invessel_build = all_data["invessel_build"]
            if not args.magnets:
                dagmc_export = all_data["dagmc_export"]

        nwl_geom = Stellarator(vmec_file, logger=logger)

        nwl_required_keys = ["toroidal_angles", "poloidal_angles", "wall_s"]

        nwl_build = {}
        for key in nwl_required_keys:
            nwl_build[key] = invessel_build[key]
        nwl_build["radial_build"] = {}

        nwl_optional_keys = ["num_ribs", "num_rib_pts", "repeat", "scale"]

        for key in invessel_build.keys() & nwl_optional_keys:
            nwl_build[key] = invessel_build[key]

        nwl_geom.construct_invessel_build(**nwl_build)
        nwl_geom.export_invessel_build(export_dir=args.export_dir)

        nwl_geom.build_cubit_model(skip_imprint=True)
        nwl_geom.export_cubit_dagmc(
            filename="nwl_geom", export_dir=args.export_dir
        )


if __name__ == "__main__":
    parastell()
