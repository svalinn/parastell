import argparse
from pathlib import Path
from abc import ABC

import numpy as np
import cadquery as cq
import gmsh
from pymoab import core

from . import log
from .utils import get_obmp_index, orient_coords
from .cubit_utils import (
    create_new_cubit_instance,
    import_geom_to_cubit,
    export_mesh_cubit,
    merge_volumes,
    mesh_volume_auto_factor,
    mesh_surface_coarse_trimesh,
    get_last_id,
)

# Import cubit_utils separately for its initialized variable. If initialized is
# imported into this namespace, changes to the variable do not persist when
# modified by calls to the imported functions
from . import cubit_utils
from .utils import read_yaml_config, filter_kwargs, reorder_loop, m2cm

export_allowed_kwargs = ["step_filename", "export_mesh", "mesh_filename"]


class MagnetSet(ABC):
    """An object representing a set of modular stellarator magnet coils.

    Arguments:
        logger (object): logger object (optional, defaults to None). If no
            logger is supplied, a default logger will be instantiated.

    Optional attributes:
        mat_tag (str or iterable of str): DAGMC material tag(s) to use for
            magnets in DAGMC neutronics model (defaults to 'magnets'). If an
            iterable is given, the first entry will be applied to coil casing
            and the second to the inner volume. If just one is given, it will
            be applied to all magnet volumes.
    """

    def __init__(
        self,
        logger=None,
        **kwargs,
    ):

        self.logger = logger

        self.mat_tag = "magnets"

        for name in kwargs.keys() & ("mat_tag",):
            self.__setattr__(name, kwargs[name])

        self.has_casing = False

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, logger_object):
        self._logger = log.check_init(logger_object)

    @property
    def all_coil_solids(self):
        return [solid for solids in self.coil_solids for solid in solids]

    def import_geom_cubit(self):
        """Import geometry file for magnet set into Coreform Cubit."""
        first_vol_id = 1
        if cubit_utils.initialized:
            first_vol_id += get_last_id("volume")

        import_geom_to_cubit(self.geometry_file, self.working_dir)
        self.cubit_volume_ids = self.volume_ids + first_vol_id

    def merge_surfaces(self):
        """Merges ParaStell inner and outer magnet volumes in Coreform Cubit
        based on volume IDs rather than imprinting and merging all. Note that
        overlaps between magnet volumes and in-vessel components will not be
        merged in this workflow.
        """
        if self.has_casing:
            [merge_volumes(id_pair) for id_pair in self.cubit_volume_ids]

    def mesh_magnets_cubit(
        self,
        mesh_size=5,
        anisotropic_ratio=100.0,
        deviation_angle=5.0,
        volumes_to_mesh="both",
    ):
        """Creates tetrahedral mesh of magnet volumes via Coreform Cubit.

        Arguments:
            mesh_size (float): controls the size of the mesh. Takes values
                between 1.0 (finer) and 10.0 (coarser) (defaults to 5.0).
            anisotropic_ratio (float): controls edge length ratio of elements
                (defaults to 100.0).
            deviation_angle (float): controls deviation angle of facet from
                surface (i.e., lesser deviation angle results in more elements
                in areas with higher curvature) (defaults to 5.0).
            volumes_to_mesh (str): volumes to include in mesh. Acceptable
                values are "inner", "outer", and "both" (defaults to "both").
                If no casing was modeled, all volumes will be meshed.
        """
        self._logger.info("Generating tetrahedral mesh of magnet coils...")

        create_new_cubit_instance()

        # Overwrite any volume IDs
        self.import_geom_cubit()

        if self.has_casing:
            if volumes_to_mesh == "inner":
                volume_ids = self.cubit_volume_ids[:, 1]
            elif volumes_to_mesh == "outer":
                volume_ids = self.cubit_volume_ids[:, 0]
            elif volumes_to_mesh == "both":
                volume_ids = self.cubit_volume_ids.flatten()
            else:
                e = ValueError(
                    f"Value specified for volumes_to_mesh, {volumes_to_mesh}, "
                    "not recognized. Please use 'inner', 'outer', or 'both'."
                )
                raise e
        else:
            volume_ids = self.cubit_volume_ids.flatten()

        mesh_surface_coarse_trimesh(
            anisotropic_ratio=anisotropic_ratio,
            deviation_angle=deviation_angle,
        )
        mesh_volume_auto_factor([id for id in volume_ids], mesh_size=mesh_size)

    def export_mesh_cubit(self, filename="magnet_mesh", export_dir=""):
        """Exports a tetrahedral mesh of magnet volumes in H5M format via
        Coreform Cubit and MOAB.

        Arguments:
            filename (str): name of H5M output file (defaults to
                'magnet_mesh').
            export_dir (str): directory to which to export the H5M output file
                (defaults to empty string).
        """
        self._logger.info("Exporting mesh H5M file...")

        export_mesh_cubit(
            filename=filename,
            export_dir=export_dir,
            delete_upon_export=True,
        )

    def mesh_magnets_gmsh(
        self,
        min_mesh_size=5.0,
        max_mesh_size=20.0,
        algorithm=1,
        volumes_to_mesh="both",
    ):
        """Creates tetrahedral mesh of magnet volumes via Gmsh.

        Arguments:
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
        """
        self._logger.info("Generating tetrahedral mesh of magnets via Gmsh...")

        gmsh.initialize()

        gmsh.option.setNumber(
            "General.NumThreads", 0
        )  # Use all available cores

        if self.has_casing:
            if volumes_to_mesh == "inner":
                solids_to_mesh = [solids[1] for solids in self.coil_solids]
            elif volumes_to_mesh == "outer":
                solids_to_mesh = [solids[0] for solids in self.coil_solids]
            elif volumes_to_mesh == "both":
                solids_to_mesh = self.all_coil_solids
            else:
                e = ValueError(
                    f"Value specified for volumes_to_mesh, {volumes_to_mesh}, "
                    "not recognized. Please use 'inner', 'outer', or 'both'."
                )
                raise e
        else:
            solids_to_mesh = self.all_coil_solids

        mesh_geometry = cq.Compound.makeCompound(solids_to_mesh)

        gmsh.model.occ.importShapesNativePointer(
            mesh_geometry.wrapped._address()
        )

        gmsh.model.occ.synchronize()

        gmsh.option.setNumber("Mesh.MeshSizeMin", min_mesh_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", max_mesh_size)
        gmsh.option.setNumber("Mesh.Algorithm", algorithm)

        gmsh.model.mesh.generate(dim=3)

    def export_mesh_gmsh(self, filename="magnet_mesh", export_dir=""):
        """Exports a tetrahedral mesh of magnet volumes in H5M format via Gmsh
        and MOAB.

        Arguments:
            filename (str): name of H5M output file (defaults to
                'magnet_mesh').
            export_dir (str): directory to which to export the h5m output file
                (defaults to empty string).
        """
        self._logger.info("Exporting mesh H5M file...")

        vtk_path = Path(export_dir) / Path(filename).with_suffix(".vtk")
        moab_path = vtk_path.with_suffix(".h5m")

        gmsh.write(str(vtk_path))

        gmsh.clear()
        gmsh.finalize()

        self.mesh_mbc = core.Core()
        self.mesh_mbc.load_file(str(vtk_path))
        self.mesh_mbc.write_file(str(moab_path))

        Path(vtk_path).unlink()


class MagnetSetFromFilaments(MagnetSet):
    """Inherits from MagnetSet. This subclass enables the construction of
    CAD solids using the filament data and CADquery.

    Arguments:
        coils_file (str): path to coil filament data file.
        width (float): width of coil cross-section in toroidal direction [cm].
        thickness (float): thickness of coil cross-section in radial direction
            [cm].
        toroidal_extent (float): toroidal extent to model [deg].
        case_thickness (float): thickness of outer coil casing (defaults to
            0.0) [cm]. Double this amount will be subtracted from the width and
            thickness parameters to form the inner coil volume.
        logger (object): logger object (optional, defaults to None). If no
            logger is supplied, a default logger will be instantiated.

    Optional attributes:
        start_line (int): starting line index for data in filament data file
            (defaults to 3).
        sample_mod (int): sampling modifier for filament points (defaults to
            1). For a user-defined value n, every nth point will be sampled.
        scale (float): a scaling factor between input and output data
            (defaults to m2cm = 100).
        mat_tag (str or iterable of str): DAGMC material tag(s) to use for
            magnets in DAGMC neutronics model (defaults to 'magnets'). If an
            iterable is given, the first entry will be applied to coil casing
            and the second to the inner volume. If just one is given, it will
            be applied to all magnet volumes.
    """

    def __init__(
        self,
        coils_file,
        width,
        thickness,
        toroidal_extent,
        case_thickness=0.0,
        logger=None,
        **kwargs,
    ):
        super().__init__(logger, **kwargs)

        self.coils_file = coils_file

        self.start_line = 3
        self.sample_mod = 1
        self.scale = m2cm

        self.width = width
        self.thickness = thickness
        self.toroidal_extent = toroidal_extent
        self.case_thickness = case_thickness

        # Define maximum length of coil cross-section
        self.max_cs_len = max(self._width, self._thickness)

        if "scale" not in kwargs.keys():
            w = Warning(
                "No factor specified to scale MagnetSet input data. "
                "Assuming a scaling factor of 100.0, which is consistent with "
                "input being in units of [m] and desired output in units of "
                "[cm]."
            )
            self._logger.warning(w.args[0])

        for name in kwargs.keys() & (
            "start_line",
            "sample_mod",
            "scale",
            "mat_tag",
        ):
            self.__setattr__(name, kwargs[name])

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = value
        if self._width < 0.0:
            e = ValueError("Coil cross-section width cannot be negative.")
            self._logger.error(e.args[0])
            raise e

    @property
    def thickness(self):
        return self._thickness

    @thickness.setter
    def thickness(self, value):
        self._thickness = value
        if self._thickness < 0.0:
            e = ValueError("Coil cross-section thickness cannot be negative.")
            self._logger.error(e.args[0])
            raise e

    @property
    def toroidal_extent(self):
        return self._toroidal_extent

    @toroidal_extent.setter
    def toroidal_extent(self, angle):
        self._toroidal_extent = np.deg2rad(angle)
        if self._toroidal_extent > 360.0:
            e = ValueError("Toroidal extent cannot exceed 360.0 degrees.")
            self._logger.error(e.args[0])
            raise e

    @property
    def case_thickness(self):
        return self._case_thickness

    @case_thickness.setter
    def case_thickness(self, value):
        if value < 0.0:
            e = ValueError("Coil case thickness cannot be negative.")
            self._logger.error(e.args[0])
            raise e

        self._case_thickness = value

        if value > 0.0:
            self.has_casing = True

    def _instantiate_filaments(self):
        """Extracts filament coordinate data from input data file and
        instantiates Filament class objects.
        (Internal function not intended to be called externally)
        """
        with open(self.coils_file, "r") as file:
            data = file.readlines()[self.start_line :]

        coords = []
        self.filaments = []

        for line in data:
            columns = line.strip().split()

            if columns[0] == "end":
                break

            # Coil current
            s = float(columns[3])

            # s = 0 signals end of filament
            if s != 0:
                coords.append(
                    [float(ord) * self.scale for ord in columns[0:3]]
                )

            else:
                coords.append(coords[0])
                self.filaments.append(Filament(np.array(coords)))
                coords.clear()

    def sort_filaments_toroidally(self):
        """Reorders list of filaments by toroidal angle on range [-pi, pi].

        Returns:
            (list of object): sorted list of Filament class objects.
        """
        return sorted(self.filaments, key=lambda x: x.com_toroidal_angle)

    def _filter_filaments(self, tol=0):
        """Filters list of Filament objects such that only those within the
        toroidal extent of the model are included and filaments are sorted by
        center-of-mass toroidal angle.
        (Internal function not intended to be called externally)
        """

        # Compute lower and upper bounds of toroidal extent within tolerance
        lower_bound = 2 * np.pi - tol
        upper_bound = self._toroidal_extent + tol

        # Create filter determining whether each coil lies within model's
        # toroidal extent
        filtered_filaments = [
            filament
            for filament in self.filaments
            if filament.in_toroidal_extent(lower_bound, upper_bound)
        ]
        self.filaments = filtered_filaments

        # Sort coils by center-of-mass toroidal angle and overwrite stored list
        self.filaments = self.sort_filaments_toroidally()

    def _instantiate_coils(self):
        """Instantiates MagnetCoil class objects using filament data.
        (Internal function not intended to be called externally)
        """
        self.magnet_coils = []
        for filament in self.filaments:
            self.magnet_coils.append(
                MagnetCoil(
                    filament,
                    self.width,
                    self.thickness,
                    self.case_thickness,
                    self.sample_mod,
                )
            )

    def _compute_radial_distance_data(self):
        """Computes average and maximum radial distance of filament points.
        (Internal function not intended to be called externally)
        """
        radii_count = 0
        self.average_radial_distance = 0
        self.max_radial_distance = -1

        for filament in self.filaments:
            radii = np.linalg.norm(filament.coords[:-1, :2], axis=1)
            radii_count += len(radii)
            self.average_radial_distance += np.sum(radii)
            self.max_radial_distance = max(
                self.max_radial_distance, np.max(radii)
            )

        self.average_radial_distance /= radii_count

    def _create_magnet_boundary(self):
        """Creates a CadQuery solid spanning the toroidal domain of the
        magnets.
        (Internal function not intended to be called externally)

        Returns:
            (object): cq.Solid object representing the boundary of the toroidal
                domain of the magnets.
        """
        side_length = 1.25 * self.max_radial_distance

        toroidal_domain = cq.Workplane("XZ")
        toroidal_domain = toroidal_domain.transformed(
            offset=(side_length / 2, 0)
        )
        toroidal_domain = toroidal_domain.rect(side_length, side_length)
        toroidal_domain = toroidal_domain.revolve(
            np.rad2deg(self._toroidal_extent),
            (-side_length / 2, 0),
            (-side_length / 2, 1),
        )

        return toroidal_domain.val()

    def populate_magnet_coils(self):
        """Populates MagnetCoil class objects representing each of the magnetic
        coils that lie within the specified toroidal extent.
        """
        self._logger.info("Populating magnet coils...")

        self._instantiate_filaments()
        self._compute_radial_distance_data()
        # Define tolerance of toroidal extent to account for dimensionality of
        # coil cross-section
        # Multiply by factor of 2 to be conservative
        tol = 2 * np.arctan2(self.max_cs_len, self.average_radial_distance)
        self._filter_filaments(tol=tol)
        self._instantiate_coils()

    def build_magnet_coils(self):
        """Builds each filament in self.filtered_filaments in cubit, then cuts
        to the toroidal extent using self._create_magnet_boundary().
        """
        self._logger.info("Constructing magnet coils...")

        toroidal_domain = self._create_magnet_boundary()

        [
            magnet_coil.create_magnet(toroidal_domain)
            for magnet_coil in self.magnet_coils
        ]

        # Check outer solid for volume only; if volume = 0, don't include
        self.magnet_coils = [
            coil
            for coil in self.magnet_coils
            if (coil.solids[0].Volume() != 0)
        ]

        self.coil_solids = [coil.solids for coil in self.magnet_coils]

        if self.has_casing:
            self.volume_ids = np.array(
                [(idx, idx + 1) for idx, _ in enumerate(self.coil_solids)]
            )
        else:
            self.volume_ids = np.array(
                [(idx) for idx, _ in enumerate(self.coil_solids)]
            )

    def export_step(self, filename="magnet_set", export_dir=""):
        """Export CAD solids as a STEP file via CadQuery.

        Arguments:
            filename (str): name of STEP output file (optional, defaults to
                'magnet_set').
            export_dir (str): directory to which to export the STEP output file
                (optional, defaults to empty string).
        """
        self._logger.info("Exporting STEP file for magnet coils...")

        self.working_dir = export_dir
        self.geometry_file = Path(filename).with_suffix(".step")

        export_path = Path(self.working_dir) / self.geometry_file

        # Flatten list of solids (inner, outer solid pairs)
        solids_list = [
            solid for solids in self.coil_solids for solid in solids
        ]

        coil_set = cq.Compound.makeCompound(solids_list)

        cq.exporters.export(coil_set, str(export_path))


class MagnetSetFromGeometry(MagnetSet):
    """An object representing a set of modular stellarator magnet coils
    with previously defined geometry files.

    Arguments:
        geometry_file (str): path to the existing coil geometry. If using Cubit
            functionality, can be of types STEP or CUB5. If using Gmsh or
            CAD-to-DAGMC functionality, must be of type STEP.
        logger (object): logger object (optional, defaults to None). If no
            logger is supplied, a default logger will be instantiated.

    Optional attributes:
        mat_tag (str or iterable of str): DAGMC material tag(s) to use for
            magnets in DAGMC neutronics model (defaults to 'magnets'). If an
            iterable is given, the first entry will be applied to coil casing
            and the second to the inner volume. If just one is given, it will
            be applied to all magnet volumes.
        volume_ids (2-D iterable of int): list of ID pairs for (outer, inner)
            volume pairs, as imported by CadQuery or Cubit, beginning from 0.
    """

    def __init__(
        self,
        geometry_file,
        logger=None,
        **kwargs,
    ):
        super().__init__(logger, **kwargs)
        self.geometry_file = Path(geometry_file).resolve()
        self.working_dir = self.geometry_file.parent

        self.volume_ids = None

        for name in kwargs.keys() & (
            "start_line",
            "sample_mod",
            "scale",
            "mat_tag",
            "volume_ids",
        ):
            self.__setattr__(name, kwargs[name])

    @property
    def geometry_file(self):
        return self._geometry_file

    @geometry_file.setter
    def geometry_file(self, file_path):
        self._geometry_file = file_path

        imported_geometry = cq.importers.importStep(
            str(self.geometry_file)
        ).vals()

        self.coil_solids = []
        for item in imported_geometry:
            if isinstance(item, cq.occ_impl.shapes.Compound):
                self.coil_solids.extend([[solid] for solid in item.Solids()])
            elif isinstance(item, cq.occ_impl.shapes.Solid):
                self.coil_solids.append([item])
            else:
                e = ValueError(
                    f"Imported object of type {type(item)} not recognized."
                )
                self._logger.error(e.args[0])

    @property
    def volume_ids(self):
        return self._volume_ids

    @volume_ids.setter
    def volume_ids(self, value):
        if value is None:
            self._volume_ids = np.array(
                [[idx] for idx, _ in enumerate(self.coil_solids)]
            )
        elif isinstance(self.mat_tag, (list, tuple)):
            self.has_casing = True
            value = np.array(value)

            if set(value[:, 0]) & set(value[:, 1]):
                e = ValueError(
                    "At least one ID pair is not unique. Ensure all IDs are "
                    "included only once."
                )
                self._logger.error(e.args[0])

            if len(value.flatten()) != len(self.all_coil_solids):
                e = ValueError(
                    "Total number of IDs given does not match the number of "
                    "imported solids."
                )
                self._logger.error(e.args[0])

            self._volume_ids = value

            self.coil_solids = [
                [self.coil_solids[outer_id][0], self.coil_solids[inner_id][0]]
                for outer_id, inner_id in value
            ]


class Filament(object):
    """Object containing basic data defining a Filament, and necessary
    functions for working with that data.

    Arguments:
        coords (2-D array of float): set of Cartesian coordinates defining
            magnet filament location.
    """

    def __init__(self, coords):
        self.coords = coords

    @property
    def coords(self):
        return self._coords

    @coords.setter
    def coords(self, data):
        self._coords = data

        tangents = np.subtract(
            np.append(data[1:], [data[1]], axis=0),
            np.append([data[-2]], data[0:-1], axis=0),
        )
        self.tangents = (
            tangents / np.linalg.norm(tangents, axis=1)[:, np.newaxis]
        )

        self.com = np.average(data[:-1], axis=0)
        self.com_toroidal_angle = np.arctan2(self.com[1], self.com[0])

    def get_obmp_index(self):
        """Finds the index of the outboard midplane coordinate on a coil
        filament.

        Returns:
            outboard_index (int): index of the outboard midplane point.
        """
        # Compute radial distance of coordinates from z-axis
        radii = np.linalg.norm(self.coords[:, :2], axis=1)
        z_values = self.coords[:, 2]

        # Reformulate coordinates as R, Z pairs
        coords = np.array(list(zip(radii, z_values)))

        return get_obmp_index(coords)

    def reorder_coords(self, index):
        """Reorders coil filament coordinate loop about a given index.

        Arguments:
            index (int): index about which to reorder coordinate loop.
        """
        self.coords = reorder_loop(self.coords, index)

    def orient_coords(self, positive=True):
        """Orients coil filament coordinate loop such that they initially
        progress positively or negatively.

        Arguments:
            positive (bool): progress coordinates in positive direciton
                (defaults to True). If negative, coordinates will progress in
                negative direction.
        """
        self.coords = orient_coords(self.coords, positive=positive)

    def in_toroidal_extent(self, lower_bound, upper_bound):
        """Determines if the coil lies within a given toroidal angular extent,
        based on filament coordinates.

        Arguments:
            lower_bound (float): lower bound of toroidal extent [rad].
            upper_bound (float): upper bound of toroidal extent [rad].

        Returns:
            in_toroidal_extent (bool): flag to indicate whether coil lies
                within toroidal bounds.
        """
        # Compute toroidal angle of each point in filament
        toroidal_angles = np.arctan2(self.coords[:, 1], self.coords[:, 0])
        # Ensure angles are positive
        toroidal_angles = (toroidal_angles + 2 * np.pi) % (2 * np.pi)
        # Compute bounds of toroidal extent of filament
        min_tor_ang = np.min(toroidal_angles)
        max_tor_ang = np.max(toroidal_angles)

        # Determine if filament toroidal extent overlaps with that of model
        if (min_tor_ang >= lower_bound or min_tor_ang <= upper_bound) or (
            max_tor_ang >= lower_bound or max_tor_ang <= upper_bound
        ):
            in_toroidal_extent = True
        else:
            in_toroidal_extent = False

        return in_toroidal_extent


class MagnetCoil(object):
    """An object representing a single modular stellarator magnet coil.

    Arguments:
        filament (Filament object): filament definining the location of the
            coil
        width (float): width of coil cross-section in toroidal direction [cm].
        thickness (float): thickness of coil cross-section in radial direction
            [cm].
        case_thickness (float): thickness of outer coil casing (defaults to
            0.0) [cm]. Double this amount will be subtracted from the width and
            thickness parameters to form the inner coil volume.
        sample_mod (int): Length of stride when sampling from coordinate data.
    """

    def __init__(self, filament, width, thickness, case_thickness, sample_mod):

        self.filament = filament
        self.sample_mod = sample_mod
        self.coords = filament.coords
        self.center_of_mass = filament.com
        self.tangents = filament.tangents
        self.width = width
        self.thickness = thickness
        self.case_thickness = case_thickness

    def _create_magnet_solid(self, width, thickness):
        """Creates a single magnet CAD solid in CadQuery.
        (Internal function not intended to be called externally)

        Arguments:
            width (float): width of solid cross-section in toroidal direction
                [cm].
            thickness (float): thickness of solid cross-section in radial
                direction [cm].

        Returns:
            (object): cq.Solid object representing a single magnet volume.
        """
        # Sample filament coordinates and tangents by modifier
        coords = self.coords[0 : -1 : self.sample_mod]
        coords = np.append(coords, [self.coords[0]], axis=0)
        tangents = self.tangents[0 : -1 : self.sample_mod]
        tangents = np.append(tangents, [self.tangents[0]], axis=0)

        tangent_vectors = [cq.Vector(tuple(tangent)) for tangent in tangents]

        # Define coil filament path normals such that they face the filament
        # center of mass
        # Compute "outward" direction as difference between filament positions
        # and filament center of mass
        outward_dirs = coords - self.center_of_mass
        outward_dirs = (
            outward_dirs / np.linalg.norm(outward_dirs, axis=1)[:, np.newaxis]
        )

        # Project outward directions onto desired coil cross-section (CS) plane
        # at each filament position to define filament path normals
        parallel_parts = np.diagonal(
            np.matmul(outward_dirs, tangents.transpose())
        )

        normals = outward_dirs - parallel_parts[:, np.newaxis] * tangents
        normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]

        # Compute binormals projected onto CS plane at each position
        binormals = np.cross(tangents, normals)

        # Compute coordinates of edges of rectangular coils
        edge_offsets = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]])

        coil_edge_coords = []
        for edge_offset in edge_offsets:
            coil_edge = (
                coords
                + edge_offset[0] * binormals * (width / 2)
                + edge_offset[1] * normals * (thickness / 2)
            )

            coil_edge_coords.append(
                [cq.Vector(tuple(pos)) for pos in coil_edge]
            )

        # Append first edge once again
        coil_edge_coords.append(coil_edge_coords[0])

        coil_edges = [
            cq.Edge.makeSpline(coord_vectors, tangents=tangent_vectors).close()
            for coord_vectors in coil_edge_coords
        ]

        face_list = [
            cq.Face.makeRuledSurface(edge1, edge2)
            for edge1, edge2 in zip(coil_edges[:-1], coil_edges[1:])
        ]

        shell = cq.Shell.makeShell(face_list)

        return cq.Solid.makeSolid(shell)

    def create_magnet(self, toroidal_domain=None):
        """Creates a single magnet coil CAD solid in CadQuery.

        Arguments:
            toroidal_domain (object): cq.Solid object representing the boundary
                of the toroidal domain of the magnets.

        Returns:
            coil (object): cq.Solid object representing a single magnet coil.
        """
        magnet_solid = self._create_magnet_solid(self.width, self.thickness)

        if toroidal_domain:
            magnet_solid = magnet_solid.intersect(toroidal_domain)

        if self.case_thickness != 0.0:
            inner_solid = self._create_magnet_solid(
                self.width - 2 * self.case_thickness,
                self.thickness - 2 * self.case_thickness,
            )
            inner_solid = inner_solid.intersect(toroidal_domain)

            outer_solid = magnet_solid.cut(inner_solid)

            self.solids = [outer_solid, inner_solid]

        else:
            self.solids = [magnet_solid]


def parse_args():
    """Parser for running as a script"""
    parser = argparse.ArgumentParser(prog="magnet_coils")

    parser.add_argument(
        "filename", help="YAML file defining ParaStell magnet configuration"
    )
    parser.add_argument(
        "-e",
        "--export_dir",
        default="",
        help=(
            "Directory to which output files are exported (default: working "
            "directory)"
        ),
        metavar="",
    )
    parser.add_argument(
        "-l",
        "--logger",
        default=False,
        help=(
            "Flag to indicate whether to instantiate a logger object (default: "
            "False)"
        ),
        metavar="",
    )

    return parser.parse_args()


def generate_magnet_set():
    """Main method when run as command line script."""
    args = parse_args()

    all_data = read_yaml_config(args.filename)

    if args.logger == True:
        logger = log.init()
    else:
        logger = log.NullLogger()

    magnet_coils_dict = all_data["magnet_coils"]

    magnet_set = MagnetSet(
        magnet_coils_dict["coils_file"],
        magnet_coils_dict["cross_section"],
        magnet_coils_dict["toroidal_extent"],
        logger=logger**magnet_coils_dict,
    )

    magnet_set.build_magnet_coils()

    magnet_set.export_step(
        export_dir=args.export_dir,
        **(filter_kwargs(magnet_coils_dict, ["step_filename"])),
    )

    if magnet_coils_dict["export_mesh"]:
        magnet_set.export_mesh(
            export_dir=args.export_dir,
            **(filter_kwargs(magnet_coils_dict, ["mesh_filename"])),
        )


if __name__ == "__main__":
    generate_magnet_set()
