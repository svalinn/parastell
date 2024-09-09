import argparse
from pathlib import Path

import numpy as np

import cadquery as cq
import cubit

from . import log
from . import cubit_io as cubit_io
from .utils import normalize, read_yaml_config, filter_kwargs, m2cm

export_allowed_kwargs = ["step_filename", "export_mesh", "mesh_filename"]


class MagnetSet(object):
    """An object representing a set of modular stellarator magnet coils.

    Arguments:
        coils_file (str): path to coil filament data file.
        width (float): width of coil cross-section in toroidal direction [cm].
        thickness (float): thickness of coil cross-section in radial direction
            [cm].
        toroidal_extent (float): toroidal extent to model [deg].
        logger (object): logger object (optional, defaults to None). If no
            logger is supplied, a default logger will be instantiated.

    Optional attributes:
        start_line (int): starting line index for data in filament data file
            (defaults to 3).
        sample_mod (int): sampling modifier for filament points (defaults to
            1). For a user-defined value n, every nth point will be sampled.
        scale (float): a scaling factor between the units of the point-locus
            data and [cm] (defaults to m2cm = 100).
        mat_tag (str): DAGMC material tag to use for magnets in DAGMC
            neutronics model (defaults to 'magnets').
    """

    def __init__(
        self,
        coils_file,
        width,
        thickness,
        toroidal_extent,
        logger=None,
        **kwargs,
    ):

        self.logger = logger
        self.coils_file = coils_file
        self.width = width
        self.thickness = thickness
        self.toroidal_extent = toroidal_extent

        self.start_line = 3
        self.sample_mod = 1
        self.scale = m2cm
        self.mat_tag = "magnets"

        for name in kwargs.keys() & (
            "start_line",
            "sample_mod",
            "scale",
            "mat_tag",
        ):
            self.__setattr__(name, kwargs[name])

        # Define maximum length of coil cross-section
        self.max_cs_len = max(self._width, self._thickness)

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
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, logger_object):
        self._logger = log.check_init(logger_object)

    def _extract_filament_data(self):
        """Extracts filament coordinate numerical data from input data file.
        (Internal function not intended to be called externally)
        """
        with open(self.coils_file, "r") as file:
            data = file.readlines()[self.start_line :]

        coords = []
        filament_coords = []

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
                filament_coords.append(np.array(coords))
                coords.clear()

        self.filament_coords = filament_coords

    def _compute_radial_distance_data(self):
        """Computes average and maximum radial distance of filament points.
        (Internal function not intended to be called externally)
        """
        radii_count = 0
        self.average_radial_distance = 0
        self.max_radial_distance = -1

        for f in self.filament_coords:
            radii = np.linalg.norm(f[:, :2], axis=1)
            radii_count += len(radii)
            self.average_radial_distance += np.sum(radii)
            self.max_radial_distance = max(
                self.max_radial_distance, np.max(radii)
            )

        self.average_radial_distance /= radii_count

    def _filter_filaments(self):
        """Cleans filament data such that only filaments within the toroidal
        extent of the model are included and filaments are sorted by toroidal
        angle.
        (Internal function not intended to be called externally)
        """
        # Initialize coordinate data for filaments within toroidal extent of
        # model
        filtered_coords = []
        # Initialize list of filament centers of mass
        com_list = []

        # Define tolerance of toroidal extent to account for dimensionality of
        # coil cross-section
        # Multiply by factor of 2 to be conservative
        tol = 2 * np.arctan2(self.max_cs_len, self.average_radial_distance)

        # Compute lower and upper bounds of toroidal extent within tolerance
        lower_bound = 2 * np.pi - tol
        upper_bound = self._toroidal_extent + tol

        for coords in self.filament_coords:
            # Compute filament center of mass
            com = np.average(coords, axis=0)
            # Compute toroidal angle of each point in filament
            toroidal_angles = np.arctan2(coords[:, 1], coords[:, 0])
            # Ensure angles are positive
            toroidal_angles = (toroidal_angles + 2 * np.pi) % (2 * np.pi)
            # Compute bounds of toroidal extent of filament
            min_tor_ang = np.min(toroidal_angles)
            max_tor_ang = np.max(toroidal_angles)

            # Determine if filament toroidal extent overlaps with that of model
            if (min_tor_ang >= lower_bound or min_tor_ang <= upper_bound) or (
                max_tor_ang >= lower_bound or max_tor_ang <= upper_bound
            ):
                filtered_coords.append(coords)
                com_list.append(com)

        filtered_coords = np.array(filtered_coords)
        com_list = np.array(com_list)

        # Compute toroidal angles of filament centers of mass
        com_toroidal_angles = np.arctan2(com_list[:, 1], com_list[:, 0])
        com_toroidal_angles = (com_toroidal_angles + 2 * np.pi) % (2 * np.pi)

        # Sort filaments by toroidal angle and overwrite coordinate data
        self.filament_coords = np.array(
            [x for _, x in sorted(zip(com_toroidal_angles, filtered_coords))]
        )
        # Store filtered centers of mass since they have already been computed
        self.filament_com = np.array(
            [x for _, x in sorted(zip(com_toroidal_angles, com_list))]
        )

    def _cut_magnets(self):
        """Cuts the magnets at the planes defining the toriodal extent.
        (Internal function not intended to be called externally)
        """
        side_length = 1.25 * self.max_radial_distance

        toroidal_region = cq.Workplane("XZ")
        toroidal_region = toroidal_region.transformed(
            offset=(side_length / 2, 0)
        )
        toroidal_region = toroidal_region.rect(side_length, side_length)
        toroidal_region = toroidal_region.revolve(
            np.rad2deg(self._toroidal_extent),
            (-side_length / 2, 0),
            (-side_length / 2, 1),
        )
        toroidal_region = toroidal_region.val()

        for coil in self.magnet_coils:
            cut_coil = coil.solid.intersect(toroidal_region)
            coil.solid = cut_coil

    def build_magnet_coils(self):
        """Builds each filament in self.filtered_filaments in cubit, then cuts
        to the toroidal extent using self._cut_magnets().
        """
        self._logger.info("Constructing magnet coils...")

        self._extract_filament_data()
        self._compute_radial_distance_data()
        self._filter_filaments()

        self.magnet_coils = [
            MagnetCoil(
                coords,
                center_of_mass,
                self._width,
                self._thickness,
                self.sample_mod,
            )
            for coords, center_of_mass in zip(
                self.filament_coords, self.filament_com
            )
        ]

        [magnet_coil.create_magnet() for magnet_coil in self.magnet_coils]

        self._cut_magnets()

    def export_step(self, step_filename="magnet_set", export_dir=""):
        """Export CAD solids as a STEP file via CadQuery.

        Arguments:
            step_filename (str): name of STEP output file, excluding '.step'
                extension (optional, defaults to 'magnet_set').
            export_dir (str): directory to which to export the STEP output file
                (optional, defaults to empty string).
        """
        self._logger.info("Exporting STEP file for magnet coils...")

        self.export_dir = export_dir
        self.step_filename = step_filename

        export_path = Path(self.export_dir) / Path(
            self.step_filename
        ).with_suffix(".step")

        coil_set = cq.Compound.makeCompound(
            [coil.solid for coil in self.magnet_coils]
        )
        cq.exporters.export(coil_set, str(export_path))

    def mesh_magnets(self):
        """Creates tetrahedral mesh of magnet volumes via Coreform Cubit."""
        self._logger.info("Generating tetrahedral mesh of magnet coils...")

        last_vol_id = cubit_io.import_step_cubit(
            self.step_filename, self.export_dir
        )

        self.volume_ids = range(1, last_vol_id + 1)

        for vol in self.volume_ids:
            cubit.cmd(f"volume {vol} scheme tetmesh")
            cubit.cmd(f"mesh volume {vol}")

    def export_mesh(self, mesh_filename="magnet_mesh", export_dir=""):
        """Creates tetrahedral mesh of magnet volumes and exports H5M format
        via Coreform Cubit and  MOAB.

        Arguments:
            mesh_filename (str): name of H5M output file, excluding '.h5m'
                extension (optional, defaults to 'magnet_mesh').
            export_dir (str): directory to which to export the H5M output file
                (optional, defaults to empty string).
        """
        self._logger.info("Exporting mesh H5M file for magnet coils...")

        cubit_io.export_mesh_cubit(
            filename=mesh_filename, export_dir=export_dir
        )


class MagnetCoil(object):
    """An object representing a single modular stellarator magnet coil.

    Arguments:
        coords (2-D array of float): set of Cartesian coordinates defining
            magnet filament location.
        tangents (2-D array of float): set of tangent vectors at each filament
            location.
        center_of_mass (1-D array of float): Cartesian coordinates of filament
            center of mass.
        width (float): width of coil cross-section in toroidal direction [cm].
        thickness (float): thickness of coil cross-section in radial direction
            [cm].
    """

    def __init__(self, coords, center_of_mass, width, thickness, sample_mod):

        self.sample_mod = sample_mod
        self.coords = coords
        self.center_of_mass = center_of_mass
        self.width = width
        self.thickness = thickness

    @property
    def coords(self):
        return self._coords

    @coords.setter
    def coords(self, data):
        # Compute tangents
        tangents = np.subtract(
            np.append(data[1:], [data[1]], axis=0),
            np.append([data[-2]], data[0:-1], axis=0),
        )
        tangents = tangents / np.linalg.norm(tangents, axis=1)[:, np.newaxis]

        # Sample filament coordinates and tangents by modifier
        self._coords = data[0 : -1 : self.sample_mod]
        self._coords = np.append(self._coords, [self._coords[0]], axis=0)
        self.tangents = tangents[0 : -1 : self.sample_mod]
        self.tangents = np.append(self.tangents, [self.tangents[0]], axis=0)

    def create_magnet(self):
        """Creates a single magnet coil CAD solid in CadQuery.

        Returns:
            coil (object): cq.Solid object representing a single magnet coil.
        """
        tangent_vectors = [
            cq.Vector(tuple(tangent)) for tangent in self.tangents
        ]

        # Define coil filament path normals such that they face the filament
        # center of mass
        # Compute "outward" direction as difference between filament positions
        # and filament center of mass
        outward_dirs = self._coords - self.center_of_mass
        outward_dirs = (
            outward_dirs / np.linalg.norm(outward_dirs, axis=1)[:, np.newaxis]
        )

        # Project outward directions onto desired coil cross-section (CS) plane
        # at each filament position to define filament path normals
        parallel_parts = []
        for dir, tangent in zip(outward_dirs, self.tangents):
            parallel_parts.append(np.dot(dir, tangent))
        parallel_parts = np.array(parallel_parts)

        parallel_parts = np.diagonal(
            np.matmul(outward_dirs, self.tangents.transpose())
        )

        normals = outward_dirs - parallel_parts[:, np.newaxis] * self.tangents
        normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]

        # Compute binormals projected onto CS plane at each position
        binormals = np.cross(self.tangents, normals)

        # Compute coordinates of edges of rectangular coils
        edge_offsets = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]])

        coil_edge_coords = []
        for edge_offset in edge_offsets:
            coil_edge = (
                self._coords
                + edge_offset[0] * binormals * (self.width / 2)
                + edge_offset[1] * normals * (self.thickness / 2)
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
        self.solid = cq.Solid.makeSolid(shell)


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
