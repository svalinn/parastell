import argparse
from pathlib import Path

import math
import numpy as np
from scipy.interpolate import RegularGridInterpolator

import cubit
import cadquery as cq
import cad_to_dagmc
import pystell.read_vmec as read_vmec

from . import log
from . import cubit_io
from .utils import (
    normalize,
    expand_list,
    read_yaml_config,
    filter_kwargs,
    m2cm,
)

export_allowed_kwargs = ["export_cad_to_dagmc", "dagmc_filename"]


def orient_spline_surfaces(volume_id):
    """Extracts the inner and outer surface IDs for a given ParaStell in-vessel
    component volume in Coreform Cubit.

    Arguments:
        volume_id (int): Cubit volume ID.

    Returns:
        inner_surface_id (int): Cubit ID of in-vessel component inner surface.
        outer_surface_id (int): Cubit ID of in-vessel component outer surface.
    """

    surfaces = cubit.get_relatives("volume", volume_id, "surface")

    spline_surfaces = []
    for surface in surfaces:
        if cubit.get_surface_type(surface) == "spline surface":
            spline_surfaces.append(surface)

    if len(spline_surfaces) == 1:
        outer_surface_id = spline_surfaces[0]
        inner_surface_id = None
    else:
        # The outer surface bounding box will have the larger maximum XY value
        if (
            cubit.get_bounding_box("surface", spline_surfaces[1])[4]
            > cubit.get_bounding_box("surface", spline_surfaces[0])[4]
        ):
            outer_surface_id = spline_surfaces[1]
            inner_surface_id = spline_surfaces[0]
        else:
            outer_surface_id = spline_surfaces[0]
            inner_surface_id = spline_surfaces[1]

    return inner_surface_id, outer_surface_id


class InVesselBuild(object):
    """Parametrically models fusion stellarator in-vessel components using
    plasma equilibrium VMEC data and a user-defined radial build.

    Arguments:
        vmec_obj (object): plasma equilibrium VMEC object as defined by the
            PyStell-UW VMEC reader. Must have a method
            'vmec2xyz(s, theta, phi)' that returns an (x,y,z) coordinate for
            any closed flux surface label, s, poloidal angle, theta, and
            toroidal angle, phi.
        radial_build (object): RadialBuild class object with all attributes
            defined.
        logger (object): logger object (optional, defaults to None). If no
            logger is supplied, a default logger will be instantiated.

    Optional attributes:
        toroidal_grid_size (int): number of toroidal angle grid points defining
            point clouds over which spline surfaces are fit. If this value is
            greater than the length of "toroidal_angles", additional grid
            points are set at toroidal angles interpolated between those
            specified in "toroidal_angles".
        poloidal_grid_size (int): number of poloidal angle grid points defining
            point clouds over which spline surfaces are fit. If this value is
            greater than the length of "poloidal_angles", additional grid
            points are set at poloidal angles interpolated between those
            specified in "poloidal_angles".
        transpose_fit (bool): flag to indicate whether the 2-D iterable of
            points through which each component's spline surface is fit should
            be transposed. Can sometimes fix errant CAD generation.
        scale (float): a scaling factor between the units of VMEC and [cm]
            (defaults to m2cm = 100).
    """

    def __init__(self, vmec_obj, radial_build, logger=None, **kwargs):

        self.logger = logger
        self.vmec_obj = vmec_obj
        self.radial_build = radial_build

        self.toroidal_grid_size = 61
        self.poloidal_grid_size = 61
        self.transpose_fit = False
        self.scale = m2cm

        for name in kwargs.keys() & (
            "toroidal_grid_size",
            "poloidal_grid_size",
            "transpose_fit",
            "scale",
        ):
            self.__setattr__(name, kwargs[name])

        self.Surfaces = {}
        self.Components = {}

    @property
    def vmec_obj(self):
        return self._vmec_obj

    @vmec_obj.setter
    def vmec_obj(self, vmec_object):
        self._vmec_obj = vmec_object

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, logger_object):
        self._logger = log.check_init(logger_object)

    def _interpolate_offset_matrix(self, offset_mat):
        """Interpolates total offset for expanded angle lists using cubic spline
        interpolation.
        (Internal function not intended to be called externally)

        Returns:
            interpolated_offset_mat (np.ndarray(double)): expanded matrix
                including interpolated offset values at additional rows and
                columns [cm].
        """
        interpolator = RegularGridInterpolator(
            (
                self.radial_build.toroidal_angles,
                self.radial_build.poloidal_angles,
            ),
            offset_mat,
            method="pchip",
        )

        interpolated_offset_mat = np.array(
            [
                [
                    interpolator([np.rad2deg(phi), np.rad2deg(theta)])[0]
                    for theta in self._poloidal_angles_exp
                ]
                for phi in self._toroidal_angles_exp
            ]
        )

        return interpolated_offset_mat

    def populate_surfaces(self):
        """Populates Surface class objects representing the outer surface of
        each component specified in the radial build.
        """
        self._logger.info(
            "Populating surface objects for in-vessel components..."
        )

        self._toroidal_angles_exp = np.deg2rad(
            expand_list(
                self.radial_build.toroidal_angles, self.toroidal_grid_size
            )
        )
        self._poloidal_angles_exp = np.deg2rad(
            expand_list(
                self.radial_build.poloidal_angles, self.poloidal_grid_size
            )
        )

        offset_mat = np.zeros(
            (
                len(self.radial_build.toroidal_angles),
                len(self.radial_build.poloidal_angles),
            )
        )

        for name, layer_data in self.radial_build.radial_build.items():
            if name == "plasma":
                s = 1.0
            else:
                s = self.radial_build.wall_s

            offset_mat += np.array(layer_data["thickness_matrix"])
            interpolated_offset_mat = self._interpolate_offset_matrix(
                offset_mat
            )

            self.Surfaces[name] = Surface(
                self._vmec_obj,
                s,
                self._poloidal_angles_exp,
                self._toroidal_angles_exp,
                interpolated_offset_mat,
                self.transpose_fit,
                self.scale,
            )

    def calculate_loci(self):
        """Calls calculate_loci method in Surface class for each component
        specified in the radial build.
        """
        self._logger.info("Computing point cloud for in-vessel components...")

        [surface.calculate_loci() for surface in self.Surfaces.values()]

    def _check_edge(self, edge):
        """Checks a CadQuery edge to determine if the its center of mass is
        located at either end of the stellarator build.
        (Internal function not intended to be called externally)

        Arguments:
            edge (object): CadQuery.Edge object.

        Returns:
            (bool): flag to indicate whether edge is located at either end of
                stellarator build.
        """
        x = edge.Center().x
        y = edge.Center().y

        toroidal_angle = np.arctan2(y, x)
        """if self.radial_build.toroidal_angles[0] >= 0.0:
            toroidal_angle = (toroidal_angle + 2 * np.pi) % (2 * np.pi)"""
        toroidal_angle = np.rad2deg(toroidal_angle)

        toroidal_extent = [
            self.radial_build.toroidal_angles[0],
            self.radial_build.toroidal_angles[-1],
        ]

        return np.any(np.isclose(toroidal_extent, toroidal_angle))

    def generate_components(self):
        """Constructs a CAD solid for each component specified in the radial
        build by combining each component's outer surface with that of the
        component interior to it.
        """
        self._logger.info(
            "Constructing CadQuery objects for in-vessel components..."
        )

        inner_surface = None

        for name, surface in self.Surfaces.items():
            outer_surface = surface.generate_surface()
            edge_list = [
                edge
                for edge in outer_surface.Edges()
                if self._check_edge(edge)
            ]
            wire_list = [cq.Wire.assembleEdges([edge]) for edge in edge_list]

            if inner_surface:
                face_list = [
                    cq.Face.makeFromWires(wire, innerWires=[inner_wire])
                    for wire, inner_wire in zip(wire_list, inner_wire_list)
                ]
                face_list.append(inner_surface)

            else:
                face_list = [cq.Face.makeFromWires(wire) for wire in wire_list]

            face_list.append(outer_surface)
            shell = cq.Shell.makeShell(face_list)
            solid = cq.Solid.makeSolid(shell)

            self.Components[name] = solid

            inner_wire_list = wire_list
            inner_surface = outer_surface

    def get_loci(self):
        """Returns the set of point-loci defining the outer surfaces of the
        components specified in the radial build.
        """
        return np.array(
            [surface.get_loci() for surface in self.Surfaces.values()]
        )

    def merge_layer_surfaces(self):
        """Merges ParaStell in-vessel component surfaces in Coreform Cubit
        based on surface IDs rather than imprinting and merging all. Assumes
        that the radial_build dictionary is ordered radially outward. Note that
        overlaps between magnet volumes and in-vessel components will not be
        merged in this workflow.
        """
        # Tracks the surface id of the outer surface of the previous layer
        prev_outer_surface_id = None

        for data in self.radial_build.radial_build.values():

            inner_surface_id, outer_surface_id = orient_spline_surfaces(
                data["vol_id"]
            )

            # Conditionally skip merging (first iteration only)
            if prev_outer_surface_id is None:
                prev_outer_surface_id = outer_surface_id
            else:
                cubit.cmd(
                    f"merge surface {inner_surface_id} {prev_outer_surface_id}"
                )
                prev_outer_surface_id = outer_surface_id

    def import_step_cubit(self):
        """Imports STEP files from in-vessel build into Coreform Cubit."""
        for name, data in self.radial_build.radial_build.items():
            vol_id = cubit_io.import_step_cubit(name, self.export_dir)
            data["vol_id"] = vol_id

    def export_step(self, export_dir=""):
        """Export CAD solids as STEP files via CadQuery.

        Arguments:
            export_dir (str): directory to which to export the STEP output files
                (optional, defaults to empty string).
        """
        self._logger.info("Exporting STEP files for in-vessel components...")

        self.export_dir = export_dir

        for name, component in self.Components.items():
            export_path = Path(self.export_dir) / Path(name).with_suffix(
                ".step"
            )
            cq.exporters.export(component, str(export_path))

    def export_cad_to_dagmc(
        self,
        min_mesh_size=10,
        max_mesh_size=50,
        dagmc_filename="dagmc",
        export_dir="",
    ):
        """Exports DAGMC neutronics H5M file of ParaStell in-vessel components
        via CAD-to-DAGMC.

        Arguments:
            min_mesh_size (float): minimum size of mesh elements [cm] (defaults
                to 1.0).
            max_mesh_size (float): maximum size of mesh elements [cm] (defaults
                to 5.0).
            dagmc_filename (str): name of DAGMC output file, excluding '.h5m'
                extension (optional, defaults to 'dagmc').
            export_dir (str): directory to which to export the DAGMC output file
                (optional, defaults to empty string).
        """
        self._logger.info(
            "Exporting DAGMC neutronics model of in-vessel components..."
        )

        model = cad_to_dagmc.CadToDagmc()

        for name, component in self.Components.items():
            model.add_cadquery_object(
                component,
                material_tags=[
                    self.radial_build.radial_build[name]["mat_tag"]
                ],
            )

        export_path = Path(export_dir) / Path(dagmc_filename).with_suffix(
            ".h5m"
        )

        model.export_dagmc_h5m_file(
            filename=str(export_path),
            min_mesh_size=min_mesh_size,
            max_mesh_size=max_mesh_size,
        )


class Surface(object):
    """An object representing a surface formed by approximating a spline
    surface through a set of points defined by a toroidal angle, poloidal angle
    grid and offset from a reference surface.

    Arguments:
        vmec_obj (object): plasma equilibrium VMEC object as defined by the
            PyStell-UW VMEC reader. Must have a method
            'vmec2xyz(s, theta, phi)' that returns an (x,y,z) coordinate for
            any closed flux surface label, s, poloidal angle, theta, and
            toroidal angle, phi.
        s (float): the normalized closed flux surface label defining the point
            of reference for offset.
        poloidal_angles (array of float): poloidal angles adefining grid through
            which the spline surface is approximated [deg]. This array should
            always span exactly 360 degrees.
        toroidal_angles (array of float): toroidal angles defining grid through
            which the spline surface is approximated [deg]. This array should
            never exceed 360 degrees.
        offset_mat (np.array(double)): the set of offsets from the surface
            defined by s for each toroidal angle, poloidal angle pair on the
            surface [cm].
        transpose_fit (bool): flag to indicate whether the 2-D iterable of
            points through which the spline surface is fit should be
            transposed. Can sometimes fix errant CAD generation.
        scale (float): a scaling factor between the units of VMEC and [cm].
    """

    def __init__(
        self,
        vmec_obj,
        s,
        poloidal_angles,
        toroidal_angles,
        offset_mat,
        transpose_fit,
        scale,
    ):

        self.vmec_obj = vmec_obj
        self.s = s
        self.poloidal_angles = poloidal_angles
        self.toroidal_angles = toroidal_angles
        self.offset_mat = offset_mat
        self.transpose_fit = transpose_fit
        self.scale = scale

        self.surface = None

    def _vmec2xyz(self, poloidal_offset=0):
        """Return an N x 3 NumPy array containing the Cartesian coordinates of
        the points at this toroidal angle and N different poloidal angles, each
        offset slightly.
        (Internal function not intended to be called externally)

        Arguments:
            poloidal_offset (float) : some offset to apply to the full set of
                poloidal angles for evaluating the location of the Cartesian
                points (optional, defaults to 0).
        """
        return self.scale * np.array(
            [
                [
                    self.vmec_obj.vmec2xyz(self.s, theta, phi)
                    for theta in (self.poloidal_angles + poloidal_offset)
                ]
                for phi in self.toroidal_angles
            ]
        )

    def _normals(self):
        """Approximate the normal to the curve at each poloidal angle by first
        approximating the tangent to the curve and then taking the
        cross-product of that tangent with a vector defined as normal to the
        plane at this toroidal angle.
        (Internal function not intended to be called externally)
        """
        eps = 1e-4
        perturbed_loci = self._vmec2xyz(eps)

        tangents = perturbed_loci - self.surface_loci

        binormals = np.array(
            [
                -np.sin(self.toroidal_angles),
                np.cos(self.toroidal_angles),
                np.zeros(len(self.toroidal_angles)),
            ]
        ).T

        normals = np.cross(binormals[:, :, np.newaxis], tangents, axisa=1)

        return normalize(normals)

    def calculate_loci(self):
        """Generates Cartesian point-loci for stellarator surface."""
        self.surface_loci = self._vmec2xyz()

        if not np.all(self.offset_mat == 0):
            self.surface_loci += (
                self.offset_mat[:, :, np.newaxis] * self._normals()
            )

    def generate_surface(self):
        """Constructs a surface by approximating a spline surface through the
        set of points computed by calculate_loci.
        """
        if not self.surface:
            if self.transpose_fit:
                self.surface_loci = np.transpose(self.surface_loci, (1, 0, 2))

            vectors = [
                [cq.Vector(tuple(pt)) for pt in grid_row]
                for grid_row in self.surface_loci
            ]

            self.surface = cq.Face.makeSplineApprox(
                vectors,
                tol=1e-2,
                minDeg=1,
                maxDeg=6,
                smoothing=(1, 1, 1),
            )

        return self.surface.clean()


class RadialBuild(object):
    """Parametrically defines ParaStell in-vessel component geometries.
    In-vessel component thicknesses are defined on a grid of toroidal and
    poloidal angles, and the first wall profile is defined by a closed flux
    surface extrapolation.

    Arguments:
        toroidal_angles (array of float): toroidal angles at which radial build
            is specified [deg]. This array should never exceed 360 degrees.
        poloidal_angles (array of float): poloidal angles adefining grid through
            which the spline surface is approximated [deg]. This array should
            always span exactly 360 degrees.
        wall_s (float): closed flux surface label extrapolation at wall.
        radial_build (dict): dictionary representing the three-dimensional
            radial build of in-vessel components, including
            {
                'component': {
                    'thickness_matrix': 2-D matrix defining component
                        thickness at (toroidal angle, poloidal angle)
                        locations. Rows represent toroidal angles, columns
                        represent poloidal angles, and each must be in the same
                        order provided in toroidal_angles and poloidal_angles
                        [cm](ndarray(float)).
                    'mat_tag': DAGMC material tag for component in DAGMC
                        neutronics model (str, optional, defaults to None). If
                        none is supplied, the 'component' key will be used.
                }
            }.
        split_chamber (bool): if wall_s > 1.0, separate interior vacuum
            chamber into plasma and scrape-off layer components (optional,
            defaults to False). If an item with a 'sol' key is present in the
            radial_build dictionary, settting this to False will not combine
            the resultant 'chamber' with 'sol'. To include a custom scrape-off
            layer definition for 'chamber', add an item with a 'chamber' key
            and desired 'thickness_matrix' value to the radial_build dictionary.
        logger (object): logger object (optional, defaults to None). If no
            logger is supplied, a default logger will be instantiated.

    Optional attributes:
        plasma_mat_tag (str): DAGMC material tag to use for plasma if
            split_chamber is True (defaults to 'Vacuum').
        sol_mat_tag (str): DAGMC material tag to use for scrape-off layer if
            split_chamber is True (defaults to 'Vacuum').
        chamber_mat_tag (str): DAGMC material tag to use for interior vacuum
            chamber if split_chamber is False (defaults to 'Vacuum).
    """

    def __init__(
        self,
        toroidal_angles,
        poloidal_angles,
        wall_s,
        radial_build,
        split_chamber=False,
        logger=None,
        **kwargs,
    ):

        self.logger = logger
        self.toroidal_angles = toroidal_angles
        self.poloidal_angles = poloidal_angles
        self.wall_s = wall_s
        self.radial_build = radial_build
        self.split_chamber = split_chamber

        for name in kwargs.keys() & (
            "plasma_mat_tag",
            "sol_mat_tag",
            "chamber_mat_tag",
        ):
            self.__setattr__(name, kwargs[name])

        self._logger.info("Constructing radial build...")

    @property
    def toroidal_angles(self):
        return self._toroidal_angles

    @toroidal_angles.setter
    def toroidal_angles(self, angle_list):
        if hasattr(self, "toroidal_angles"):
            e = AttributeError(
                '"toroidal_angles" cannot be set after class initialization. '
                "Please create new class instance to alter this attribute."
            )
            self._logger.error(e.args[0])
            raise e

        if angle_list[-1] - angle_list[0] > 360.0:
            e = ValueError("Toroidal extent cannot exceed 360.0 degrees.")
            self._logger.error(e.args[0])
            raise e

        self._toroidal_angles = angle_list

    @property
    def poloidal_angles(self):
        return self._poloidal_angles

    @poloidal_angles.setter
    def poloidal_angles(self, angle_list):
        if hasattr(self, "poloidal_angles"):
            e = AttributeError(
                '"poloidal_angles" cannot be set after class initialization. '
                "Please create new class instance to alter this attribute."
            )
            self._logger.error(e.args[0])
            raise e

        if angle_list[-1] - angle_list[0] != 360.0:
            e = AssertionError(
                "Poloidal extent must span exactly 360.0 degrees."
            )
            self._logger.error(e.args[0])
            raise e

        self._poloidal_angles = angle_list

    @property
    def wall_s(self):
        return self._wall_s

    @wall_s.setter
    def wall_s(self, s):
        if hasattr(self, "wall_s"):
            e = AttributeError(
                '"wall_s" cannot be set after class initialization. Please '
                "create new class instance to alter this attribute."
            )
            self._logger.error(e.args[0])
            raise e

        self._wall_s = s
        if self._wall_s < 1.0:
            e = ValueError("wall_s must be greater than or equal to 1.0.")
            self._logger.error(e.args[0])
            raise e

    @property
    def radial_build(self):
        return self._radial_build

    @radial_build.setter
    def radial_build(self, build_dict):
        self._radial_build = build_dict

        for name, component in self._radial_build.items():
            component["thickness_matrix"] = np.array(
                component["thickness_matrix"]
            )
            if component["thickness_matrix"].shape != (
                len(self._toroidal_angles),
                len(self._poloidal_angles),
            ):
                e = AssertionError(
                    f"The dimensions of {name}'s thickness matrix "
                    f'{component["thickness_matrix"].shape} must match the '
                    "dimensions defined by the toroidal and poloidal angle "
                    "lists "
                    f"{len(self._toroidal_angles),len(self._poloidal_angles)}, "
                    "which define the rows and columns of the matrix, "
                    "respectively."
                )
                self._logger.error(e.args[0])
                raise e

            if np.any(component["thickness_matrix"] < 0):
                e = ValueError(
                    "Component thicknesses must be greater than or equal to 0. "
                    "Check thickness inputs for negative values."
                )
                self._logger.error(e.args[0])
                raise e

            if "mat_tag" not in component:
                self._set_mat_tag(name, name)

    @property
    def split_chamber(self):
        return self._split_chamber

    @split_chamber.setter
    def split_chamber(self, value):
        if hasattr(self, "split_chamber"):
            e = AttributeError(
                '"split_chamber" cannot be set after class initialization. '
                "Please create new class instance to alter this attribute."
            )
            self._logger.error(e.args[0])
            raise e

        self._split_chamber = value

        if self._split_chamber:
            if self._wall_s > 1.0 and "sol" not in self._radial_build:
                self.radial_build = {
                    "sol": {
                        "thickness_matrix": np.zeros(
                            (
                                len(self._toroidal_angles),
                                len(self._poloidal_angles),
                            )
                        )
                    },
                    **self.radial_build,
                }
                if not hasattr(self, "sol_mat_tag"):
                    self.sol_mat_tag = "Vacuum"

            inner_volume_name = "plasma"
            inner_volume_tag = "plasma_mat_tag"
        else:
            inner_volume_name = "chamber"
            inner_volume_tag = "chamber_mat_tag"

        self.radial_build = {
            inner_volume_name: {
                "thickness_matrix": np.zeros(
                    (len(self._toroidal_angles), len(self._poloidal_angles))
                )
            },
            **self.radial_build,
        }
        if not hasattr(self, inner_volume_tag):
            self.__setattr__(inner_volume_tag, "Vacuum")

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, logger_object):
        self._logger = log.check_init(logger_object)

    @property
    def plasma_mat_tag(self):
        return self._plasma_mat_tag

    @plasma_mat_tag.setter
    def plasma_mat_tag(self, mat_tag):
        self._plasma_mat_tag = mat_tag
        self._set_mat_tag("plasma", self._plasma_mat_tag)

    @property
    def sol_mat_tag(self):
        return self._sol_mat_tag

    @sol_mat_tag.setter
    def sol_mat_tag(self, mat_tag):
        self._sol_mat_tag = mat_tag
        self._set_mat_tag("sol", self._sol_mat_tag)

    @property
    def chamber_mat_tag(self):
        return self._chamber_mat_tag

    @chamber_mat_tag.setter
    def chamber_mat_tag(self, mat_tag):
        self._chamber_mat_tag = mat_tag
        self._set_mat_tag("chamber", self._chamber_mat_tag)

    def _set_mat_tag(self, name, mat_tag):
        """Sets DAGMC material tag for a given component.
        (Internal function not intended to be called externally)

        Arguments:
            name (str): name of component.
            mat_tag (str): DAGMC material tag.
        """
        self.radial_build[name]["mat_tag"] = mat_tag


def parse_args():
    """Parser for running as a script."""
    parser = argparse.ArgumentParser(prog="invessel_build")

    parser.add_argument(
        "filename",
        help="YAML file defining ParaStell in-vessel component configuration",
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


def generate_invessel_build():
    """Main method when run as a command line script."""
    args = parse_args()

    all_data = read_yaml_config(args.filename)

    if args.logger == True:
        logger = log.init()
    else:
        logger = log.NullLogger()

    vmec_file = all_data["vmec_file"]
    vmec_obj = read_vmec.VMECData(vmec_file)

    invessel_build_dict = all_data["invessel_build"]

    radial_build = RadialBuild(
        invessel_build_dict["toroidal_angles"],
        invessel_build_dict["poloidal_angles"],
        invessel_build_dict["wall_s"],
        invessel_build_dict["radial_build"],
        logger=logger,
        **invessel_build_dict,
    )

    invessel_build = InVesselBuild(
        vmec_obj, radial_build, logger=logger, **invessel_build_dict
    )

    invessel_build.populate_surfaces()
    invessel_build.calculate_loci()
    invessel_build.generate_components()

    invessel_build.export_step(export_dir=args.export_dir)

    if invessel_build_dict["export_cad_to_dagmc"]:

        invessel_build.export_cad_to_dagmc(
            export_dir=args.export_dir,
            **(filter_kwargs(invessel_build_dict, ["dagmc_filename"])),
        )


if __name__ == "__main__":
    generate_invessel_build()
