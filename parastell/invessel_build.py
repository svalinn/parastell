import argparse
from pathlib import Path
from abc import ABC

import numpy as np
from scipy.interpolate import (
    RegularGridInterpolator,
    CloughTocher2DInterpolator,
)


import cadquery as cq
import pydagmc
from pymoab import core, types
import gmsh

from . import log
from .cubit_utils import (
    create_new_cubit_instance,
    import_step_cubit,
    export_mesh_cubit,
    orient_spline_surfaces,
    merge_surfaces,
    mesh_volume_auto_factor,
    mesh_surface_coarse_trimesh,
)
from .utils import (
    ToroidalMesh,
    normalize,
    expand_list,
    read_yaml_config,
    create_vol_mesh_from_surf_mesh,
    m2cm,
)
from .pystell import read_vmec


def create_moab_tris_from_verts(corners, mbc, reverse=False):
    """Create 2 moab triangle elements from a list of 4 pymoab verts.

    Arguments:
        corners (4x3 numpy array): list of 4 (x,y,z) points. Connecting the
            points in the order given should result in a polygon
        mbc (pymoab core): pymoab core instance to create elements with.

    Returns:
        list of two pymoab MBTRI elements
    """
    if reverse:
        tri_1 = mbc.create_element(
            types.MBTRI, [corners[0], corners[1], corners[2]]
        )
        tri_2 = mbc.create_element(
            types.MBTRI, [corners[0], corners[2], corners[3]]
        )
    else:
        tri_1 = mbc.create_element(
            types.MBTRI, [corners[2], corners[1], corners[0]]
        )
        tri_2 = mbc.create_element(
            types.MBTRI, [corners[3], corners[2], corners[0]]
        )

    return [tri_1, tri_2]


class ReferenceSurface(ABC):
    """An object representing the innermost surface from which subsequent
    layers are built.
    """

    def __init__():
        pass

    def angles_to_xyz(self, toroidal_angle, poloidal_angles, s, scale):
        """Method to go from a location defined by two angles and some
        constant to x, y, z coordinates.

        Arguments:
            toroidal_angles (float): Toroidal angle at which to
                evaluate cartesian coordinates. Measured in radians.
            poloidal_angles (iterable of float): Poloidal angles at which to
                evaluate cartesian coordinates. Measured in radians.
            s (float): Generic parameter which may affect the evaluation of
                the cartesian coordinate at a given angle pair.
            scale (float): a scaling factor between input and output data.

        Returns:
            coords (numpy array): Nx3 array of Cartesian coordinates at each
                angle pair specified.
        """
        pass


class VMECSurface(ReferenceSurface):
    """An object that uses VMEC data to represent the innermost surface
    of an in vessel build.

    Arguments:
        vmec_obj (object): plasma equilibrium VMEC object as defined by the
            PyStell-UW VMEC reader. Must have a method
            'vmec2xyz(s, theta, phi)' that returns an (x,y,z) coordinate for
            any closed flux surface label, s, poloidal angle, theta, and
            toroidal angle, phi.
    """

    def __init__(self, vmec_obj):
        self.vmec_obj = vmec_obj

    def angles_to_xyz(self, toroidal_angle, poloidal_angles, s, scale):
        """Evaluate the cartesian coordinates for a set of toroidal and
        poloidal angles and flux surface label.

        Arguments:
            toroidal_angles (float): Toroidal angle at which to
                    evaluate cartesian coordinates. Measured in radians.
            poloidal_angles (iterable of float): Poloidal angles at which to
                    evaluate cartesian coordinates. Measured in radians.
            s (float): the normalized closed flux surface label defining the
                point of reference for offset.
            scale (float): a scaling factor between input and output data.

        Returns:
            coords (numpy array): Nx3 array of Cartesian coordinates at each
                poloidal angle specified.
        """
        coords = []
        for poloidal_angle in poloidal_angles:
            x, y, z = self.vmec_obj.vmec2xyz(s, poloidal_angle, toroidal_angle)
            coords.append([x, y, z])
        return np.array(coords) * scale


class RibBasedSurface(ReferenceSurface):
    """An object that uses closed loops of R, Z points (ribs) on planes of
    constant toroidal angle to approximate the first wall surface of an in-
    vessel build. This class must be used with split_chamber = False.

    Arguments:
        rib_data (numpy array): NxMx2 array of of R, Z points. The first
            dimension corresponds to the plane of constant toroidal angle on
            which the closed loop of points lies. The second dimension is the
            location on the closed loop at which the point lies, and the third
            dimension is the R, Z values of that point. ParaStell expects the
            following from this data set:
            - The data spans exactly one field period
            - The coordinates of each toroidal slice (rib) precess counter-
              clockwise
            - The coordinates obey helical (stellarator) symmetry, i.e.,
                - The (R,Z) coordinates of the first and final ribs are exactly
                  equal
                - The (R,Z) coordinates of the first, toroidal midplane, and
                  final ribs are symmetric about the axial midplane
                - The (R,Z) coordinates each half-period are a helical
                  reflection of the other half-period
        toroidal_angles (iterable of float): List of toroidal angles
            corresponding to the first dimension of rib_data. Measured in
            degrees.
        poloidal_angles (iterable of float): List of poloidal angles
            corresponding to the second dimension of rib_data. Measured in
            degrees. Should start at 0 degrees and end at 360 degrees.
    """

    def __init__(self, rib_data, toroidal_angles, poloidal_angles):
        self.rib_data = rib_data
        self.toroidal_angles = toroidal_angles
        self.poloidal_angles = poloidal_angles
        self.build_analytic_surface()

    def _extract_rib_data(self, ribs, toroidal_angles, poloidal_angles):
        """Internal function, not intended for use externally. Updates
        member variables that track R, Z values corresponding to
        angle pairs for use when building the interpolators.

        Arguments:
            ribs (np array): NxMx2 array of of R, Z points. The first
                dimension corresponds to the plane of constant toroidal angle
                on which the closed loop of points lies. The second dimension
                is the location on the closed loop at which the point lies, and
                the third dimension is the R, Z values of that point.
            toroidal_angles (iterable of float): List of toroidal angles
                corresponding to the first dimension of rib_data. Measured in
                degrees.
            poloidal_angles (iterable of float): List of poloidal angles
                corresponding to the second dimension of rib_data. Measured in
                degrees.
        """
        for phi, rib in zip(toroidal_angles, ribs):
            for theta, rib_locus in zip(poloidal_angles, rib):
                self.r_data.append(rib_locus[0])
                self.z_data.append(rib_locus[1])
                self.grid_points.append([phi, theta])

    def build_analytic_surface(self):
        """Build interpolators for R, Z coordinates using provided
        rib_data, toroidal_angles, and poloidal_angles. Adds copies of the data
        shifted by one period ahead of and behind provided data in the toroidal
        and poloidal directions to preserve periodicity.
        """
        self.r_data = []
        self.z_data = []
        self.grid_points = []

        # Toroidal Periodicity Before Period
        toroidal_shift = -max(self.toroidal_angles)
        shifted_toroidal_angles = self.toroidal_angles[0:-1] + toroidal_shift
        rib_subset = self.rib_data[0:-1]
        self._extract_rib_data(
            rib_subset, shifted_toroidal_angles, self.poloidal_angles
        )

        # Poloidal Periodicity Before Period
        poloidal_shift = -max(self.poloidal_angles)
        shifted_poloidal_angles = self.poloidal_angles[0:-1] - poloidal_shift
        rib_subset = self.rib_data[:, 0:-1, :]
        self._extract_rib_data(
            rib_subset, self.toroidal_angles, shifted_poloidal_angles
        )

        # Provided data
        self._extract_rib_data(
            self.rib_data,
            self.toroidal_angles,
            self.poloidal_angles,
        )

        # Toroidal Periodicity After Period
        toroidal_shift = max(self.toroidal_angles)
        shifted_toroidal_angles = self.toroidal_angles[1:] + toroidal_shift
        rib_subset = self.rib_data[1:]
        self._extract_rib_data(
            rib_subset, shifted_toroidal_angles, self.poloidal_angles
        )

        # Poloidal Periodicity After Period
        poloidal_shift = max(self.poloidal_angles)
        shifted_poloidal_angles = self.poloidal_angles[1:] + poloidal_shift
        rib_subset = self.rib_data[:, 1:, :]
        self._extract_rib_data(
            rib_subset, self.toroidal_angles, shifted_poloidal_angles
        )

        self.r_interp = CloughTocher2DInterpolator(
            self.grid_points, self.r_data
        )
        self.z_interp = CloughTocher2DInterpolator(
            self.grid_points, self.z_data
        )

    def angles_to_xyz(self, toroidal_angle, poloidal_angles, s, scale):
        """Return the cartesian coordinates from the interpolators for a
        toroidal angle and a set of poloidal angles. Takes s as a argument for
        compatibility, but does nothing with it.

        Arguments:
            toroidal_angles (float): Toroidal angle at which to
                    evaluate cartesian coordinates. Measured in radians.
            poloidal_angles (iterable of float): Poloidal angles at which to
                    evaluate cartesian coordinates. Measured in radians.
            s (float): Not used.
            scale (float): a scaling factor between input and output data.

        Returns:
            coords (numpy array): Nx3 array of Cartesian coordinates at each
                angle pair specified.
        """
        coords = []
        toroidal_angle = np.rad2deg(toroidal_angle)
        poloidal_angles = np.rad2deg(poloidal_angles)
        for poloidal_angle in poloidal_angles:
            r = self.r_interp(toroidal_angle, poloidal_angle)
            z = self.z_interp(toroidal_angle, poloidal_angle)
            x = r * np.cos(np.deg2rad(toroidal_angle))
            y = r * np.sin(np.deg2rad(toroidal_angle))
            coord = np.array([x, y, z])
            coords.append(coord)

        return np.array(coords) * scale


class InVesselBuild(object):
    """Parametrically models fusion stellarator in-vessel components using
    plasma equilibrium VMEC data and a user-defined radial build.

    Arguments:
        ref_surf (object): ReferenceSurface object. Must have a method
            'angles_to_xyz(toroidal_angles, poloidal_angles, s, scale)' that
            returns an Nx3 numpy array of cartesian coordinates for any closed
            flux surface label, s, poloidal angle (theta), and toroidal angle
            (phi).
        radial_build (object): RadialBuild class object with all attributes
            defined.
        logger (object): logger object (defaults to None). If no logger is
            supplied, a default logger will be instantiated.

    Optional attributes:
        repeat (int): number of times to repeat build segment for full model
            (defaults to 0).
        num_ribs (int): total number of ribs over which to loft for each build
            segment (defaults to 61). Ribs are set at toroidal angles
            interpolated between those specified in 'toroidal_angles' if this
            value is greater than the number of entries in 'toroidal_angles'.
        num_rib_pts (int): total number of points defining each rib spline
            (defaults to 67). Points are set at poloidal angles interpolated
            between those specified in 'poloidal_angles' if this value is
            greater than the number of entries in 'poloidal_angles'.
        scale (float): a scaling factor between input and output data
            (defaults to m2cm = 100).
        use_pydagmc (bool): If True, generate components with pydagmc, rather
            than CadQuery (defaults to False).
    """

    def __init__(self, ref_surf, radial_build, logger=None, **kwargs):

        self.logger = logger
        self.ref_surf = ref_surf
        self.radial_build = radial_build

        self.repeat = 0
        self.num_ribs = 61
        self.num_rib_pts = 61
        self.scale = m2cm
        self.use_pydagmc = False

        if "scale" not in kwargs.keys():
            w = Warning(
                "No factor specified to scale InVesselBuild input data. "
                "Assuming a scaling factor of 100.0, which is consistent with "
                "input being in units of [m] and desired output in units of "
                "[cm]."
            )
            self._logger.warning(w.args[0])

        for name in kwargs.keys() & (
            "repeat",
            "num_ribs",
            "num_rib_pts",
            "scale",
            "use_pydagmc",
        ):
            self.__setattr__(name, kwargs[name])

        self.Surfaces = {}
        self.Components = {}

    @property
    def ref_surf(self):
        return self._ref_surf

    @ref_surf.setter
    def ref_surf(self, ref_surf):
        self._ref_surf = ref_surf

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, logger_object):
        self._logger = log.check_init(logger_object)

    @property
    def repeat(self):
        return self._repeat

    @repeat.setter
    def repeat(self, num):
        self._repeat = num
        if (self._repeat + 1) * self.radial_build.toroidal_angles[-1] > 360.0:
            e = AssertionError(
                "Total toroidal extent requested with repeated geometry "
                'exceeds 360 degrees. Please examine the "repeat" parameter '
                'and the "toroidal_angles" parameter of "radial_build".'
            )
            self._logger.error(e.args[0])
            raise e

    @property
    def use_pydagmc(self):
        return self._use_pydagmc

    @use_pydagmc.setter
    def use_pydagmc(self, value):
        self._use_pydagmc = value
        if self._use_pydagmc:
            self.mbc = core.Core()
            self.dag_model = pydagmc.Model(self.mbc)

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
            method="linear" if self.use_pydagmc else "pchip",
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
            expand_list(self.radial_build.toroidal_angles, self.num_ribs)
        )
        self._poloidal_angles_exp = np.deg2rad(
            expand_list(self.radial_build.poloidal_angles, self.num_rib_pts)
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
                self._ref_surf,
                s,
                self._poloidal_angles_exp,
                self._toroidal_angles_exp,
                interpolated_offset_mat,
                self.scale,
            )

        [surface.populate_ribs() for surface in self.Surfaces.values()]

    def calculate_loci(self):
        """Calls calculate_loci method in Surface class for each component
        specified in the radial build.
        """
        self._logger.info("Computing point cloud for in-vessel components...")

        [surface.calculate_loci() for surface in self.Surfaces.values()]

    def generate_components(self):
        if self.use_pydagmc:
            self.generate_components_pydagmc()
        else:
            self.generate_components_cadquery()

    def generate_components_cadquery(self):
        """Constructs a CAD solid for each component specified in the radial
        build by cutting the interior surface solid from the outer surface
        solid for a given component.
        """
        self._logger.info(
            "Constructing CadQuery objects for in-vessel components..."
        )

        interior_surface = None

        segment_angles = np.linspace(
            self.radial_build.toroidal_angles[-1],
            self._repeat * self.radial_build.toroidal_angles[-1],
            num=self._repeat,
        )

        for name, surface in self.Surfaces.items():
            outer_surface = surface.generate_surface()

            if interior_surface is not None:
                segment = outer_surface.cut(interior_surface)
            else:
                segment = outer_surface

            component = segment

            for angle in segment_angles:
                rot_segment = segment.rotate((0, 0, 0), (0, 0, 1), angle)
                component = component.fuse(rot_segment)

            self.Components[name] = component
            interior_surface = outer_surface

    def _connect_ribs_with_tris_moab(self, rib1, rib2, reverse=False):
        """Creat MBTRI elements add add them to a surface between two ribs.
        (Internal function not intended to be called externally)

        Arguments:
            rib1 (Rib object): First of two ribs to be connected.
            rib2 (Rib object): Second of two ribs to be connected.
            reverse (bool): Optional. Whether to reverse the connectivity of
                the MBTRIs being generated (defaults to False).

        Returns:
            mb_tris (list of Entity Handle): List of the entity handles of the
                MBTRIs connecting the two ribs.
        """
        mb_tris = []
        for rib_loci_index, _ in enumerate(rib1.rib_loci[0:-1]):
            corner1 = rib1.mb_verts[rib_loci_index]
            corner2 = rib1.mb_verts[rib_loci_index + 1]
            corner3 = rib2.mb_verts[rib_loci_index + 1]
            corner4 = rib2.mb_verts[rib_loci_index]
            corners = [corner1, corner2, corner3, corner4]
            mb_tris += create_moab_tris_from_verts(
                corners, self.mbc, reverse=reverse
            )
        return mb_tris

    def _generate_pymoab_verts(self):
        """Generate MBVERTEX entities from rib loci in all surfaces
        (Internal function not intended to be called externally)
        """
        [
            surface._generate_pymoab_verts(self.mbc)
            for surface in self.Surfaces.values()
        ]

    def _generate_curved_surfaces_pydagmc(self, continuous_360=False):
        """Generate the faceted representation of each curved surface and
        add it to the PyDAGMC model, remembering the surface ids. The sense
        of the triangles should point outward (increasing radial direction),
        with the exception of the first surface, which should point inward
        since the implicit complement is being used for the plasma chamber.
        (Internal function not intended to be called externally)

        Arguments:
            continuous_360 (bool): flag indicating whether 360-degree,
                continuous geometries should be generated.
        """
        self.curved_surface_ids = []
        surfaces = list(self.Surfaces.values())
        first_surface = surfaces[0]
        for surface in surfaces:
            mb_tris = []

            if continuous_360:
                ribs = surface.Ribs[:-1] + [surface.Ribs[0]]
            else:
                ribs = surface.Ribs

            for rib, next_rib in zip(ribs[0:-1], ribs[1:]):
                mb_tris += self._connect_ribs_with_tris_moab(
                    rib,
                    next_rib,
                    reverse=(surface == first_surface),
                )
            dagmc_surface = self.dag_model.create_surface()
            self.dag_model.mb.add_entities(dagmc_surface.handle, mb_tris)
            self.curved_surface_ids.append(dagmc_surface.id)

    def _generate_end_cap_surfaces_pydagmc(self):
        """Generate the faceted representation of the planar end cap surfaces
        and add them to the PyDAGMC model, remembering the surface ids.
        The sense of the triangles should point toward the implicit complement.
        (Internal function not intended to be called externally)
        """
        self.end_cap_surface_ids = []
        for surface, next_surface in zip(
            list(self.Surfaces.values())[0:-1],
            list(self.Surfaces.values())[1:],
        ):
            end_cap_pair = []
            for index in (0, -1):
                mb_tris = self._connect_ribs_with_tris_moab(
                    surface.Ribs[index],
                    next_surface.Ribs[index],
                    reverse=(index == -1),
                )
                end_cap = self.dag_model.create_surface()
                self.mbc.add_entities(end_cap.handle, mb_tris)
                end_cap_pair.append(end_cap.id)

            self.end_cap_surface_ids.append(end_cap_pair)

    def _generate_volumes_pydagmc(self, continuous_360=False):
        """Use the curved surface and end cap surface IDs to build the
        the volumes by applying the correct surface sense to each surface.
        The convention here is to point the surface sense toward the implicit
        complement, or if the surface is between two volumes then the surface
        sense should point in the increasing radial direction.
        (Internal function not intended to be called externally)

        Arguments:
            continuous_360 (bool): flag indicating whether 360-degree,
                continuous geometries should be generated.
        """

        [self.dag_model.create_volume() for _ in list(self.Surfaces)[:-1]]

        # First surface goes to the implicit complement (plasma chamber)
        first_surface = self.dag_model.surfaces_by_id[
            self.curved_surface_ids[0]
        ]
        first_surface.senses = [
            self.dag_model.volumes_by_id[first_surface.id],
            None,
        ]

        for surface_id in self.curved_surface_ids[1:-1]:
            self.dag_model.surfaces_by_id[surface_id].senses = [
                self.dag_model.volumes_by_id[surface_id - 1],
                self.dag_model.volumes_by_id[surface_id],
            ]

        # if it the last surface it goes to the implicit complement
        last_surface = self.dag_model.surfaces_by_id[
            self.curved_surface_ids[-1]
        ]
        last_surface.senses = [
            self.dag_model.volumes_by_id[last_surface.id - 1],
            None,
        ]

        # all end caps go to the implicit complement.
        if not continuous_360:
            for vol_id, end_cap_ids in enumerate(
                self.end_cap_surface_ids, start=1
            ):
                for end_cap_id in end_cap_ids:
                    self.dag_model.surfaces_by_id[end_cap_id].senses = [
                        self.dag_model.volumes_by_id[vol_id],
                        None,
                    ]

    def _tag_volumes_with_materials_pydagmc(self):
        """Tag each volume with the appropriate material name
        (Internal function not intended to be called externally)
        """
        for vol, (layer_name, layer_data) in zip(
            self.dag_model.volumes,
            list(self.radial_build.radial_build.items())[1:],
        ):

            mat = layer_data.get("mat_tag", layer_name)
            group = pydagmc.Group.create(self.dag_model, name="mat:" + mat)
            group.add_set(vol)
            layer_data["vol_id"] = vol.id

    def generate_components_pydagmc(self):
        """Use PyDAGMC to build a DAGMC model of the invessel components"""
        self._logger.info(
            "Generating DAGMC model of in-vessel components with PyDAGMC..."
        )

        if np.isclose(
            self.radial_build.toroidal_angles[-1]
            - self.radial_build.toroidal_angles[0],
            360.0,
        ):
            continuous_360 = True
        else:
            continuous_360 = False

        self._generate_pymoab_verts()
        self._generate_curved_surfaces_pydagmc(continuous_360=continuous_360)
        if not continuous_360:
            self._generate_end_cap_surfaces_pydagmc()
        self._generate_volumes_pydagmc(continuous_360=continuous_360)
        self._tag_volumes_with_materials_pydagmc()

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
                merge_surfaces(inner_surface_id, prev_outer_surface_id)
                prev_outer_surface_id = outer_surface_id

    def import_step_cubit(self):
        """Imports STEP files from in-vessel build into Coreform Cubit."""
        for name, data in self.radial_build.radial_build.items():
            vol_id = import_step_cubit(name, self.export_dir)
            data["vol_id"] = vol_id

    def export_step(self, export_dir=""):
        """Export CAD solids as STEP files via CadQuery.

        Arguments:
            export_dir (str): directory to which to export the STEP output files
                (defaults to empty string).
        """
        self._logger.info("Exporting STEP files for in-vessel components...")

        self.export_dir = export_dir

        for name, component in self.Components.items():
            export_path = Path(self.export_dir) / Path(name).with_suffix(
                ".step"
            )
            cq.exporters.export(component, str(export_path))

    def extract_solids_and_mat_tags(self):
        """Get a list of all cadquery solids, and a corresponding list of
        the respective material tags.

        Returns:
            solids (list): list of in-vessel component CadQuery solid objects.
            mat_tags (list): list of in-vessel component material tags.
        """
        solids = []
        mat_tags = []

        for name, solid in self.Components.items():
            solids.append(solid)
            mat_tags.append(self.radial_build.radial_build[name]["mat_tag"])

        return solids, mat_tags

    def mesh_components_moab(self, components):
        """Creates a tetrahedral mesh of in-vessel component volumes via MOAB.
        This mesh is created using the point cloud of the specified components
        and as such, each component's mesh will be one tetrahedron thick.

        Arguments:
            components (array of str): array containing the names of the
                in-vessel components to be meshed.
        """
        self._logger.info(
            "Generating tetrahedral mesh of in-vessel component(s) via MOAB..."
        )

        def remove_inner_component(component):
            """Upon identification of a requested component whose meshing is
            not supported by the MOAB workflow, removes that component from the
            input list and raises a warning.

            Arguments:
                component (str): component to be removed.
            """
            w = Warning(
                f"Meshing of {component} volume not supported for MOAB "
                f"workflow; {component} volume will be removed from list of "
                "components to be meshed."
            )
            self._logger.warning(w.args[0])
            components.remove(component)

        if "plasma" in components:
            remove_inner_component("plasma")
        elif "chamber" in components:
            remove_inner_component("chamber")

        surface_keys = list(self.Surfaces.keys())

        # Check if components list is ordered correctly
        sorted_components = sorted(
            components, key=lambda component: surface_keys.index(component)
        )
        if components != sorted_components:
            w = Warning(
                "List of components to be meshed is not properly ordered. "
                "Reordering input list."
            )
            self._logger.warning(w.args[0])
            components = sorted_components

        # Initialize the list of Surface class objects to be included in the
        # mesh, to be used to define mesh vertices on those surfaces later
        surfaces = []
        # Initialize the list booleans identifying whether the regions between
        # mesh surfaces should be meshed or not
        gap_map = []

        # Identify surfaces and gaps in mesh
        for component in components:
            # Extract inner and outer surfaces of current component
            outer_surface = self.Surfaces[component]
            # Inner surface of current component is outer surface of the
            # previous component. Since surfaces are created in order of
            # components and named after the component for which they are the
            # outer surface, it can be found by the ordered list of surface
            # keys
            inner_surf_idx = surface_keys.index(component) - 1
            inner_component = surface_keys[inner_surf_idx]
            inner_surface = self.Surfaces[inner_component]

            # Handle first component
            if len(surfaces) == 0:
                surfaces.append(inner_surface)
            # If the inner component is not the previous component specified to
            # be meshed, identify a gap and add the inner surface
            elif surfaces[-1] != inner_surface:
                surfaces.append(inner_surface)
                # Don't mesh the gap between this surface and its predecessor
                gap_map.append(True)

            surfaces.append(outer_surface)
            gap_map.append(False)

        self.moab_mesh = InVesselComponentMesh(surfaces, gap_map, self._logger)
        self.moab_mesh.create_vertices()
        self.moab_mesh.create_mesh()

    def export_mesh_moab(self, filename, export_dir=""):
        """Exports a tetrahedral mesh of in-vessel component volumes in H5M
        format via MOAB.

        Arguments:
            filename (str): name of H5M output file.
            export_dir (str): directory to which to export the h5m output file
                (defaults to empty string).
        """
        self.moab_mesh.export_mesh(filename, export_dir=export_dir)

    def mesh_components_gmsh(
        self, components, min_mesh_size=5.0, max_mesh_size=20.0, algorithm=1
    ):
        """Creates a tetrahedral mesh of in-vessel component volumes via Gmsh.

        Arguments:
            components (array of str): array containing the names of the
                in-vessel components to be meshed.
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
        """
        self._logger.info(
            "Generating tetrahedral mesh of in-vessel component(s) via Gmsh..."
        )

        gmsh.initialize()

        if self._use_pydagmc:
            self._gmsh_from_pydagmc(
                components, min_mesh_size, max_mesh_size, algorithm
            )
        else:
            self._gmsh_from_cadquery(
                components, min_mesh_size, max_mesh_size, algorithm
            )

    def _gmsh_from_pydagmc(
        self, components, min_mesh_size, max_mesh_size, algorithm
    ):
        """Adds PyDAGMC geometry to Gmsh instance.
        (Internal function not intended to be called externally)

        Arguments:
            components (array of str): array containing the names of the
                in-vessel components to be meshed.
            min_mesh_size (float): minimum size of mesh elements.
            max_mesh_size (float): maximum size of mesh elements.
            algorithm (int): integer identifying the meshing algorithm to use
                for the surface boundary.
        """
        mesh_files = []

        # Extract each component from PyDAGMC model and remesh it in Gmsh
        for component in components:
            volume_id = self.radial_build.radial_build[component]["vol_id"]

            vtk_path = str(Path(f"volume_{volume_id}_tmp").with_suffix(".vtk"))
            self.dag_model.volumes_by_id[volume_id].to_vtk(vtk_path)

            mesh_files.append(
                create_vol_mesh_from_surf_mesh(
                    min_mesh_size, max_mesh_size, algorithm, vtk_path
                )
            )

        # Combine all component meshes into one
        for mesh_file in mesh_files:
            gmsh.merge(mesh_file)
            Path(mesh_file).unlink()

        gmsh.model.mesh.removeDuplicateNodes()

    def _gmsh_from_cadquery(
        self, components, min_mesh_size, max_mesh_size, algorithm
    ):
        """Adds CadQuery geometry to Gmsh instance.
        (Internal function not intended to be called externally)

        Arguments:
            components (array of str): array containing the names of the
                in-vessel components to be meshed.
            min_mesh_size (float): minimum size of mesh elements.
            max_mesh_size (float): maximum size of mesh elements.
            algorithm (int): integer identifying the meshing algorithm to use
                for the surface boundary.
        """
        gmsh.option.setNumber("Mesh.MeshSizeMin", min_mesh_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", max_mesh_size)
        gmsh.option.setNumber("Mesh.Algorithm", algorithm)

        for component in components:
            gmsh.model.occ.importShapesNativePointer(
                self.Components[component].wrapped._address()
            )

        gmsh.model.occ.synchronize()

        gmsh.model.mesh.generate(dim=3)

    def export_mesh_gmsh(self, filename, export_dir=""):
        """Exports a tetrahedral mesh of in-vessel component volumes in H5M
        format via Gmsh and MOAB.

        Arguments:
            filename (str): name of H5M output file.
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

    def mesh_components_cubit(
        self,
        components,
        mesh_size=5,
        anisotropic_ratio=100.0,
        deviation_angle=5.0,
        import_dir="",
    ):
        """Creates a tetrahedral mesh of in-vessel component volumes via
        Coreform Cubit.

        Arguments:
            components (array of str): array containing the names of the
                in-vessel components to be meshed.
            mesh_size (float): controls the size of the mesh. Takes values
                between 1.0 (finer) and 10.0 (coarser) (defaults to 5.0).
            anisotropic_ratio (float): controls edge length ratio of elements
                (defaults to 100.0).
            deviation_angle (float): controls deviation angle of facet from
                surface (i.e., lesser deviation angle results in more elements
                in areas with higher curvature) (defaults to 5.0).
            import_dir (str): directory containing the STEP file of
                the in-vessel component (defaults to empty string).
        """
        self._logger.info(
            "Generating tetrahedral mesh of in-vessel component(s) via Coreform"
            " Cubit..."
        )

        create_new_cubit_instance()

        volume_ids = []

        for component in components:
            volume_id = import_step_cubit(component, import_dir)
            volume_ids.append(volume_id)

        mesh_surface_coarse_trimesh(
            anisotropic_ratio=anisotropic_ratio,
            deviation_angle=deviation_angle,
        )
        mesh_volume_auto_factor(volume_ids, mesh_size=mesh_size)

    def export_mesh_cubit(self, filename, export_dir=""):
        """Exports a tetrahedral mesh of in-vessel component volumes in H5M
        format via Coreform Cubit and MOAB.

        Arguments:
            filename (str): name of H5M output file.
            export_dir (str): directory to which to export the h5m output file
                (defaults to empty string).
        """
        self._logger.info("Exporting mesh H5M file...")

        export_mesh_cubit(
            filename=filename,
            export_dir=export_dir,
            delete_upon_export=True,
        )


class Surface(object):
    """An object representing a surface formed by lofting across a set of
    "ribs" located at different toroidal planes and offset from a reference
    surface.

    Arguments:
        ref_surf (object): ReferenceSurface object. Must have a method
            'angles_to_xyz(toroidal_angles, poloidal_angles, s, scale)' that
            returns an Nx3 numpy array of cartesian coordinates for any closed
            flux surface label, s, poloidal angle (theta), and toroidal angle
            (phi).
        s (float): the normalized closed flux surface label defining the point
            of reference for offset.
        theta_list (np.array(double)): the set of poloidal angles specified for
            each rib [rad].
        phi_list (np.array(double)): the set of toroidal angles defining the
            plane in which each rib is located [rad].
        offset_mat (np.array(double)): the set of offsets from the surface
            defined by s for each toroidal angle, poloidal angle pair on the
            surface [cm].
        scale (float): a scaling factor between input and output data.
    """

    def __init__(self, ref_surf, s, theta_list, phi_list, offset_mat, scale):

        self.ref_surf = ref_surf
        self.s = s
        self.theta_list = theta_list
        self.phi_list = phi_list
        self.offset_mat = offset_mat
        self.scale = scale

        self.surface = None

    def populate_ribs(self):
        """Populates Rib class objects for each toroidal angle specified in
        the surface.
        """
        self.Ribs = [
            Rib(
                self.ref_surf,
                self.s,
                self.theta_list,
                phi,
                self.offset_mat[i, :],
                self.scale,
            )
            for i, phi in enumerate(self.phi_list)
        ]

    def calculate_loci(self):
        """Calls calculate_loci method in Rib class for each rib in the surface."""
        [rib.calculate_loci() for rib in self.Ribs]

    def _generate_pymoab_verts(self, mbc):
        """Generate MBTVERTEX entities from rib loci in all ribs.
        (Internal function not intended to be called externally)

        Arguments:
            mbc (PyMOAB Core): PyMOAB Core instance to add the MBVERTEX
                entities to.
        """
        [rib._generate_pymoab_verts(mbc) for rib in self.Ribs]

    def generate_surface(self):
        """Constructs a surface by lofting across a set of rib splines."""
        if not self.surface:
            self.surface = cq.Solid.makeLoft(
                [rib.generate_rib() for rib in self.Ribs]
            )

        return self.surface

    def get_loci(self):
        """Returns the set of point-loci defining the ribs in the surface."""
        return np.array([rib.rib_loci for rib in self.Ribs])


class Rib(object):
    """An object representing a curve formed by interpolating a spline through
    a set of points located in the same toroidal plane but differing poloidal
    angles and offset from a reference curve.

    Arguments:
        ref_surf (object): ReferenceSurface object. Must have a method
            'angles_to_xyz(toroidal_angles, poloidal_angles, s, scale)' that
            returns an Nx3 numpy array of cartesian coordinates for any closed
            flux surface label, s, poloidal angle (theta), and toroidal angle
            (phi).
        s (float): the normalized closed flux surface label defining the point
            of reference for offset.
        phi (np.array(double)): the toroidal angle defining the plane in which
            the rib is located [rad].
        theta_list (np.array(double)): the set of poloidal angles specified for
            the rib [rad].
        offset_list (np.array(double)): the set of offsets from the curve
            defined by s for each toroidal angle, poloidal angle pair in the rib
            [cm].
        scale (float): a scaling factor between input and output data.
    """

    def __init__(self, ref_surf, s, theta_list, phi, offset_list, scale):

        self.ref_surf = ref_surf
        self.s = s
        self.theta_list = theta_list
        self.phi = phi
        self.offset_list = offset_list
        self.scale = scale

    def _calculate_cartesian_coordinates(self, poloidal_offset=0):
        """Return an N x 3 NumPy array containing the Cartesian coordinates of
        the points at this toroidal angle and N different poloidal angles, each
        offset slightly.
        (Internal function not intended to be called externally)

        Arguments:
            poloidal_offset (float) : some offset to apply to the full set of
                poloidal angles for evaluating the location of the Cartesian
                points (defaults to 0).
        """
        return self.ref_surf.angles_to_xyz(
            self.phi,
            self.theta_list + poloidal_offset,
            self.s,
            self.scale,
        )

    def _normals(self):
        """Approximate the normal to the curve at each poloidal angle by first
        approximating the tangent to the curve and then taking the
        cross-product of that tangent with a vector defined as normal to the
        plane at this toroidal angle.
        (Internal function not intended to be called externally)

        Arguments:
            r_loci (np.array(double)): Cartesian point-loci of reference
                surface rib [cm].
        """
        eps = 1e-4
        next_pt_loci = self._calculate_cartesian_coordinates(eps)

        tangent = next_pt_loci - self.rib_loci

        plane_norm = np.array([-np.sin(self.phi), np.cos(self.phi), 0])

        norm = np.cross(plane_norm, tangent)

        return normalize(norm)

    def calculate_loci(self):
        """Generates Cartesian point-loci for stellarator rib. Sets the last
        element to the value of the first to ensure the loop is closed exactly.
        """
        self.rib_loci = self._calculate_cartesian_coordinates()
        if not np.all(self.offset_list == 0):
            self.rib_loci += self.offset_list[:, np.newaxis] * self._normals()

        self.rib_loci[-1] = self.rib_loci[0]

    def _generate_pymoab_verts(self, mbc):
        """Converts point-loci to MBVERTEX and adds them to a PyMOAB
        Core instance. The first and last rib loci are identical. To avoid
        having separate MBVERTEX entities which are coincident, the last
        element in rib_loci is not made into an MBVERTEX, and the entity
        handle corresponding to the first rib locus is appended to the array
        of MBVERTEX, closing the loop.
        (Internal function not intended to be called externally)

        Arguments:
            mbc (PyMOAB Core): PyMOAB Core instance to add the MBVERTEX
                entities to.
        """
        self.mb_verts = mbc.create_vertices(
            self.rib_loci[0:-1].flatten()
        ).to_array()
        self.mb_verts = np.append(self.mb_verts, self.mb_verts[0])

    def generate_rib(self):
        """Constructs component rib by constructing a spline connecting all
        specified Cartesian point-loci.
        """
        rib_loci = [cq.Vector(tuple(r)) for r in self.rib_loci]
        spline = cq.Edge.makeSpline(rib_loci).close()
        rib_spline = cq.Wire.assembleEdges([spline]).close()

        return rib_spline


class InVesselComponentMesh(ToroidalMesh):
    """Generates a tetrahedral mesh of in-vessel component volumes via MOAB.
    This mesh is created using the point cloud of each component's Surface
    class objects and as such, each component's mesh will be one tetrahedron
    thick. Inherits from ToroidalMesh.

    Arguments:
        surfaces (list of object): the Surface class objects of the components
            in the mesh, ordered radially outward.
        gap_map (list of bool): an ordered map indicating gaps in the mesh. As
            such, should be one entry shorter than "surfaces" argument.
        logger (object): logger object (defaults to None). If no logger is
            supplied, a default logger will be instantiated.
    """

    def __init__(self, surfaces, gap_map, logger=None):
        super().__init__(logger=logger)

        self.surfaces = surfaces
        self.gap_map = gap_map

    @property
    def surfaces(self):
        return self._surfaces

    @surfaces.setter
    def surfaces(self, list):
        self._surfaces = list
        # Extract dimensions of surface point cloud
        self._num_ribs = len(list[0].phi_list)
        self._num_rib_pts = len(list[0].theta_list)

    @property
    def gap_map(self):
        return self._gap_map

    @gap_map.setter
    def gap_map(self, list):
        if len(list) != len(self._surfaces) - 1:
            e = AssertionError(
                "'gap_map' indicates gap regions in the mesh between the "
                "'surfaces' argument and as such, should be one entry shorter "
                "than 'surfaces'."
            )
            self._logger.error(e.args[0])
            raise e

        self._gap_map = list

    def create_vertices(self):
        """Creates mesh vertices and adds them to PyMOAB core."""
        coords = []
        for surface in self.surfaces:
            for rib in surface.Ribs:
                coords.extend(rib.rib_loci)
        coords = np.array(coords)
        self.add_vertices(coords)

    def create_mesh(self):
        """Creates volumetric mesh in real space."""
        for surface_idx, _ in enumerate(self.surfaces[:-1]):
            if self.gap_map[surface_idx]:
                continue  # Skip iteration if a gap is indicated
            for toroidal_idx in range(self._num_ribs - 1):
                for poloidal_idx in range(self._num_rib_pts - 1):
                    self._create_tets_from_hex(
                        surface_idx, poloidal_idx, toroidal_idx
                    )

    def _get_vertex_id(self, vertex_idx):
        """Computes vertex index in row-major order as stored by MOAB from
        three-dimensional n x 3 matrix indices.
        (Internal function not intended to be called externally)

        Arguments:
            vertex_idx (list): vertex's 3-D grid indices in order
                [surface index, poloidal angle index, toroidal angle index]

        Returns:
            id (int): vertex index in row-major order as stored by MOAB
        """
        surface_idx, poloidal_idx, toroidal_idx = vertex_idx

        verts_per_surface = self._num_ribs * self._num_rib_pts
        surface_offset = surface_idx * verts_per_surface

        toroidal_offset = toroidal_idx * self._num_rib_pts

        poloidal_offset = poloidal_idx
        # Wrap around if poloidal angle is 2*pi
        if poloidal_idx == self._num_rib_pts - 1:
            poloidal_offset = 0

        id = surface_offset + toroidal_offset + poloidal_offset

        return id


class RadialBuild(object):
    """Parametrically defines ParaStell in-vessel component geometries.
    In-vessel component thicknesses are defined on a grid of toroidal and
    poloidal angles, and the first wall profile is defined by a closed flux
    surface extrapolation.

    Arguments:
        toroidal_angles (array of float): toroidal angles at which radial build
            is specified. This list should always begin at 0.0 and it is
            advised not to extend beyond one stellarator period. To build a
            geometry that extends beyond one period, make use of the 'repeat'
            parameter [deg].
        poloidal_angles (array of float): poloidal angles at which radial build
            is specified. This array should always span 360 degrees [deg].
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
                        neutronics model (str, defaults to None). If None is
                        supplied, the 'component' key will be used.
                }
            }.
        split_chamber (bool): if wall_s > 1.0, separate interior vacuum
            chamber into plasma and scrape-off layer components (defaults to
            False). If an item with a 'sol' key is present in the radial_build
            dictionary, settting this to False will not combine the resultant
            'chamber' with 'sol'. To include a custom scrape-off layer
            definition for 'chamber', add an item with a 'chamber' key and
            desired 'thickness_matrix' value to the radial_build dictionary.
        logger (object): logger object (defaults to None). If no
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

        self._toroidal_angles = angle_list
        if self._toroidal_angles[0] != 0.0:
            e = ValueError("The first entry in toroidal_angles must be 0.0.")
            self._logger.error(e.args[0])
            raise e
        if self._toroidal_angles[-1] > 360.0:
            e = ValueError("Toroidal extent cannot exceed 360.0 degrees.")
            self._logger.error(e.args[0])
            raise e

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

        self._poloidal_angles = angle_list
        if self._poloidal_angles[-1] - self._poloidal_angles[0] > 360.0:
            e = AssertionError(
                "Poloidal extent must span exactly 360.0 degrees."
            )
            self._logger.error(e.args[0])
            raise e

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


if __name__ == "__main__":
    generate_invessel_build()
