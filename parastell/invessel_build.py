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

    def __init__(self):
        self.poloidal_perturbation = 1e-4

    def angles_to_xyz(self, toroidal_angle, poloidal_angles, s, scale):
        """Method to go from a location defined by two angles and some
        constant to x, y, z coordinates.

        Arguments:
            toroidal_angles (float): toroidal angle at which to evaluate
                Cartesian coordinates [rad].
            poloidal_angles (iterable of float): poloidal angles at which to
                evaluate Cartesian coordinates [rad].
            s (float): Generic parameter which may affect the evaluation of
                the cartesian coordinate at a given angle pair.
            scale (float): a scaling factor between input and output data.

        Returns:
            coords (Nx3 numpy.array): array of Cartesian coordinates at each
                angle pair specified.
        """
        pass

    def calculate_tangents(self, toroidal_angle, poloidal_angles, s, scale):
        """Compute the tangents of a set of points, defined by a set of
        poloidal angles, at a given toroidal angle.

        Arguments:
            toroidal_angles (float): toroidal angle at which to evaluate
                tangents [rad].
            poloidal_angles (iterable of float): poloidal angles at which to
                evaluate tangents [rad].
            s (float): Generic parameter which may affect the evaluation of
                the cartesian coordinate at a given angle pair.
            scale (float): a scaling factor between input and output data.

        Returns:
            (Nx3 numpy.array): array of poloidal tangents at each angle pair
                specified.
        """
        backward_pt_loci = self.angles_to_xyz(
            toroidal_angle,
            poloidal_angles - self.poloidal_perturbation,
            s,
            scale,
        )
        forward_pt_loci = self.angles_to_xyz(
            toroidal_angle,
            poloidal_angles + self.poloidal_perturbation,
            s,
            scale,
        )

        return normalize(forward_pt_loci - backward_pt_loci)


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
        super().__init__()

        self.vmec_obj = vmec_obj

    def angles_to_xyz(self, toroidal_angle, poloidal_angles, s, scale):
        """Evaluate the Cartesian coordinates for a set of toroidal and
        poloidal angles and flux surface label.

        Arguments:
            toroidal_angles (float): toroidal angle at which to evaluate
                Cartesian coordinates [rad].
            poloidal_angles (iterable of float): poloidal angles at which to
                evaluate Cartesian coordinates [rad].
            s (float): the normalized closed flux surface label defining the
                point of reference for offset.
            scale (float): a scaling factor between input and output data.

        Returns:
            coords (Nx3 numpy.array): array of Cartesian coordinates at each
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
              clockwise (facing in the positive toroidal direction)
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

    Optional attributes:
        poloidal_perturbation (float): perturbation to apply to poloidal angles
            for computing profile tangents via central difference (defaults to
            1e-4).
    """

    def __init__(self, rib_data, toroidal_angles, poloidal_angles, **kwargs):
        super().__init__()

        self.rib_data = rib_data
        self.toroidal_angles = toroidal_angles
        self.poloidal_angles = poloidal_angles

        for name in kwargs.keys() & ("poloidal_perturbation", "stuff"):
            self.__setattr__(name, kwargs[name])

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
        shifted_toroidal_angles = self.toroidal_angles[:-1] + toroidal_shift
        rib_subset = self.rib_data[:-1]
        self._extract_rib_data(
            rib_subset, shifted_toroidal_angles, self.poloidal_angles
        )

        # Poloidal Periodicity Before Period
        poloidal_shift = -360.0
        shifted_poloidal_angles = self.poloidal_angles[:-1] + poloidal_shift
        rib_subset = self.rib_data[:, :-1]
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
        poloidal_shift = 360.0
        shifted_poloidal_angles = self.poloidal_angles[1:] + poloidal_shift
        rib_subset = self.rib_data[:, 1:]
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
            toroidal_angles (float): toroidal angle at which to evaluate
                Cartesian coordinates [rad].
            poloidal_angles (iterable of float): poloidal angles at which to
                evaluate Cartesian coordinates [rad].
            s (float): Not used.
            scale (float): a scaling factor between input and output data.

        Returns:
            coords (Nx3 numpy.array): array of Cartesian coordinates at each
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


class Port(object):
    """An object to model a port that can cut through an `InVesselBuild`,
    subject to the geometric constraints of pre-existing blanket
    components. A `Port` is defined by three fundamental attributes:

        1) Planar shape/parameter
            - `Port`s are constructed as CadQuery `Solid` objects.
            - Current support only exists for cylindrical ports, which
                have circular faces.
            - For a circular `Port`, a corresponding `radius` must be provided
                (see Arguments).
            - Future implementation will include support for other planar
                shapes and corresponding defining parameters.

        2) Orientation vector
            - Vector defined by spherical polar and azimuthal angles in a
                local coordinate frame at the `Port`'s base position.
            - Spherical to Cartesian conversions are done in a local frame of
                reference at a base position (see below) following this right-
                handed convention:
                * x: Positive along the stellarator's radial direction.
                * y: Positive along the counter-clockwise toroidal
                        direction.
                * z: Coterminous with the stellarator's z-axis.
            - By defining the coordinate system in a local frame, multiple
                `Port`s with the same relative orientations at different
                toroidal positions along the stellarator can be defined with
                the same polar and azimuthal angles.
            - If neither orientation angle is specified, the `Port` axis
                defaults to the surface's outward unit normal.

        3) Base position vector
            - Position vector defining the intersection of the `Port` axis
                with the plasma-facing reference surface.
            - Defined by a toroidal and poloidal angle (phi, theta) to
                determine position along the stellarator body.
            - Radial position is calculated internally by producing the
                minimum inset value such that the entirety of the `Port`
                extends beyond the first wall into the plasma chamber for a
                complete cut.
            - Minimum inset value is dependent on the relationship between the
                `Port`'s orientation, cross-sectional area, and local surface
                topology.

    Currently, support is limited to cylindrical `Port`s with circular cross-
        sections. `Port` wall thickness and material are not modeled. Future
        implementations will include:
        * Ability to define `Port` wall thickness/material
        * Additional geometries (elliptical planes, rectangular planes)

    Arguments:
        ref_surf (object): `ReferenceSurface` object. Must have a method
            'angles_to_xyz(toroidal_angles, poloidal_angles, s, scale)' that
            returns an Nx3 `numpy.array` of Cartesian coordinates for any
            closed flux surface label, s, poloidal angle (theta), and
            toroidal angle (phi).
        wall_s (float): closed flux surface label extrapolation at wall.
        base_toroidal_angle (float): toroidal angle of the `Port`'s interior
            base. Taken with `base_poloidal_angle` to create a position vector
            to define base location. Range [0, period] [deg].
        base_poloidal_angle (float): poloidal angle of the `Port`'s interior
            base. Taken with `base_toroidal_angle` to create a position vector
            to define base location. Range [0, 360] [deg].

    Optional attributes:
        orientation_polar_angle (float or None): polar angle of the `Port`'s
            orientation vector in spherical coordinates [deg]. The coordinate
            system is defined in a local frame of reference relative to the
            base point and a global z-axis, which corresponds to the
            stellarator's cylindrical z-axis (see above explanation). If a
            value is provided, must also supply one for
            `orientation_azimuthal_angle`. Otherwise, they will default
            together to correspond to the surface's outward normal unit
            vector.
            (defaults to None)
        orientation_azimuthal_angle (float or None): azimuthal angle of the
            `Port`'s orientation vector in spherical coordinates [deg].
            Coordinate system defined in local frame of reference relative to
            the base point and a global z-axis, which corresponds to the
            stellarator's cylindrical z-axis (see above explanation). If a
            value is provided, must also supply one for
            `orientation_polar_angle`. Otherwise, they will default together
            to correspond to the surface's outward normal unit vector.
            (defaults to None)
        length (float or str): length of the `Port` from the base point to the
            end point [cm]. If a length is provided that is shorter than the
            distance across the `InVesselBuild` at a given orientation, a
            `ValueError` will be raised. If 'min' is provided, the minimum
            length to have the `Port` extend entirely through the
            `InVesselBuild` at a given orientation will be calculated. This
            length does not include an inset addition, which is always added
            inwardly from the base point.
            (defaults to 'min')
        extension (float): added to 'min' length [cm], or a fraction of the
            blanket length if `relative_extension` is `True`.
            (defaults to 0.0)
        relative_extension (bool): tandem argument with `extension` to allow
            for extensions in port length to be relative to blanket thickness,
            rather than absolute.
            (defaults to False)
        scale (float): a scaling factor between input and output data.
            (defaults to m2cm)
        plane_shape (str): cross-sectional shape of the `Port`. Currently only
            'circle' is accepted, corresponding to a cylindrical `Port`.
            (defaults to 'circle')
        radius (float or None): radius of a cylindrical `Port`. Required if
            `plane_shape == 'circle`.
            (defaults to None)
        logger (object): logger object (defaults to None). If no logger is
            supplied, a default logger will be instantiated.
    """

    EINSUM_SUBSCRIPTS = "ij,ij->i"

    def __init__(
        self,
        ref_surf,
        wall_s,
        base_toroidal_angle,
        base_poloidal_angle,
        orientation_polar_angle=None,
        orientation_azimuthal_angle=None,
        length="min",
        extension=0.0,
        relative_extension=False,
        scale=m2cm,
        plane_shape="circle",
        radius=None,
        logger=None,
    ):

        self.logger = logger

        if (orientation_polar_angle is None) != (
            orientation_azimuthal_angle is None
        ):
            e = ValueError(
                "Give both orientation angles, or neither (which uses the "
                "outward surface normal)."
            )
            self._logger.error(e.args[0])
            raise e

        if orientation_polar_angle is not None and not (
            0.0 <= orientation_polar_angle <= 180
        ):
            e = ValueError(
                "orientation_polar_angle must be in [0, 180] degrees."
            )
            self._logger.error(e.args[0])
            raise e

        if not (isinstance(length, str) and length == "min") and not (
            isinstance(length, (int, float)) and length > 0
        ):
            e = ValueError("length must be 'min' or a positive number.")
            self._logger.error(e.args[0])
            raise e

        if extension < 0:
            e = ValueError("extension must be finite and non-negative.")
            self._logger.error(e.args[0])
            raise e

        if plane_shape.lower() == "circle":
            self.plane_shape = plane_shape
            if radius is None or radius <= 0 or not np.isfinite(radius):
                e = ValueError(
                    "Cylindrical port needs a finite positive radius."
                )
                self._logger.error(e.args[0])
                raise e

            else:
                self.radius = radius

        else:
            e = NotImplementedError(
                "Only cylindrical (circular) ports are currently supported."
            )
            self._logger.error(e.args[0])
            raise e

        self.ref_surf = ref_surf
        self.wall_s = wall_s
        self.base_toroidal_angle = base_toroidal_angle
        self.base_poloidal_angle = base_poloidal_angle
        self.orientation_polar_angle = orientation_polar_angle
        self.orientation_azimuthal_angle = orientation_azimuthal_angle
        self.length = length
        self.extension = extension
        self.relative_extension = relative_extension
        self.scale = scale

        self._tor = np.deg2rad(base_toroidal_angle)
        self._pol = np.deg2rad(base_poloidal_angle)
        self._base_cache = None
        self._orientation = None

        # Filled in by resolve()
        self.inset = None
        self.blanket_length = None
        self.resolved_length = None

        # Filled in by _set_origin_point()
        self.origin = None

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, logger_object):
        self._logger = log.check_init(logger_object)

    @staticmethod
    def _full_period_linspace(n, endpoint=True):
        """Generate evenly spaced angular samples over a full revolution.
        (Internal function not intended to be called externally)

        Arguments:
            n (int): number of angular samples.
            endpoint (bool, optional): option to include 2*pi.
            (defaults to True)

        Returns:
            angles (numpy.ndarray): angular samples [rad] spanning [0, 2*pi]
                (if `endpoint == True`) or [0, 2*pi) (if `endpoint == False`).
        """

        return np.linspace(0.0, 2 * np.pi, n, endpoint=endpoint)

    ##### Base Point Geometry #####
    def _xyz(self, toroidal, poloidal):
        """Convert surface flux-coordinates to a vector of Cartesian
        coordinates.
        (Internal function not intended to be called externally)

        Arguments:
            toroidal (float): toroidal angle [rad].
            poloidal (float): poloidal angle [rad].

        Returns:
            cartesian (numpy.ndarray): vector of Cartesian coordinates.
        """

        return self.ref_surf.angles_to_xyz(
            toroidal, np.array([poloidal]), self.wall_s, self.scale
        )[0]

    @property
    def base_point(self):
        """Cache the Cartesian position on the reference surface."""
        if self._base_cache is None:
            self._base_cache = self._xyz(self._tor, self._pol)

        return self._base_cache

    def _calculate_tangents(self, delta=1e-4):
        """Calculate a `Port`'s surface tangent angles by central difference.
        (Internal function not intended to be called externally)

        Arguments:
            delta (float, optional): numerical delta for the central
                difference calculation [rad].
                (defaults to 1e-4)

        Returns:
            e_theta (numpy.ndarray): Cartesian vector of central difference
                for the poloidal angle.
            e_phi (numpy.ndarray): Cartesian vector of central difference
                for the toroidal angle.
        """

        e_theta = (
            self._xyz(self._tor, self._pol + delta)
            - self._xyz(self._tor, self._pol - delta)
        ) / (2.0 * delta)

        e_phi = (
            self._xyz(self._tor + delta, self._pol)
            - self._xyz(self._tor - delta, self._pol)
        ) / (2.0 * delta)

        return e_theta, e_phi

    def _orient_outward_normal(self, e_theta, e_phi):
        """Calculate the outward unit normal vector for a given toroidal and
        poloidal position on a reference surface.
        (Internal function not intended to be called externally)

        Arguments:
            e_theta (numpy.ndarray): Cartesian vector of central difference
                for the poloidal angle.
            e_phi (numpy.ndarray): Cartesian vector of central difference
                for the toroidal angle.

        Returns:
            normal (numpy.ndarray): outward surface normal vector at a given
                position on a reference surface.
        """

        normal = normalize(np.cross(e_theta, e_phi))
        ring = self.ref_surf.angles_to_xyz(
            self._tor,
            self._full_period_linspace(61)[:-1],
            self.wall_s,
            self.scale,
        )

        # Ensure outward orientation
        if np.dot(normal, self.base_point - ring.mean(axis=0)) < 0:
            normal *= -1

        return normal

    ##### Port Orientation Vector #####
    @staticmethod
    def _spherical_to_xyz(polar, azimuthal):
        """Convert spherical to Cartesian coordinates.
        (Internal function not intended to be called externally)

        Arguments:
            polar (float): spherical polar angle (theta) [rad].
            azimuthal (float): spherical azimuthal angle (phi) [rad].

        Returns:
            coordinates (numpy.ndarray): Cartesian unit vector corresponding
                to the supplied spherical angles.
        """

        return np.array(
            [
                np.sin(polar) * np.cos(azimuthal),
                np.sin(polar) * np.sin(azimuthal),
                np.cos(polar),
            ]
        )

    def local_frame(self):
        """Define a local right-handed orthonormal Cartesian coordinate frame
        whose basis directions are expressed in global Cartesian coordinates
        as:
            * x: Positive along the stellarator's radial direction.
            * y: Positive along the counter-clockwise toroidal
                direction.
            * z: Coterminous with the stellarator's z-axis.

        Arguments:
            None

        Returns:
            radial (numpy.ndarray): vector mapping the local radial direction
                (x-axis) to a global frame.
            toroidal (numpy.ndarray): vector mapping the local toroidal
                direction (y-axis) to a global frame.
            z (numpy.ndarray): global z-axis.
        """

        radial = np.array([np.cos(self._tor), np.sin(self._tor), 0.0])
        toroidal = np.array([-np.sin(self._tor), np.cos(self._tor), 0.0])

        return radial, toroidal, np.array([0.0, 0.0, 1.0])

    @property
    def orientation(self):
        """Unit vector along the `Port` axis in a global Cartesian frame."""

        if self._orientation is None:
            if self.orientation_polar_angle is None:
                self._orientation = self._orient_outward_normal(
                    *self._calculate_tangents()
                )

            else:
                x_hat, y_hat, z_hat = self.local_frame()
                c = self._spherical_to_xyz(
                    np.deg2rad(self.orientation_polar_angle),
                    np.deg2rad(self.orientation_azimuthal_angle),
                )
                self._orientation = normalize(
                    c[0] * x_hat + c[1] * y_hat + c[2] * z_hat
                )

        return self._orientation

    ##### Port Cross-Section Helpers #####
    @property
    def section_extent(self):
        """Largest in-plane distance from the axis to the section boundary."""

        return self.radius

    def _section_axes(self):
        """Generate two orthonormal vectors that span the plane perpendicular
        to the `Port` axis.
        """

        a = self.orientation
        helper = (
            np.array([0.0, 0.0, 1.0])  # global z-axis
            if abs(a[2]) < 0.9
            else np.array([1.0, 0.0, 0.0])  # global x-axis
        )
        u = normalize(np.cross(a, helper))

        return u, np.cross(a, u)

    def _in_section(self, lu, lv, tol=1.05):
        """Mask of in-plane coordinates within the section.

        Arguments:
            lu (numpy.ndarray): coordinates along the first in-plane axis.
            lv (numpy.ndarray): coordinates along the second in-plane axis.
            tol (float, optional): multiplicative tolerance applied to the
                section radius.
                (defaults to 1.05)

        Returns:
            inside (numpy.ndarray): Boolean mask identifying points within the
                cross-sectional boundary.
        """

        return np.hypot(lu, lv) <= self.radius * tol

    ##### Plasma Chamber Inset #####
    def compute_minimum_inset(
        self,
        n=15,
        margin_frac=0.1,
        min_margin=1,
        pad=1.2,
        pad_grow=1.5,
        max_grow=4,
        grazing=0.2,
        footprint_grazing=0.05,
    ):
        """Calculate the minimum inward displacement [cm] of a `Port`'s origin
        point, based on its orientation and the local curvature of the
        reference surface, to ensure that the `Port` fully intersects the
        plasma-facing surface.

        Arguments:
            n (int, optional): number of surface samples per angular
                direction. Even values are automatically incremented by one so
                that a sample always falls on a base point. Each attempt costs
                n^2 evaluations of the reference surface.
                (defaults to 15)
            margin_frac (float, optional): safety margin added to the inset,
                as a fraction of the cross-section extent, i.e., the largest
                in-plane distance from the axis to the section boundary (the
                radius, for a circular `Port`)
                (defaults to 0.1)
            min_margin (float, optional): minimum safety  margin added to the
                inset, regardless of the `Port` size [cm]. The margin applied
                is the larger of this and `margin_frac` multiplied by the
                section extent.
                (defaults to 1.0)
            pad (float, optional): factor by which the sampled patch is
                enlarged beyond the footprint predicted for a flat surface.
                Must exceed 1.0 so that curvature does not push part of the
                footprint outside the patch. It is grown automatically if the
                footprint still reaches the patch edge.
                (defaults to 1.2)
            pad_grow (float, optional): factor by which to enlarge the pad
                size after each successive attempt.
                (defaults to 1.5)
            max_grow (int, optional): maximum number of successive attempts to
                enlarge the sampled surface patch by a factor of `pad_grow`
                before raising a `RuntimeError`.
                (defaults to 4)
            grazing (float, optional): minimum cosine of the angle between the
                `Port` axis and the outward surface normal at the base point.
                Rejects axes that point inward or run nearly parallel to the
                surface, where the footprint becomes very long.
                (defaults to 0.2)
            footprint_grazing (float, optional): minimum cosine of the angle
                between the `Port` axis and the outward surface normal at every
                sample inside the footprint. Catches `Port`s that start on a
                favorable part of the surface but reach a region that is
                parallel or back-facing to the axis.
                (defaults to 0.05)

        Returns:
            inset (float): minimum inset to apply to the base point to ensure
                no overlap between the `Port` body and local surface [cm].
        """

        if pad < 1:
            e = ValueError(
                "Padding factor must be at least 1, so as not to shrink the"
                "sampled patch."
            )
            self._logger.error(e.args[0])
            raise e

        e_theta, e_phi = self._calculate_tangents()
        normal = self._orient_outward_normal(e_theta, e_phi)
        cos_a = np.dot(normal, self.orientation)
        if cos_a < grazing:
            deg_violation = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
            deg_grazing = np.degrees(np.arccos(grazing))
            e = ValueError(
                f"Port axis is {deg_violation:.1f} degrees from the outward"
                " surface normal at its base; it must be within "
                f"{deg_grazing:.0f} degrees."
            )
            self._logger.error(e.args[0])
            raise e

        flip = np.sign(np.dot(np.cross(e_theta, e_phi), normal))
        u, v = self._section_axes()

        # Ensure even-valued n so that the base point is sampled exactly
        if n % 2 == 0:
            n += 1

        for _ in range(max_grow):
            half_theta = (
                pad * self.section_extent / (cos_a * np.linalg.norm(e_theta))
            )
            half_phi = (
                pad * self.section_extent / (cos_a * np.linalg.norm(e_phi))
            )
            thetas = self._pol + np.linspace(-half_theta, half_theta, n)
            phis = self._tor + np.linspace(-half_phi, half_phi, n)

            # P[i, j] = point at (phis[i], thetas[j])
            P = np.array(
                [
                    self.ref_surf.angles_to_xyz(
                        p, thetas, self.wall_s, self.scale
                    )
                    for p in phis
                ]
            )
            rel = P - self.base_point
            axial = rel @ self.orientation
            inside = self._in_section(rel @ u, rel @ v)
            touches_edge = (
                inside[0].any()
                or inside[-1].any()
                or inside[:, 0].any()
                or inside[:, -1].any()
            )
            if not touches_edge:
                break

            pad *= pad_grow

        else:
            e = RuntimeError(
                "Port footprint is not enclosed by the sampled surface patch;"
                " port is too oblique for the local surface curvature."
            )
            self._logger.error(e.args[0])
            raise e

        # Facing check across the whole footprint
        nrm = flip * np.cross(np.gradient(P, axis=1), np.gradient(P, axis=0))
        nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
        if ((nrm @ self.orientation)[inside] < footprint_grazing).any():
            e = ValueError(
                "Port footprint reaches a region where the first-wall "
                "surface is parallel or back-facing to the axis."
            )
            self._logger.error(e.args[0])
            raise e

        return max(0.0, -axial[inside].min()) + max(
            min_margin, margin_frac * self.section_extent
        )

    ##### Tools for Calculating Blanket Length along Port Axis #####
    def _set_origin_point(self):
        """Calculate and cache the `Port`'s interior origin point using the
        base point and inset.
        """

        if self.inset is None:
            self.inset = self.compute_minimum_inset()

        self.origin = self.base_point - self.inset * self.orientation

    @staticmethod
    def _triangulate_loci(loci):
        """Mesh a structured grid of outer-surface point-loci into triangles,
        in the form needed by the Möller-Trumbore ray-triangle intersection
        algorithm.
        (Internal function not intended to be called externally)

        Arguments:
            loci (numpy.ndarray): array of outer-surface point-loci.

        Returns:
            triangles (numpy.ndarray): array containing positional vectors
                for each triangle, with shape (num_triangles, 3, 3).
        """

        def _stack_triangles(a, b, c):
            """Stack three corresponding arrays of triangle vertices into a
            single array of triangles, flattening the leading grid dimensions.
            """

            return np.stack([a, b, c], axis=-2).reshape(-1, 3, 3)

        p00, p01 = loci[:-1, :-1], loci[:-1, 1:]
        p10, p11 = loci[1:, :-1], loci[1:, 1:]
        t1 = _stack_triangles(p00, p01, p11)
        t2 = _stack_triangles(p00, p11, p10)

        return np.concatenate([t1, t2])

    def _prepare_moller_trumbore_tri_vectors(self, triangles, eps=1e-12):
        """Produce source-independent vectors and algebraic values for
        application of the Möller-Trumbore ray-triangles intersection
        algorithm. Called before iteration over each origin point for
        computational efficiency.

        Algorithm and variable naming conventions implemented from:
        https://doi.org/10.1080/10867651.1997.10487468

        (Internal function not intended to be called externally)

        Arguments:
            triangles (numpy.ndarray): array of triangle vertices produced
                from outer-surface point-loci.
            eps (float, optional): relative tolerance for rejecting triangles
                that the ray cannot intersect at a single point. A triangle is
                rejected if |det| <= eps * |E1| * |E2|, which catches rays
                parallel to the triangle plane and degenerate triangles.
                (defaults to 1e-12)

        Returns:
            V0 (numpy.ndarray): first vertex of each triangle.
            E1 (numpy.ndarray): first edge of each triangle (V1 - V0).
            E2 (numpy.ndarray): second edge of each triangle (V2 - V0).
            P (numpy.ndarray): cross-product of the port-axis orientation unit
                vector and E2.
            inv_det (numpy.ndarray): array of the reciprocals of the
                determinant for each triangle, zero for invalid triangles.
            valid (numpy.ndarray): mask of the triangles who pass the
                determinant test described above for `eps`.
        """

        V0, V1, V2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        E1 = V1 - V0
        E2 = V2 - V0
        P = np.cross(self.orientation, E2)
        det = np.einsum(self.EINSUM_SUBSCRIPTS, E1, P)
        scale = np.linalg.norm(E1, axis=1) * np.linalg.norm(E2, axis=1)
        valid = np.abs(det) > eps * scale

        inv_det = np.zeros_like(det)
        inv_det[valid] = 1.0 / det[valid]

        return V0, E1, E2, P, inv_det, valid

    def _moller_trumbore_intersection(
        self, start_point, origin_indep_MT_vals, slack=1e-9
    ):
        """Calculate ray-triangle intersections for a given ray origin using
        the precomputed, origin-independent quantities from the Möller-
        Trumbore algorithm. Current implementation limited to locally
        convex surfaces; multiple hits possible for concave regions. Future
        updates will handle this case.

        Algorithm and variable naming conventions implemented from:
        https://doi.org/10.1080/10867651.1997.10487468

        (Internal function not intended to be called externally)

        Arguments:
            start_point (numpy.ndarray): position vector for a trial ray
                origin.
            origin_indep_MT_vals (tuple): collection of grouped source-
                independent outputs of
                `_prepare_moller_trumbore_tri_vectors()`.
            slack (float, optional): numerical tolerance applied to the
                barycentric coordinates (u,v) when determining whether an
                intersection lies within a triangle.
                (defaults to 1e-9)

        Returns:
            t (numpy.ndarray): distances [cm] from ray origin point to each
                intersected triangle's plane.
        """

        # Triangle and matrix data calculate pre-iteration to maximize loop
        # efficiency
        V0, E1, E2, P, inv_det, valid = origin_indep_MT_vals
        T = start_point - V0
        u = np.einsum(self.EINSUM_SUBSCRIPTS, T, P) * inv_det
        Q = np.cross(T, E1)
        v = (Q @ self.orientation) * inv_det
        t = np.einsum(self.EINSUM_SUBSCRIPTS, E2, Q) * inv_det
        return t[
            (
                valid
                & (u >= -slack)
                & (v >= -slack)
                & (u + v <= 1.0 + slack)
                & (t > 0.0)
            )
        ]

    def _compile_disk_sample_offsets(self, n_radial, n_angular):
        """Generate a concentric sampling of points spanning the `Port`'s
        cross-section, given the desired radial and angular resolutions.
        (Internal function not intended to be called externally)

        Arguments:
            n_radial (int): number of radial samples to generate, spaced
                evenly from 0 to `self.radius`.
            n_angular (int): number of angular samples to generate, spaced
                evenly from 0 to 360 degrees.

        Returns:
            offsets (numpy.ndarray): Array of Cartesian offset vectors [cm]
                from the center of the `Port` cross-section.
        """

        u, v = self._section_axes()
        radii = self.section_extent * np.linspace(0.0, 1.0, n_radial + 1)
        angles = self._full_period_linspace(n_angular, endpoint=True)
        offsets = [np.zeros((1, 3))]
        offsets.extend(
            r * (np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v)
            for r in radii[1:]
        )

        return np.concatenate(offsets, axis=0)

    def _calculate_ivb_length_along_port(
        self,
        outer_loci,
        n_radial=4,
        n_angular=16,
        max_refinements=3,
        atol=0.1,
        rtol=1e-3,
    ):
        """Determine the maximum distance from the `Port`'s base point to the
        first positive intersection of each sampled ray with the triangulated
        outer surface of the `InVesselBuild`. Intersections calculated by the
        Möller-Trumbore ray-triangle intersection algorithm.
        (Internal function not intended to be called externally)

        Arguments:
            outer_loci (numpy.ndarray): array of point-loci defining the
                outermost surface of the `InVesselBuild`.
            n_radial (int, optional): number of radial samples at which to
                generate concentric offsets, spaced evenly from 0 to
                `self.radius`.
                (defaults to 4)
            n_angular (int, optional): number of angular samples at which to
                generate concentric offsets, spaced evenly from 0 to 360
                degrees.
                (defaults to 16)
            max_refinements (int, optional): number of grid refinements to
                iteratively apply to (`n_radial`, `n_angular`) to check that
                a particular intersection distance is not found as a result of
                too coarse a grid. Iterates by 2^refinement.
                (defaults to 3)
            atol (float, optional): absolute tolerance [cm] for similarity
                between each refinement iteration's calculated length and the
                previous iteration. Used in `numpy.isclose()`.
                (defaults to 0.1)
            rtol (float, optional): relative tolerance for similarity between
                each refinement iteration's calculated length and the previous
                iteration. Used in `numpy.isclose()`.
                (defaults to 1e-3)

        Returns:
            ivb_length (float): maximum distance across the `InVesselBuild` in
                the space spanned by the `Port`.
        """

        self._set_origin_point()
        origin_indep_MT_vals = self._prepare_moller_trumbore_tri_vectors(
            self._triangulate_loci(np.asarray(outer_loci))
        )

        previous_length = None
        for refinement in range(max_refinements + 1):
            starts = self.origin + self._compile_disk_sample_offsets(
                n_radial * 2**refinement, n_angular * 2**refinement
            )
            exits = []
            for start in starts:
                hits = self._moller_trumbore_intersection(
                    start, origin_indep_MT_vals
                )
                if hits.size == 0:
                    e = RuntimeError(
                        "A ray from the port cross-section does not intersect"
                        " the modeled outer surface. The port may leave "
                        "through a toroidal segment end."
                    )
                    self._logger.error(e.args[0])
                    raise e

                exits.append(hits.min())

            ivb_length = max(exits) - self.inset
            if previous_length is not None:
                if np.isclose(
                    ivb_length, previous_length, atol=atol, rtol=rtol
                ):
                    return ivb_length

            previous_length = ivb_length

        e = RuntimeError(
            f"Minimum port length did not converge after {max_refinements} "
            f"refinements. Last estimate: {previous_length:.3f}."
        )
        self._logger.error(e.args[0])
        raise e

    @staticmethod
    def _estimate_chord_sagitta(loci, min_sagitta=0.5):
        """Estimate the chord sagitta of a surface curve using the small-angle
        approximation (s ≈ second difference / 8), subject to a minimum
        correction specified by `min_sagitta`.
        (Internal function not intended to be called externally)

        Arguments:
            loci (numpy.ndarray): array of point-loci defining the outermost
                surface of the `InVesselBuild`.
            min_sagitta (float, optional): lower bound on the estimated
                correction.

        Returns:
            chord_sagitta (float): sagitta distance [cm] to add to the
                `InVesselBuild` length calculated by
                `_calculate_ivb_length_along_port()` to account for local
                curvature effects.
        """

        loci = np.asarray(loci)
        d2_pol = np.linalg.norm(
            loci[:, :-2] - 2 * loci[:, 1:-1] + loci[:, 2:], axis=-1
        )
        d2_tor = np.linalg.norm(loci[:-2] - 2 * loci[1:-1] + loci[2:], axis=-1)
        max_second_difference = max(d2_pol.max(), d2_tor.max())

        return max(max_second_difference / 8.0, min_sagitta)

    def resolve(self, outer_loci):
        """Resolve a `Port`'s length attributes to ensure sufficient extension
        along its axis to traverse the associated `InVesselBuild`.

        Arguments:
            outer_loci (numpy.ndarray): array of point-loci defining the
                outermost surface of the `InVesselBuild`.

        Returns:
            Port (object): resolved `Port` object.
        """

        if self.inset is None:
            self.inset = self.compute_minimum_inset()

        self.blanket_length = self._calculate_ivb_length_along_port(
            outer_loci
        ) + self._estimate_chord_sagitta(outer_loci)

        if self.length == "min":
            ext = (
                self.extension * self.blanket_length
                if self.relative_extension
                else self.extension
            )
            self.resolved_length = self.blanket_length + ext

        else:
            self.resolved_length = float(self.length)
            if self.resolved_length < self.blanket_length and not np.isclose(
                self.resolved_length, self.blanket_length
            ):
                e = ValueError(
                    f"Specified port length ({self.resolved_length:.3f} cm) "
                    "is shorter than the minimum required length "
                    f"({self.blanket_length:.3f} cm) to traverse the "
                    "InVesselBuild."
                )
                self._logger.error(e.args[0])
                raise e

        return self

    ##### Outputs #####
    def generate_port(self, pad=0.0):
        """Create a CadQuery solid object for a resolved `Port`.

        Arguments:
            pad (float, optional): optional radial clearance added to the
                cylindrical cutting solid [cm]. Does not modify the stored
                `Port` radius.

        Returns:
            port_solid (cadquery.Solid): oriented CadQuery `Solid` object
                representing the `Port`, to be cut from other `InVesselBuild`
                layers.
        """

        if self.resolved_length is None:
            e = RuntimeError("Call resolve() before building the solid.")
            self._logger.error(e.args[0])
            raise e

        if self.origin is None:
            self._set_origin_point()

        # Currently only "circle" allowed anyways, but conditional
        # infrastructure in place to accomadate future implementations of
        # other plane shapes
        if self.plane_shape.lower() == "circle":
            return cq.Solid.makeCylinder(
                self.radius + pad,
                self.resolved_length + self.inset,
                cq.Vector(*self.origin),
                cq.Vector(*self.orientation),
            )

    def compile_end_points(self, distance, n=15):
        """Generate points spanning the `Port`'s cross-section at a specified
        distance along its axis from the base point. The returned points
        include the cross-section center and evenly spaced points around its
        circular rim.

        Arguments:
            distance (float): distance [cm] from the base point along the
                `Port`'s orientation vector.
            n (int, optional): number of angular samples around the rim.
                (defaults to 15)

        Returns:
            end_points (numpy.ndarray): array of 3D coordinates [cm] defining
                the cross-section center and rim, with shape (n + 1, 3).
        """

        u, v = self._section_axes()
        angles = self._full_period_linspace(n, endpoint=False)
        center = self.base_point + distance * self.orientation
        rim = center + self.section_extent * (
            np.outer(np.cos(angles), u) + np.outer(np.sin(angles), v)
        )

        return np.vstack([center, rim])

    def check_within_period(
        self, period, reach=None, num_s=24, num_ang=16, eps=1e-9
    ):
        """Check whether the sampled points along the circumference of a
            `Port` lie within the toroidal period bounds of the associated
            `InVesselBuild`.

        Arguments:
            period (float): toroidal angular extent of the `InVesselBuild`
                periodic segment [deg].
            reach (float or None, optional): maximum length [cm] over which to
                check `Port` containment. If None is provided, value
                overridden by `self.resolved_length`.
                (defaults to None)
            num_s (int, optional): number of distances over which to iterate
                between 0 and `reach`.
                (defaults to 24)
            num_ang (int, optional): number of angles over which to iterate,
                spaced evenly between 0 and 360 degrees.
                (defaults to 16)
            eps (float, optional): relative tolerance for `Port` exceeding
                periodic boundaries.
                (defaults to 1e-9)

        Returns:
            None
        """

        if period >= 360.0:
            return

        if reach is None:
            if self.resolved_length is None:
                e = RuntimeError("Port length must be resolved first.")
                self._logger.error(e.args[0])
                raise e

            reach = self.resolved_length + self.inset

        if self.origin is None:
            self._set_origin_point()

        for s in np.linspace(0.0, reach, num_s):
            pts = self.compile_end_points(s, n=num_ang)
            phis = np.mod(np.rad2deg(np.arctan2(pts[:, 1], pts[:, 0])), 360.0)
            if (phis < eps).any() or (phis > period + eps).any():
                e = ValueError(
                    "Port crosses a toroidal period boundary; move its base "
                    "toward the middle of the period or reorient it."
                )
                self._logger.error(e.args[0])
                raise e


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
        ports (list): list of kwarg dictionaries for each desired `Port`.
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
        self.ports = []
        self.port_solids = []

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

        for port_kwargs in kwargs.get("ports", []):
            self.add_port(**port_kwargs)

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

    def add_port(self, **port_kwargs):
        """Generate and include a `Port` object to the `InVesselBuild`,
        according to `port_kwargs`. Currently only compatible with CadQuery.
        """

        if self.use_pydagmc:
            raise NotImplementedError("Ports are only supported for CadQuery.")

        period = self.radial_build.toroidal_angles[-1]
        tor = port_kwargs["base_toroidal_angle"]
        if not 0.0 <= tor <= period:
            raise ValueError(
                f"Port base toroidal angle {tor} must lie within the period "
                f"[0, {period}] degrees."
            )

        self.ports.append(
            Port(
                self.ref_surf,
                self.radial_build.wall_s,
                scale=self.scale,
                **port_kwargs,
            )
        )

    def _check_port_exits(self, outermost_solid, tol=1e-3):
        """Ensure all `Port` objects fully traverse the `InVesselBuild`.
        (Internal function not intended to be called externally)

        Arguments:
            outermost_solid (object): CadQuery object representing the
                outermost modelled layer in the `InVesselBuild`.
            tol (float, optional): relative tolerance for `Port` intersection
                with outer surface.
                (defaults to 1e-3)

        Returns:
            None
        """

        for idx, port in enumerate(self.ports):
            pts = port.compile_end_points(port.resolved_length)
            pts += tol * port.orientation

            if any(outermost_solid.isInside(cq.Vector(*p)) for p in pts):
                raise ValueError(
                    f"Port {idx} does not fully exit the outermost component;"
                    " increase its length."
                )

    def check_port_magnet_clearance(
        self, magnet_solids, clearance=0.0, tol=1e-3, on_conflict="raise"
    ):
        """Ensure that `Port` objects do not interfere with magnet positions.

        Arguments:
            magnet_solids (list): list all CadQuery magnet `Solid` objects.
            clearance (float, optional): padding clearance [cm] for acceptable
                distance between a `Port` and a magnet.
                (defaults to 0.0)
            tol (float, optional): spatial tolerance for `Port`/magnet
                intersection [cm^3].
                (defaults to 1e-3)
            on_conflicts (str, optional): setting for raising or warning if
                `Port`/magnet intersection is found.
                (defaults to 'raise')

        Returns:
            conflicts (list): list containing error messages for any `Port`/
                magnet intersections. If `on_conflicts == 'raise'`, then any
                conflicts will result in a `ValueError` being raised, rather
                than the function returning.
        """
        
        period = self.radial_build.toroidal_angles[-1]
        conflicts = []
        for p_idx, port in enumerate(self.ports):
            base_tool = port.generate_port(pad=clearance)
            for k in range(self._repeat + 1):
                tool = base_tool.rotate((0, 0, 0), (0, 0, 1), k * period)
                tb = tool.BoundingBox()
                for m_idx, magnet in enumerate(magnet_solids):
                    mb = magnet.BoundingBox()
                    if (
                        tb.xmax < mb.xmin or tb.xmin > mb.xmax
                        or tb.ymax < mb.ymin or tb.ymin > mb.ymax
                        or tb.zmax < mb.zmin or tb.zmin > mb.zmax
                    ):
                        continue

                    vol = tool.intersect(magnet).Volume()
                    if vol > tol:
                        conflicts.append((p_idx, k, m_idx, vol))

        if conflicts:
            msg = (
                "Port/magnet overlaps (port, period, magnet, cm^3): "
                + "; ".join(
                    f"({p}, {k}, {m}, {v:.1f})" for p, k, m, v in conflicts
                )
            )
            if on_conflict == "raise":
                e = ValueError(msg)
                self._logger.error(e.args[0])
                raise e

            self._logger.warning(msg)

        return conflicts

    def resolve_ports(self, period):
        """Attempt to resolve all `Port` lengths in the `InVesselBuild` using
        `Port.resolve()`.

        Arguments:
            period (float): toroidal angular extent of the `InVesselBuild`
                periodic segment [deg].

        Returns:
            None
        """

        outer_loci = list(self.Surfaces.values())[-1].get_loci()
        for idx, port in enumerate(self.ports):
            try:
                port.compute_minimum_inset()
                port.check_within_period(period, reach=0.0)

                port.resolve(outer_loci)
                port.check_within_period(period)

            except (ValueError, RuntimeError) as err:
                e = type(err)(f"Port {idx}: {err}")
                self._logger.error(e.args[0])
                raise e from err

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

        period = self.radial_build.toroidal_angles[-1]
        segment_angles = period * np.arange(1, self._repeat + 1)

        self.resolve_ports(period)
        self.port_solids = [port.generate_port() for port in self.ports]

        for name, surface in self.Surfaces.items():
            outer_surface = surface.generate_surface()

            if interior_surface is not None:
                segment = outer_surface.cut(interior_surface)
            else:
                segment = outer_surface

            if self.port_solids and name not in ("plasma", "chamber", "sol"):
                for port_solid in self.port_solids:
                    segment = segment.cut(port_solid)

            component = segment
            for angle in segment_angles:
                rot_segment = segment.rotate((0, 0, 0), (0, 0, 1), angle)
                component = component.fuse(rot_segment)

            self.Components[name] = component
            interior_surface = outer_surface

        if self.ports:
            self._check_port_exits(outer_surface)

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

    def merge_surfaces(self):
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
            import_path = Path(self.export_dir) / Path(name)
            vol_id = import_step_cubit(import_path)
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

        gmsh.option.setNumber(
            "General.NumThreads", 0
        )  # Use all available cores

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
            import_path = Path(import_dir) / Path(component)
            volume_id = import_step_cubit(import_path)
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
        phi (float): the toroidal angle defining the plane in which the rib is
            located [rad].
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
        tangents = self.ref_surf.calculate_tangents(
            self.phi, self.theta_list, self.s, self.scale
        )

        plane_norm = np.array([-np.sin(self.phi), np.cos(self.phi), 0])

        normals = np.cross(plane_norm, tangents)

        return normalize(normals)

    def calculate_loci(self):
        """Generates Cartesian point-loci for stellarator rib. Sets the last
        element to the value of the first to ensure the loop is closed exactly.
        """
        self.rib_loci = self.ref_surf.angles_to_xyz(
            self.phi, self.theta_list, self.s, self.scale
        )

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

        self.volumes = []

        self._add_tags_to_core()

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

    def _add_tags_to_core(self):
        """Creates PyMOAB core instance with source strength tag.
        (Internal function not intended to be called externally)
        """
        tag_type = types.MB_TYPE_DOUBLE
        tag_size = 1
        storage_type = types.MB_TAG_DENSE

        vol_tag_name = "Volume"
        self.volume_tag = self.mbc.tag_get_handle(
            vol_tag_name,
            tag_size,
            tag_type,
            storage_type,
            create_if_missing=True,
        )

    def create_vertices(self):
        """Creates mesh vertices and adds them to PyMOAB core."""
        self.coords = []
        for surface in self.surfaces:
            for rib in surface.Ribs:
                self.coords.extend(rib.rib_loci)
        self.coords = np.array(self.coords)
        self.add_vertices(self.coords)

    def _compute_and_tag_tet_volume(self, vert_ids, tet):
        """Computes tetrahedron volume, and sets the corresponding value of
        the respective tag for that tetrahedron.
        (Internal function not intended to be called externally)

        Arguments:
            vert_ids (list of int): tetrahedron vertex indices.
            tet (object): pymoab.EntityHandle of tetrahedron.

        Returns:
            tet_vol (float): volume of tetrahedron.
        """
        tet_vol = self._compute_tet_volume(vert_ids)
        # Tag tetrahedra with data
        self.mbc.tag_set_data(self.volume_tag, tet, [tet_vol])

        return tet_vol

    def create_mesh(self):
        """Creates volumetric mesh in real space."""
        for surface_idx, _ in enumerate(self.surfaces[:-1]):
            if self.gap_map[surface_idx]:
                continue  # Skip iteration if a gap is indicated
            for toroidal_idx in range(self._num_ribs - 1):
                for poloidal_idx in range(self._num_rib_pts - 1):
                    tets, vertex_id_list = self._create_tets_from_hex(
                        surface_idx, poloidal_idx, toroidal_idx
                    )
                    self.volumes.extend(
                        [
                            self._compute_and_tag_tet_volume(vert_ids, tet)
                            for vert_ids, tet in zip(vertex_id_list, tets)
                        ]
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
