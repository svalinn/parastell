import cadquery as cq
import read_vmec
from src.utils import *
import log

import numpy as np
from scipy.interpolate import RegularGridInterpolator

import argparse
import yaml

m2cm = 100


class InVesselBuild(object):
    '''Parametrically models fusion stellarator in-vessel components using
    plasma equilibrium data and user-defined parameters. In-vessel component
    geometries are determined by a user-defined radial build, in which
    thickness values are supplied in a grid of toroidal and poloidal angles,
    and plasma equilibrium VMEC data.

    Arguments:
        vmec (object): plasma equilibrium VMEC object from PyStell-UW.
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
        plasma_h5m_tag (str): optional alternate material tag to use for
            plasma. If none is supplied and the plasma is not excluded,
            'plasma' will be used (defaults to None).
        sol_h5m_tag (str): optional alternate material tag to use for
            scrape-off layer. If none is supplied and the scrape-off layer is
            not excluded, 'sol' will be used (defaults to None).
        logger (object): logger object (defaults to None). If no logger is
            supplied, a default logger will be instantiated.
    '''

    def __init__(
            self,
            vmec,
            build,
            repeat,
            num_phi=61,
            num_theta=61,
            scale=m2cm,
            plasma_h5m_tag=None,
            sol_h5m_tag=None,
            logger=None
    ):
        self.vmec = vmec
        self.build = build
        self.repeat = repeat
        self.num_phi = num_phi
        self.num_theta = num_theta
        self.scale = scale
        self.plasma_h5m_tag = plasma_h5m_tag
        self.sol_h5m_tag = sol_h5m_tag

        self.logger = logger
        if self.logger == None or not self.logger.hasHandlers():
            self.logger = log.init()

        self.Surfaces = []
        self.Components = []

        try:
            assert (self.repeat + 1) * self.build['phi_list'][-1] <= 360.0, (
                'Total toroidal extent requested with repeated geometry '
                'exceeds 360 degrees. Please examine phi_list and the repeat '
                'parameter.'
            )
        except AssertionError as e:
            self.logger.error(e.args[0])
            raise e

        if self.build['wall_s'] != 1.0:
            self.build['radial_build'] = {
                'sol': {
                    'thickness_matrix': np.zeros((
                        len(self.build['phi_list']),
                        len(self.build['theta_list'])
                    ))
                },
                **self.build['radial_build']
            }
            if self.sol_h5m_tag:
                self.build['radial_build']['sol']['h5m_tag'] = self.sol_h5m_tag

        self.build['radial_build'] = {
            'plasma': {
                'thickness_matrix': np.zeros((
                    len(self.build['phi_list']),
                    len(self.build['theta_list'])
                )),
                'h5m_tag': self.plasma_h5m_tag
            },
            **self.build['radial_build']
        }
        if self.plasma_h5m_tag:
            self.build['radial_build']['sol']['h5m_tag'] = self.plasma_h5m_tag

        self.phi_list = expand_ang_list(self.build['phi_list'], self.num_phi)
        self.theta_list = expand_ang_list(
            self.build['theta_list'], self.num_theta
        )

    def populate_surfaces(self):
        '''Populates Surface class objects representing the outer surface of
        each component specified in the radial build.
        '''
        offset_mat = np.zeros((
            len(self.build['phi_list']), len(self.build['theta_list'])
        ))

        for name, layer_data in self.build['radial_build'].items():
            try:
                assert np.all(np.array(layer_data['thickness_matrix']) >= 0), (
                    'Component thicknesses must be greater than or equal to 0. '
                    'Check thickness inputs for negative values.'
                )
            except AssertionError as e:
                self.logger.error(e.args[0])
                raise e

            if name == 'plasma':
                s = 1.0
            else:
                s = self.build['wall_s']

            if 'h5m_tag' not in layer_data:
                layer_data['h5m_tag'] = name

            offset_mat += np.array(layer_data['thickness_matrix'])
            interpolated_offset_mat = self.interpolate_offset_matrix(
                offset_mat
            )

            self.Surfaces.append(
                Surface(
                    self.vmec, s, self.theta_list, self.phi_list,
                    interpolated_offset_mat, self.scale
                )
            )

        [surface.populate_ribs() for surface in self.Surfaces]

    def interpolate_offset_matrix(self, offset_mat):
        '''Interpolates total offset for expanded angle lists using cubic spline
        interpolation.
        '''
        interpolator = RegularGridInterpolator(
            (self.build['phi_list'], self.build['theta_list']),
            offset_mat,
            method='cubic'
        )

        interpolated_offset_mat = np.array([
            [interpolator([np.rad2deg(phi), np.rad2deg(theta)])[0]
             for theta in self.theta_list]
            for phi in self.phi_list
        ])

        return interpolated_offset_mat

    def calculate_loci(self):
        '''Calls calculate_loci method in Surface class for each component
        specified in the radial build.
        '''
        self.logger.info(f'Computing point cloud...')
        [surface.calculate_loci() for surface in self.Surfaces]

    def generate_components(self):
        '''Constructs a CAD solid for each component specified in the radial
        build by cutting the interior surface solid from the outer surface
        solid for a given component.
        '''
        self.logger.info(f'Constructing in-vessel components...')

        interior_surface = None

        segment_angles = np.linspace(
            self.build['phi_list'][-1],
            self.repeat * self.build['phi_list'][-1],
            num=self.repeat
        )

        for surface in self.Surfaces:
            outer_surface = surface.generate_surface()

            if interior_surface is not None:
                segment = outer_surface.cut(interior_surface)
            else:
                segment = outer_surface

            component = segment

            for angle in segment_angles:
                rot_segment = segment.rotate((0, 0, 0), (0, 0, 1), angle)
                component = component.union(rot_segment)

            self.Components.append(component)
            interior_surface = outer_surface

    def get_loci(self):
        '''Returns the set of point-loci defining the outer surfaces of the
        components specified in the radial build.
        '''
        return np.array([surface.get_loci() for surface in self.Surfaces])


class Surface(object):
    '''An object representing a surface formed by lofting across a set of
    "ribs" located at different toroidal planes and offset from a reference
    surface.

    Arguments:
        vmec (object): plasma equilibrium VMEC object from PyStell-UW.
        s (double): the normalized closed flux surface label defining the point
            of reference for offset.
        theta_list (np.array(double)): the set of poloidal angles specified for
            each rib [rad].
        phi_list (np.array(double)): the set of toroidal angles defining the
            plane in which each rib is located [rad].
        offset_mat (np.array(double)): the set of offsets from the surface
            defined by s for each toroidal angle, poloidal angle pair on the
            surface [cm].
        scale (double): a scaling factor between the units of VMEC and [cm].
    '''

    def __init__(
            self,
            vmec,
            s,
            theta_list,
            phi_list,
            offset_mat,
            scale
    ):
        self.vmec = vmec
        self.s = s
        self.theta_list = theta_list
        self.phi_list = phi_list
        self.offset_mat = offset_mat
        self.scale = scale

        self.surface = None

    def populate_ribs(self):
        '''Populates Rib class objects for each toroidal angle specified in
        the surface.
        '''
        self.Ribs = [
            Rib(
                self.vmec, self.s, self.theta_list, phi, self.offset_mat[i, :],
                self.scale
            )
            for i, phi in enumerate(self.phi_list)
        ]

    def calculate_loci(self):
        '''Calls calculate_loci method in Rib class for each rib in the surface.
        '''
        [rib.calculate_loci() for rib in self.Ribs]

    def generate_surface(self):
        '''Constructs a surface by lofting across a set of rib splines.
        '''
        if not self.surface:
            self.surface = cq.Solid.makeLoft(
                [rib.generate_rib() for rib in self.Ribs]
            )

        return self.surface

    def get_loci(self):
        '''Returns the set of point-loci defining the ribs in the surface.
        '''
        return np.array([rib.rib_loci() for rib in self.Ribs])


class Rib(object):
    '''An object representing a curve formed by interpolating a spline through
    a set of points located in the same toroidal plane but differing poloidal
    angles and offset from a reference curve.

    Arguments:
        vmec (object): plasma equilibrium VMEC object from PyStell-UW.
        s (double): the normalized closed flux surface label defining the point
            of reference for offset.
        phi (np.array(double)): the toroidal angle defining the plane in which
            the rib is located [rad].
        theta_list (np.array(double)): the set of poloidal angles specified for
            the rib [rad].
        offset_list (np.array(double)): the set of offsets from the curve
            defined by s for each toroidal angle, poloidal angle pair in the rib
            [cm].
        scale (double): a scaling factor between the units of VMEC and [cm].
    '''

    def __init__(
            self,
            vmec,
            s,
            theta_list,
            phi,
            offset_list,
            scale
    ):
        self.vmec = vmec
        self.s = s
        self.theta_list = theta_list
        self.phi = phi
        self.offset_list = offset_list
        self.scale = scale

    def calculate_loci(self):
        '''Generates Cartesian point-loci for stellarator rib.
        '''
        self.rib_loci = self.vmec2xyz()

        if not np.all(self.offset_list == 0):
            self.rib_loci += (
                self.offset_list[:, np.newaxis] * self.normals()
            )

    def vmec2xyz(self, poloidal_offset=0):
        '''Return an N x 3 NumPy array containing the Cartesian coordinates of
        the points at this toroidal angle and N different poloidal angles, each
        offset slightly.

        Arguments:
            poloidal_offset (double) : some offset to apply to the full set of
                poloidal angles for evaluating the location of the Cartesian
                points (defaults to 0).
        '''
        return self.scale * np.array(
            [
                self.vmec.vmec2xyz(self.s, theta, self.phi)
                for theta in (self.theta_list + poloidal_offset)
            ]
        )

    def normals(self):
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

        tangent = next_pt_loci - self.rib_loci

        plane_norm = np.array([-np.sin(self.phi), np.cos(self.phi), 0])

        norm = np.cross(plane_norm, tangent)

        return normalize(norm)

    def generate_rib(self):
        '''Constructs component rib by constructing a spline connecting all
        specified Cartesian point-loci.
        '''
        rib_loci = [cq.Vector(tuple(r)) for r in self.rib_loci]
        spline = cq.Edge.makeSpline(rib_loci).close()
        rib_spline = cq.Wire.assembleEdges([spline]).close()

        return rib_spline


def parse_args():
    '''Parser for running as a script.
    '''
    parser = argparse.ArgumentParser(prog='invessel_components')

    parser.add_argument('filename', help='YAML file defining this case')

    return parser.parse_args()


def read_yaml_src(filename):
    '''Read YAML file describing the stellarator in-vessel components and
    extract all data.
    '''
    with open(filename) as yaml_file:
        all_data = yaml.safe_load(yaml_file)

    # Extract data to define in-vessel components
    return (
        all_data['plasma_eq'], all_data['build'], all_data['repeat'],
        all_data['num_phi'], all_data['num_theta'],
        all_data['export']['plasma_h5m_tag'], all_data['export']['sol_h5m_tag'],
        all_data['logger']
    )


def invessel_components():
    '''Main method when run as a command line script.
    '''
    args = parse_args()

    (
        plasma_eq, build, repeat, num_phi, num_theta, plasma_h5m_tag,
        sol_h5m_tag, logger
    ) = read_yaml_src(args.filename)

    vmec = read_vmec.vmec_data(plasma_eq)

    invessel_components = InVesselBuild(
        vmec, build, repeat, num_phi, num_theta, m2cm, plasma_h5m_tag,
        sol_h5m_tag, logger
    )
    invessel_components.populate_surfaces()
    invessel_components.calculate_loci()
    invessel_components.generate_components()


if __name__ == "__main__":
    invessel_components()
