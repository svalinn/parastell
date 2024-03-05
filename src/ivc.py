import argparse
import log
import read_vmec
import cadquery as cq
import numpy as np
import math
from scipy.interpolate import RegularGridInterpolator
import yaml

m2cm = 100


class IVC(object):
    '''Parametrically generates a fusion stellarator model using plasma
    equilibrium data and user-defined parameters. In-vessel component
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

    def populate_data(self):
        '''Constructs Data class object.
        '''
        self.data = Data(
            self.vmec, self.build, self.repeat, self.num_phi,
            self.num_theta, self.scale, self.plasma_h5m_tag, self.sol_h5m_tag,
            self.logger
        )

    def construct_geometry(self):
        '''Constructs CADGeometry class object.
        '''
        self.cad_geometry = CADGeometry(
            self.data, self.repeat, self.logger
        )
        self.components = self.cad_geometry.components


class Data(object):
    '''Generates all stellarator in-vessel component data necessary for CAD
    generation according to user-specification. All user-specified data, the
    actual toroidal and poloidal angles used to build the geometry, and segment
    and total model toroidal angular extent are contained in the Data object.
    In-vessel component outer surface Cartesian point-loci data is contained
    within the Data.radial_build dictionary.

    Arguments:
        vmec (object): plasma equilibrium VMEC object from PyStell-UW.
        build (dict): dictionary defining stellarator build. See Stellarator
            class docstring for more detail.
        repeat (int): number of times to repeat build segment.
        num_phi (int): number of phi geometric cross-sections to make for each
            build segment (defaults to 61).
        num_theta (int): number of points defining the geometric cross-section
            (defaults to 61).
        scale (double): a scaling factor between the units of VMEC and [cm].
        plasma_h5m_tag (str): optional alternate material tag to use for
            plasma. If none is supplied and the plasma is not excluded,
            'plasma' will be used.
        sol_h5m_tag (str): optional alternate material tag to use for
            scrape-off layer. If none is supplied and the scrape-off layer is
            not excluded, 'sol' will be used.
        logger (object): logger object.
    '''

    def __init__(
            self,
            vmec,
            build,
            repeat,
            num_phi,
            num_theta,
            scale,
            plasma_h5m_tag,
            sol_h5m_tag,
            logger
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
                if self.plasma_h5m_tag is not None:
                    layer_data['h5m_tag'] = self.plasma_h5m_tag
                s = 1.0
            else:
                s = self.wall_s

            if name == 'sol':
                if self.sol_h5m_tag is not None:
                    layer_data['h5m_tag'] = self.sol_h5m_tag

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
        '''Return an N x 3 NumPy array containing the Cartesian coordinates of
        the points at this toroidal angle and N different poloidal angles, each
        offset slightly.

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
    '''Builds CAD geometry for stellarator in-vessel components using CadQuery.
    All relevant user-defined parameters and segment and total toroidal angular
    extents are contained in the CADGeometry class object. Component
    parameters, including CadQuery solid, H5M material tag, and volume IDs are
    contained within the CADGeometry.components dictionary.

    Arguments:
        data (object): Data class object.
        repeat (int): number of times to repeat build segment.
        logger (object): logger object.
    '''

    def __init__(
            self,
            data,
            repeat,
            logger
    ):
        self.data = data
        self.repeat = repeat
        self.logger = logger

        self.seg_tor_ext = self.data.seg_tor_ext
        self.tot_tor_ext = self.data.tot_tor_ext

        self.components = self.create_geometry()

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


def parse_args():
    '''Parser for running as a script.
    '''
    parser = argparse.ArgumentParser(prog='ivc')

    parser.add_argument('filename', help='YAML file defining this case')

    return parser.parse_args()


def read_yaml_src(filename):
    '''Read YAML file describing the stellarator in-vessel components and
    extract all data.
    '''
    with open(filename) as yaml_file:
        all_data = yaml.safe_load(yaml_file)

    # Extract data to define source mesh
    return (
        all_data['plasma_eq'], all_data['build'], all_data['repeat'],
        all_data['num_phi'], all_data['num_theta'],
        all_data['export']['plasma_h5m_tag'], all_data['export']['sol_h5m_tag'],
        all_data['logger']
    )


def ivc():
    '''Main method when run as a command line script.
    '''
    args = parse_args()

    (
        plasma_eq, build, repeat, num_phi, num_theta, plasma_h5m_tag,
        sol_h5m_tag, logger
    ) = read_yaml_src(args.filename)

    vmec = read_vmec.vmec_data(plasma_eq)

    invessel_components = IVC(
        vmec, build, repeat, num_phi, num_theta, plasma_h5m_tag,
        sol_h5m_tag, logger
    )
    invessel_components.populate_data()
    invessel_components.construct_geometry()


if __name__ == "__main__":
    ivc()
