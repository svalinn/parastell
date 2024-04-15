import argparse
import yaml
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

import cubit
import cadquery as cq
import cad_to_dagmc
import src.pystell.read_vmec as read_vmec

from . import log
from .utils import expand_ang_list, normalize, m2cm


def orient_spline_surfaces(volume_id):
    """Extracts the inner and outer surface IDs for a given ParaStell in-vessel
    component volume in Coreform Cubit.

    Arguments:
        volume_id (int): Cubit volume ID.

    Returns:
        inner_surface_id (int): Cubit ID of in-vessel component inner surface.
        outer_surface_id (int): Cubit ID of in-vessel component outer surface.
    """
    
    surfaces = cubit.get_relatives('volume', volume_id, 'surface')

    spline_surfaces = []
    for surface in surfaces:
        if cubit.get_surface_type(surface) == 'spline surface':
            spline_surfaces.append(surface)

    if len(spline_surfaces) == 1:
        outer_surface_id = spline_surfaces[0]
        inner_surface_id = None
    else:
        # The outer surface bounding box will have the larger maximum XY value
        if (
            cubit.get_bounding_box('surface', spline_surfaces[1])[4] >
            cubit.get_bounding_box('surface', spline_surfaces[0])[4]
        ):
            outer_surface_id = spline_surfaces[1]
            inner_surface_id = spline_surfaces[0]
        else:
            outer_surface_id = spline_surfaces[0]
            inner_surface_id = spline_surfaces[1]

    return inner_surface_id, outer_surface_id


class InVesselBuild(object):
    """Parametrically models fusion stellarator in-vessel components using
    plasma equilibrium data and user-defined parameters. In-vessel component
    geometries are determined by a user-defined radial build, in which
    thickness values are supplied in a grid of toroidal and poloidal angles,
    and plasma equilibrium VMEC data.

    Arguments:
        vmec (object): plasma equilibrium VMEC object as defined by the
            PyStell-UW VMEC reader. Must have a method
            'vmec2xyz(s, theta, phi)' that returns an (x,y,z) coordinate for
            any closed flux surface label, s, poloidal angle, theta, and
            toroidal angle, phi.
        toroidal_angles (array of double): toroidal angles at which radial
            build is specified.This list should always begin at 0.0 and it is
            advised not to extend beyond one stellarator period. To build a
            geometry that extends beyond one period, make use of the 'repeat'
            parameter [deg].
        poloidal_angles (array of double): poloidal angles at which radial
            build is specified. This array should always span 360 degrees [deg].
        wall_s (double): closed flux surface label extrapolation at wall.
        radial_build (dict): dictionary representing the three-dimensional
            radial build of in-vessel components, including
            {
                'component': {
                    'thickness_matrix': 2-D matrix defining component
                        thickness at (toroidal angle, poloidal angle)
                        locations. Rows represent toroidal angles, columns
                        represent poloidal angles, and each must be in the same
                        order provided in toroidal_angles and poloidal_angles
                        [cm](ndarray(double)).
                    'mat_tag': DAGMC material tag for component in DAGMC
                        neutronics model (str, defaults to None). If none is
                        supplied, the 'component' key will be used.
                }
            }.
        logger (object): logger object (defaults to None). If no logger is
            supplied, a default logger will be instantiated.
    """

    def __init__(
            self,
            vmec,
            radial_build,
            logger=None
    ):

        self.vmec = vmec
        self.radial_build = radial_build
        self.logger = logger
        
        self.repeat = 0
        self.num_ribs = int(2*radial_build.toroidal_angles[-1]/3) + 1
        self.num_rib_pts = 67
        self.scale = m2cm

        self.Surfaces = {}
        self.Components = {}
            
        self._phi_list = expand_ang_list(
            self.radial_build.toroidal_angles, self._num_ribs
        )
        self._theta_list = expand_ang_list(
            self.radial_build.poloidal_angles, self._num_rib_pts
        )

    @property
    def vmec(self):
        return self._vmec
    
    @vmec.setter
    def vmec(self, vmec_object):
        self._vmec = vmec_object

    @property
    def logger(self):
        return self._logger
    
    @logger.setter
    def logger(self, logger_object):
        self._logger = logger_object
        if self._logger == None or not self._logger.hasHandlers():
            self._logger = log.init()
        
    @property
    def repeat(self):
        return self._repeat
    
    @repeat.setter
    def repeat(self, num):
        self._repeat = num
        if (self._repeat + 1) * self.radial_build.toroidal_angles[-1] > 360.0:
            e = AssertionError(
                'Total toroidal extent requested with repeated geometry '
                'exceeds 360 degrees. Please examine the \'repeat\' parameter '
                'and the \'toroidal_angles\' parameter of \'radial_build\'.'
            )
            self._logger.error(e.args[0])
            raise e

    @property
    def num_ribs(self):
        return self._num_ribs
    
    @num_ribs.setter
    def num_ribs(self, num):
        self._num_ribs = num

    @property
    def num_rib_pts(self):
        return self._num_rib_pts
    
    @num_rib_pts.setter
    def num_rib_pts(self, num):
        self._num_rib_pts = num

    @property
    def scale(self):
        return self._scale
    
    @scale.setter
    def scale(self, value):
        self._scale = value
    
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
                self.radial_build.poloidal_angles
            ),
            offset_mat,
            method='cubic'
        )

        interpolated_offset_mat = np.array([
            [interpolator([np.rad2deg(phi), np.rad2deg(theta)])[0]
             for theta in self._theta_list]
            for phi in self._phi_list
        ])

        return interpolated_offset_mat

    def populate_surfaces(self):
        """Populates Surface class objects representing the outer surface of
        each component specified in the radial build.
        """
        self._logger.info(
            'Populating surface objects for in-vessel components...'
        )

        offset_mat = np.zeros((
            len(self.radial_build.toroidal_angles),
            len(self.radial_build.poloidal_angles)
        ))

        for name, layer_data in self.radial_build.radial_build.items():
            if np.any(np.array(layer_data['thickness_matrix']) < 0):
                e = AssertionError(
                    'Component thicknesses must be greater than or equal to 0. '
                    'Check thickness inputs for negative values.'
                )
                self._logger.error(e.args[0])
                raise e

            if name == 'plasma':
                s = 1.0
            else:
                s = self.radial_build.wall_s

            if 'mat_tag' not in layer_data:
                layer_data['mat_tag'] = name

            offset_mat += np.array(layer_data['thickness_matrix'])
            interpolated_offset_mat = self._interpolate_offset_matrix(
                offset_mat
            )

            self.Surfaces[name] = Surface(
                    self._vmec, s, self._theta_list, self._phi_list,
                    interpolated_offset_mat, self._scale
                )
            
        [surface.populate_ribs() for surface in self.Surfaces.values()]

    def calculate_loci(self):
        """Calls calculate_loci method in Surface class for each component
        specified in the radial build.
        """
        self._logger.info('Computing point cloud for in-vessel components...')

        [surface.calculate_loci() for surface in self.Surfaces.values()]

    def generate_components(self):
        """Constructs a CAD solid for each component specified in the radial
        build by cutting the interior surface solid from the outer surface
        solid for a given component.
        """
        self._logger.info(
            'Constructing CadQuery objects for in-vessel components...'
        )

        interior_surface = None

        segment_angles = np.linspace(
            self.radial_build.toroidal_angles[-1],
            self._repeat * self.radial_build.toroidal_angles[-1],
            num=self._repeat
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
                component = component.union(rot_segment)

            self.Components[name] = component
            interior_surface = outer_surface

    def get_loci(self):
        """Returns the set of point-loci defining the outer surfaces of the
        components specified in the radial build.
        """
        return np.array([
            surface.get_loci() for surface in self.Surfaces.values()
        ])

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

            inner_surface_id, outer_surface_id = (
                orient_spline_surfaces(data['vol_id'])
            )

            # Conditionally skip merging (first iteration only)
            if prev_outer_surface_id is None:
                prev_outer_surface_id = outer_surface_id
            else:
                cubit.cmd(
                    f'merge surface {inner_surface_id} {prev_outer_surface_id}'
                )
                prev_outer_surface_id = outer_surface_id


    def export_step(self, export_dir=''):
        """Export CAD solids as STEP files via CadQuery.

        Arguments:
            export_dir (str): directory to which to export the STEP output files
                (defaults to empty string).
        """
        self._logger.info('Exporting STEP files for in-vessel components...')

        self.export_dir = export_dir

        for name, component in self.Components.items():
            export_path = (
                Path(self.export_dir) / Path(name).with_suffix('.step')
            )
            cq.exporters.export(
                component,
                str(export_path)
            )

    def export_cad_to_dagmc(self, filename='dagmc', export_dir=''):
        """Exports DAGMC neutronics H5M file of ParaStell in-vessel components
        via CAD-to-DAGMC.

        Arguments:
            filename (str): name of DAGMC output file, excluding '.h5m'
                extension (str, defaults to 'dagmc').
            export_dir (str): directory to which to export the DAGMC output file
                (defaults to empty string).
        """
        self._logger.info(
            'Exporting DAGMC neutronics model of in-vessel components...'
        )

        model = cad_to_dagmc.CadToDagmc()

        for name, component in self.Components.items():
            model.add_cadquery_object(
                component,
                material_tags=[self.radial_build.radial_build[name]['mat_tag']]
            )

        export_path = Path(export_dir) / Path(filename).with_suffix('.h5m')

        model.export_dagmc_h5m_file(
            filename=str(export_path)
        )


class Surface(object):
    """An object representing a surface formed by lofting across a set of
    "ribs" located at different toroidal planes and offset from a reference
    surface.

    Arguments:
        vmec (object): plasma equilibrium VMEC object as defined by the
            PyStell-UW VMEC reader. Must have a method
            'vmec2xyz(s, theta, phi)' that returns an (x,y,z) coordinate for
            any closed flux surface label, s, poloidal angle, theta, and
            toroidal angle, phi.
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
    """

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
        """Populates Rib class objects for each toroidal angle specified in
        the surface.
        """
        self.Ribs = [
            Rib(
                self.vmec, self.s, self.theta_list, phi, self.offset_mat[i, :],
                self.scale
            )
            for i, phi in enumerate(self.phi_list)
        ]

    def calculate_loci(self):
        """Calls calculate_loci method in Rib class for each rib in the surface.
        """
        [rib.calculate_loci() for rib in self.Ribs]

    def generate_surface(self):
        """Constructs a surface by lofting across a set of rib splines.
        """
        if not self.surface:
            self.surface = cq.Solid.makeLoft(
                [rib.generate_rib() for rib in self.Ribs]
            )

        return self.surface

    def get_loci(self):
        """Returns the set of point-loci defining the ribs in the surface.
        """
        return np.array([rib.rib_loci() for rib in self.Ribs])


class Rib(object):
    """An object representing a curve formed by interpolating a spline through
    a set of points located in the same toroidal plane but differing poloidal
    angles and offset from a reference curve.

    Arguments:
        vmec (object): plasma equilibrium VMEC object as defined by the
            PyStell-UW VMEC reader. Must have a method
            'vmec2xyz(s, theta, phi)' that returns an (x,y,z) coordinate for
            any closed flux surface label, s, poloidal angle, theta, and
            toroidal angle, phi.
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
    """

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

    def _vmec2xyz(self, poloidal_offset=0):
        """Return an N x 3 NumPy array containing the Cartesian coordinates of
        the points at this toroidal angle and N different poloidal angles, each
        offset slightly.
        (Internal function not intended to be called externally)

        Arguments:
            poloidal_offset (double) : some offset to apply to the full set of
                poloidal angles for evaluating the location of the Cartesian
                points (defaults to 0).
        """
        return self.scale * np.array(
            [
                self.vmec.vmec2xyz(self.s, theta, self.phi)
                for theta in (self.theta_list + poloidal_offset)
            ]
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
        next_pt_loci = self._vmec2xyz(eps)

        tangent = next_pt_loci - self.rib_loci

        plane_norm = np.array([-np.sin(self.phi), np.cos(self.phi), 0])

        norm = np.cross(plane_norm, tangent)

        return normalize(norm)

    def calculate_loci(self):
        """Generates Cartesian point-loci for stellarator rib.
        """
        self.rib_loci = self._vmec2xyz()

        if not np.all(self.offset_list == 0):
            self.rib_loci += (
                self.offset_list[:, np.newaxis] * self._normals()
            )

    def generate_rib(self):
        """Constructs component rib by constructing a spline connecting all
        specified Cartesian point-loci.
        """
        rib_loci = [cq.Vector(tuple(r)) for r in self.rib_loci]
        spline = cq.Edge.makeSpline(rib_loci).close()
        rib_spline = cq.Wire.assembleEdges([spline]).close()

        return rib_spline
    

class RadialBuild(object):
    """
    """

    def __init__(
        self,
        toroidal_angles,
        poloidal_angles,
        wall_s,
        radial_build,
        logger=None
    ):
        self.toroidal_angles = toroidal_angles
        self.poloidal_angles = poloidal_angles
        self.wall_s = wall_s
        self.radial_build = radial_build
        logger = logger

        self.plasma_mat_tag = 'Vacuum'
        self.sol_mat_tag = 'Vacuum'

    @property
    def toroidal_angles(self):
        return self._toroidal_angles
    
    @toroidal_angles.setter
    def toroidal_angles(self, angle_list):
        self._toroidal_angles = angle_list
        if self._toroidal_angles[0] != 0.0:
            e = AssertionError(
                'The first entry in toroidal_angles must be 0.0.'
            )
            self._logger.error(e.args[0])
            raise e
        if self._toroidal_angles[-1] > 360.0:
            e = AssertionError(
                'Toroidal extent cannot exceed 360.0 degrees.'
            )
            self._logger.error(e.args[0])
            raise e

    @property
    def poloidal_angles(self):
        return self._poloidal_angles
    
    @poloidal_angles.setter
    def poloidal_angles(self, angle_list):
        self._poloidal_angles = angle_list
        if self._poloidal_angles[-1] - self._poloidal_angles[0] > 360.0:
            e = AssertionError(
                'Poloidal extent must span exactly 360.0 degrees.'
            )
            self._logger.error(e.args[0])
            raise e

    @property
    def wall_s(self):
        return self._wall_s
    
    @wall_s.setter
    def wall_s(self, s):
        self._wall_s = s
        if self._wall_s < 1.0:
            e = AssertionError(
                'wall_s must be greater than or equal to 1.0.'
            )
            self._logger.error(e.args[0])
            raise e
    
    @property
    def radial_build(self):
        return self._radial_build
    
    @radial_build.setter
    def radial_build(self, build_dict):
        self._radial_build = build_dict
        if self._wall_s != 1.0:
            self._radial_build = {
                'sol': {
                    'thickness_matrix': np.zeros((
                        len(self._toroidal_angles),
                        len(self._poloidal_angles)
                    ))
                },
                **self._radial_build
            }
        self._radial_build = {
            'plasma': {
                'thickness_matrix': np.zeros((
                    len(self._toroidal_angles),
                    len(self._poloidal_angles)
                ))
            },
            **self._radial_build
        }

    @property
    def logger(self):
        return self._logger
    
    @logger.setter
    def logger(self, logger_object):
        self._logger = logger_object
        if self._logger == None or not self._logger.hasHandlers():
            self._logger = log.init()

    @property
    def plasma_mat_tag(self):
        return self._plasma_mat_tag
    
    @plasma_mat_tag.setter
    def plasma_mat_tag(self, mat_tag):
        self._plasma_mat_tag = mat_tag
        self._radial_build['plasma']['mat_tag'] = self._plasma_mat_tag

    @property
    def sol_mat_tag(self):
        return self._sol_mat_tag
    
    @sol_mat_tag.setter
    def sol_mat_tag(self, mat_tag):
        self._sol_mat_tag = mat_tag
        self._radial_build['sol']['mat_tag'] = self._sol_mat_tag


def parse_args():
    """Parser for running as a script.
    """
    parser = argparse.ArgumentParser(prog='invessel_build')

    parser.add_argument(
        'filename',
        help='YAML file defining ParaStell in-vessel component configuration'
    )

    return parser.parse_args()


def read_yaml_config(filename):
    """Read YAML file describing the stellarator in-vessel component
    configuration and extract all data.
    """
    with open(filename) as yaml_file:
        all_data = yaml.safe_load(yaml_file)

    return all_data['vmec_file'], all_data['invessel_build']


def generate_invessel_build():
    """Main method when run as a command line script.
    """
    args = parse_args()

    vmec_file, invessel_build = read_yaml_config(args.filename)

    vmec = read_vmec.VMECData(vmec_file)

    invessel_build = {
        'vmec': vmec,
        **invessel_build
    }

    invessel_components = InVesselBuild(**invessel_build)
    invessel_components.populate_surfaces()
    invessel_components.calculate_loci()
    invessel_components.generate_components()
    invessel_components.export_step(export_dir=invessel_build['export_dir'])

    if invessel_build['export_cad_to_dagmc']:
        invessel_components.export_cad_to_dagmc(
            filename=invessel_build['dagmc_filename'],
            export_dir=invessel_build['export_dir']
        )


if __name__ == "__main__":
    generate_invessel_build()
