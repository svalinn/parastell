import yaml
import argparse

import numpy as np

import cubit

from . import log
from . import cubit_io as cubit_io 
from .utils import normalize, m2cm


class MagnetSet(object):
    """An object representing a set of modular stellarator magnet coils.

    Arguments:
        coils_file (str): path to coil filament data file.
        cross_section (list): coil cross-section definiton. The cross-section
            shape must be either a circle or rectangle. For a circular
            cross-section, the list format is
            ['circle' (str), radius [cm](float)]
            For a rectangular cross-section, the list format is
            ['rectangle' (str), width [cm](float), thickness [cm](float)]
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
        cross_section,
        toroidal_extent,
        logger=None,
        **kwargs
    ):
        
        self.logger = logger
        self.coils_file = coils_file
        self.cross_section = cross_section
        self.toroidal_extent = toroidal_extent

        self.start_line = 3
        self.sample_mod = 1
        self.scale = m2cm
        self.mat_tag = 'magnets'

        for name, value in kwargs.items():
            self.__setattr__(name, value)

        cubit_io.init_cubit()

    @property
    def cross_section(self):
        return self._cross_section
    
    @cross_section.setter
    def cross_section(self, shape):
        self._cross_section = shape
        self._extract_cross_section()

    @property
    def toroidal_extent(self):
        return self._toroidal_extent
    
    @toroidal_extent.setter
    def toroidal_extent(self, angle):
        self._toroidal_extent = np.deg2rad(angle)
        if self._toroidal_extent > 360.0:
            e = ValueError(
                'Toroidal extent cannot exceed 360.0 degrees.'
            )
            self._logger.error(e.args[0])
            raise e

    @property
    def logger(self):
        return self._logger
    
    @logger.setter
    def logger(self, logger_object):
        self._logger = logger_object
        if self._logger == None or not self._logger.hasHandlers():
            self._logger = log.init()

    def _extract_cross_section(self):
        """Extract coil cross-section parameters.
        (Internal function not intended to be called externally)

        Arguments:
            cross_section (list or tuple of str, float, float): coil
                cross-section definition. Note that the cross-section shape
                must be either a circle or rectangle.
                For a circular cross-section, the list format is
                ['circle' (str), radius (float, cm)]
                For a rectangular cross-section, the list format is
                ['rectangle' (str), width (float, cm), thickness (float, cm)]
            logger (object): logger object.

        Returns:
            shape (str): cross-section shape.
            shape_str (str): string to pass to Cubit for cross-section
                generation. For a circular cross-section, the string format is
                '{shape} radius {radius}'
                For a rectangular cross-section, the string format is
                '{shape} width {thickness} height {width}'
            mag_len (float): characteristic length of magnets.
        """
        # Extract coil cross-section shape
        shape = self._cross_section[0]

        # Conditionally extract parameters for circular cross-section
        if shape == 'circle':
            # Check that list format is correct
            if len(self._cross_section) == 1:
                e = ValueError(
                    'Format of list defining circular cross-section must be\n'
                    '[\'circle\' (str), radius (float, cm)]'
                )
                self._logger.error(e.args[0])
                raise e
            elif len(self._cross_section) > 2:
                w = Warning(
                    'More than one length dimension has been defined for '
                    'cross_section. Interpreting the first as the circle\'s'
                    'radius; did you mean to use \'rectangle\'?'
                )
                self._logger.warning(w.args[0])
                raise w
            # Extract parameters
            mag_len = self._cross_section[1]
            # Define string to pass to Cubit for cross-section generation
            shape_str = f'{shape} radius {mag_len}'
        # Conditinally extract parameters for rectangular cross-section
        elif shape == 'rectangle':
            # Check that list format is correct
            if len(self._cross_section) != 3:
                e = ValueError(
                    'Format of list defining rectangular cross-section must \n'
                    'be [\'rectangle\' (str), width (float, cm), thickness '
                    '(float, cm)]'
                )
                self._logger.error(e.args[0])
                raise e
            # Extract parameters
            width = self._cross_section[1]
            thickness = self._cross_section[2]
            # Detemine largest parameter
            mag_len = max(width, thickness)
            # Define string to pass to Cubit for cross-section generation
            shape_str = f'{shape} width {thickness} height {width}'
        # Otherwise, if input string is neither 'circle' nor 'rectangle',
        #  raise an exception
        else:
            e = ValueError(
                'Magnet cross-section must be either a circle or rectangle. '
                'The first entry of the list defining the cross-section must be'
                ' the shape, with the following entries defining the shape'
                'parameters.\n'
                '\n'
                'For a circular cross-section, the list format is\n'
                '[\'circle\' (str), radius (float, cm)]\n'
                '\n'
                'For a rectangular cross-section, the list format is\n'
                '[\'rectangle\' (str), width (float, cm),'
                'thickness (float, cm)]'
            )
            self._logger.error(e.args[0])
            raise e

        self.shape = shape
        self.shape_str = shape_str
        self.mag_len = mag_len

    def _extract_filaments(self):
        """Extracts filament data from magnet coil data file.
        (Internal function not intended to be called externally)
        """
        with open(self.coils_file, 'r') as file:
            data = file.readlines()[self.start_line:]

        coords = []
        filaments = []

        # Ensure that sampling always starts on the first line of each filament
        sample_counter = 0

        for line in data:
            columns = line.strip().split()

            if columns[0] == 'end':
                break

            x = float(columns[0])*self.scale
            y = float(columns[1])*self.scale
            z = float(columns[2])*self.scale

            # Coil current
            s = float(columns[3])

            # s = 0 signals end of filament
            if s != 0:
                if sample_counter % self.sample_mod == 0:
                    coords.append([x, y, z])
                sample_counter += 1
            else:
                coords.append([x, y, z])
                filaments.append(coords)
                sample_counter = 0
                coords = []

        self.filaments = np.array(filaments)

    def _set_average_radial_distance(self):
        """Computes average radial distance of filament points.
        (Internal function not intended to be called externally)

        Arguments:
            filaments (np array of list of list of float): list of filament
                coordinates. Each filament is a list of coordinates.

        Returns:
            average_radial_distance (float): average radial distance of
                magnets (cm).
        """
        average_radial_distance = np.square(
            self.filaments[:, :, 0]) + np.square(self.filaments[:, :, 1])
        average_radial_distance = np.sqrt(average_radial_distance)
        average_radial_distance = np.average(average_radial_distance)

        self.average_radial_distance = average_radial_distance

    def _set_filtered_filaments(self):
        """Cleans filament data such that only filaments within the toroidal
        extent of the model are included and filaments are sorted by toroidal
        angle.
        (Internal function not intended to be called externally)

        Arguments:
            filaments (np array of list of list of float): list of filament
                coordinates. Each filament is a list of coordinates.
            r_avg (float): average radial distance of magnets (cm).
            mag_len (float): characteristic length of magnets.

        Returns:
            filtered_filaments (list of list of list of float): sorted list 
            of filament coordinates.
        """
        # Initialize data for filaments within toroidal extent of model
        reduced_fils = []
        # Initialize list of filament centers of mass for those within toroidal
        # extent of model
        com_list = []

        # Define tolerance of toroidal extent to account for width of coils
        # Multiply by factor of 2 to be conservative
        tol = 2*np.arctan2(self.mag_len, self.average_radial_distance)

        # Compute lower and upper bounds of toroidal extent within tolerance
        min_rad = 2*np.pi - tol
        max_rad = self._toroidal_extent + tol

        for fil in self.filaments:
            # Compute filament center of mass
            com = np.average(fil, axis=0)
            # Compute toroidal angle of each point in filament
            phi_pts = np.arctan2(fil[:, 1], fil[:, 0])
            # Ensure angles are positive
            phi_pts = (phi_pts + 2*np.pi) % (2*np.pi)
            # Compute bounds of toroidal extent of filament
            min_phi = np.min(phi_pts)
            max_phi = np.max(phi_pts)

            # Determine if filament toroidal extent overlaps with that of model
            if (
                (min_phi >= min_rad or min_phi <= max_rad) or
                (max_phi >= min_rad or max_phi <= max_rad)
            ):
                reduced_fils.append(fil)
                com_list.append(com)

        reduced_fils = np.array(reduced_fils)
        com_list = np.array(com_list)

        # Compute toroidal angles of filament centers of mass
        phi_arr = np.arctan2(com_list[:, 1], com_list[:, 0])
        phi_arr = (phi_arr + 2*np.pi) % (2*np.pi)

        # Sort filaments by toroidal angle
        self.filtered_filaments = [
            x for _, x in sorted(zip(phi_arr, reduced_fils))]

    def _cut_magnets(self, volume_ids):
        """Cleanly cuts the magnets at the planes defining the toriodal extent.
        (Internal function not intended to be called externally)

        Arguments:
            volume_ids (list): volume ids corresponding to each magnet volume

        Returns:
            volume_ids (range): new volume ids corresponding to magnet volumes
                following cutting operation
        """
        # Define sweeping surface width
        # Multiply by factor of 2 to be conservative
        rec_width = 2*self.average_radial_distance

        cubit.cmd(f'create surface rectangle width {rec_width} yplane')
        surf_id = cubit.get_last_id("surface")

        # Shift surface to positive x axis
        cubit.cmd(f'move Surface {surf_id} x {rec_width/2}')

        # Revolve surface to create wedge spanning toroidal extent
        cubit.cmd(
            (f'sweep surface {surf_id} zaxis angle '
             f'{np.rad2deg(self._toroidal_extent)}')
        )
        sweep_id = cubit.get_last_id("volume")

        # Remove magnets and magnet portions not within toroidal extent
        cubit.cmd(
            'intersect volume ' + ' '.join(str(i) for i in volume_ids)
            + f' {sweep_id}'
        )

        # Renumber volume ids from 1 to N
        cubit.cmd('compress all')

        # Extract new volume ids
        volume_ids = cubit.get_entities('volume')

        return volume_ids
    
    def build_magnet_coils(self):
        """Builds each filament in self.filtered_filaments in cubit, then cuts
        to the toroidal extent using self._cut_magnets().
        """
        self._logger.info('Constructing magnet coils...')

        self._extract_filaments()
        self._set_average_radial_distance()
        self._set_filtered_filaments()

        self.magnet_coils = [
            MagnetCoil(filament, self.shape, self.shape_str)
            for filament in self.filtered_filaments
        ]

        volume_ids = []

        for magnet_coil in self.magnet_coils:
            volume_id = magnet_coil.create_magnet()
            volume_ids.append(volume_id)

        volume_ids = self._cut_magnets(volume_ids)

        self.volume_ids = volume_ids

    def export_step(self, filename='magnets', export_dir='', **kwargs):
        """Export CAD solids as a STEP file via Coreform Cubit.

        Arguments:
            filename (str): name of STEP output file, excluding '.step'
                extension (optional, defaults to 'magnets').
            export_dir (str): directory to which to export the STEP output file
                (optional, defaults to empty string).
        """
        self._logger.info('Exporting STEP file for magnet coils...')

        self.step_filename = filename
        cubit_io.export_step_cubit(filename=filename, export_dir=export_dir)

    def mesh_magnets(self):
        """Creates tetrahedral mesh of magnet volumes via Coreform Cubit.
        """
        self._logger.info('Generating tetrahedral mesh of magnet coils...')
        
        for vol in self.volume_ids:
            cubit.cmd(f'volume {vol} scheme tetmesh')
            cubit.cmd(f'mesh volume {vol}')
    
    def export_mesh(self, filename='magnet_mesh', export_dir='', **kwargs):
        """Creates tetrahedral mesh of magnet volumes and exports H5M format
        via Coreform Cubit and  MOAB.

        Arguments:
            filename (str): name of H5M output file, excluding '.h5m' extension
                (optional, defaults to 'magnet_mesh').
            export_dir (str): directory to which to export the H5M output file
                (optional, defaults to empty string).
        """
        self._logger.info('Exporting mesh H5M file for magnet coils...')
        
        cubit_io.export_mesh_cubit(filename=filename, export_dir=export_dir)


class MagnetCoil(object):
    """An object representing a single modular stellarator magnet coil.

    Arguments:
        filament (np.ndarray(double)): set of Cartesian coordinates defining
            magnet filament location.
        shape (str): shape of coil cross-section.
        shape_str (str): string defining cross-section shape for Coreform Cubit.
    """

    def __init__(
        self,
        filament,
        shape,
        shape_str
    ):
        
        self.filament = filament
        self.shape = shape
        self.shape_str = shape_str

    def _orient_rectangle(
        self, path_origin, surf_id, t_vec, norm, rot_axis, rot_ang_norm
    ):
        """Orients rectangular cross-section in the normal plane such that its
        thickness direction faces the origin.
        (Internal function not intended to be called externally)

        Arguments:
            path_origin (int): index of initial point in filament path.
            surf_id (int): index of cross-section surface.
            t_vec (list of float): cross-section thickness vector.
            norm (list of float): cross-section normal vector.
            rot_axis (list of float): axis about which to rotate the
                cross-section.
            rot_ang_norm (float): angle by which cross-section was rotated to
                align its normal with the initial point tangent (deg).
        """
        # Determine orientation of thickness vector after cross-section was
        # oriented along filament origin tangent

        # Compute part of thickness vector parallel to rotation axis
        t_vec_par = normalize(np.inner(t_vec, rot_axis)*rot_axis)
        # Compute part of thickness vector orthogonal to rotation axis
        t_vec_perp = normalize(t_vec - t_vec_par)

        # Compute vector othogonal to both rotation axis and orthogonal
        # part of thickness vector
        orth = normalize(np.cross(rot_axis, t_vec_perp))

        # Determine part of rotated vector parallel to original
        rot_par = np.cos(rot_ang_norm)
        # Determine part of rotated vector orthogonal to original
        rot_perp = np.sin(rot_ang_norm)

        # Compute orthogonal part of thickness vector after rotation
        t_vec_perp_rot = rot_par*t_vec_perp + rot_perp*orth
        # Compute thickness vector after rotation
        t_vec_rot = normalize(t_vec_perp_rot + t_vec_par)

        # Orient cross-section in its plane such that it faces the global origin

        # Extract initial path point
        pos = cubit.vertex(path_origin).coordinates()

        # Project position vector onto cross-section
        pos_proj = normalize(pos - np.inner(pos, norm)*norm)

        # Compute angle by which to rotate cross-section such that it faces the
        # origin
        rot_ang_orig = np.arccos(np.inner(pos_proj, t_vec_rot))

        # Re-orient rotated cross-section such that thickness vector faces
        # origin
        cubit.cmd(
            f'rotate Surface {surf_id} angle {np.rad2deg(rot_ang_orig)} about '
            'origin 0 0 0 direction ' + ' '.join(str(i) for i in norm)
        )
    
    def create_magnet(self):
        """Creates magnet coil volumes in cubit.

        Returns:
            volume_id (int): magnet volume ids in cubit
        """
        # Cross-section inititally populated with thickness vector
        # oriented along x axis
        t_vec = np.array([1, 0, 0])

        # Create cross-section for sweep
        cubit.cmd(f'create surface ' + self.shape_str + ' zplane')

        # Store cross-section index
        cs_id = cubit.get_last_id("surface")
        # Cross-section initially populated with normal oriented along z
        # axis
        cs_axis = np.array([0, 0, 1])

        # Initialize path list
        path = []

        # Create vertices in filament path
        for x, y, z in self.filament:
            cubit.cmd(f'create vertex {x} {y} {z}')
            path += [cubit.get_last_id("vertex")]

        # Ensure final vertex in path is the same as the first
        path += [path[0]]

        cubit.cmd(
            f'create curve spline location vertex ' +
            ' '.join(str(i) for i in path)
        )
        curve_id = cubit.get_last_id("curve")

        # Define new surface normal vector as that pointing between path
        # points adjacent to initial point

        # Extract path points adjacent to initial point
        next_pt = np.array(cubit.vertex(path[1]).coordinates())
        last_pt = np.array(cubit.vertex(path[-2]).coordinates())
        # Compute direction in which to align surface normal
        tang = normalize(np.subtract(next_pt, last_pt))

        # Define axis and angle of rotation to orient cross-section along
        # defined normal

        # Define axis of rotation as orthogonal to both z axis and surface
        # normal
        rot_axis = normalize(np.cross(cs_axis, tang))
        # Compute angle by which to rotate cross-section to orient along
        # defined surface normal
        rot_ang_norm = np.arccos(np.inner(cs_axis, tang))

        # Copy cross-section for sweep
        cubit.cmd(f'surface {cs_id} copy')
        surf_id = cubit.get_last_id("surface")

        # Orient cross-section along defined normal
        cubit.cmd(
            f'rotate Surface {surf_id} angle {np.rad2deg(rot_ang_norm)} about '
            'origin 0 0 0 direction ' + ' '.join(str(i) for i in rot_axis)
        )

        # Conditionally orients rectangular cross-section
        if self.shape == 'rectangle':
            self._orient_rectangle(
                path[0], surf_id, t_vec, tang, rot_axis, rot_ang_norm
            )

        # Move cross-section to initial path point
        cubit.cmd(f'move Surface {surf_id} location vertex {path[0]}')

        # Sweep cross-section to create magnet coil
        cubit.cmd(
            f'sweep surface {surf_id} along curve {curve_id} '
            f'individual'
        )
        volume_id = cubit.get_last_id("volume")

        # Delete extraneous curves and vertices
        cubit.cmd(f'delete curve {curve_id}')
        cubit.cmd('delete vertex all')

        # Delete original cross-section
        cubit.cmd(f'delete surface {cs_id}')

        return volume_id


def parse_args():
    """Parser for running as a script
    """
    parser = argparse.ArgumentParser(prog='magnet_coils')

    parser.add_argument(
        'filename', help='YAML file defining ParaStell magnet configuration'
    )

    return parser.parse_args()


def read_yaml_config(filename):
    """Read YAML file describing the stellarator magnet configuration and
    extract all data.
    """
    with open(filename) as yaml_file:
        all_data = yaml.safe_load(yaml_file)

    return all_data['magnet_coils']


def generate_magnet_set():
    """Main method when run as command line script.
    """
    args = parse_args()

    magnet_coils_dict = read_yaml_config(args.filename)

    magnet_set = MagnetSet(**magnet_coils_dict)

    magnet_set.build_magnet_coils()
    magnet_set.export_step(**magnet_coils_dict)

    if magnet_coils_dict['export_mesh']:
        magnet_set.export_mesh(**magnet_coils_dict)


if __name__ == '__main__':
    generate_magnet_set()
