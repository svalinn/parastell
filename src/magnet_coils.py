import log
import cubit
import numpy as np
from pathlib import Path
import subprocess
import yaml
import argparse
from utils import normalize

m2cm = 100

class MagnetSet(object):
    '''
    A representation of the magnet filaments, with methods for generating
    step files and volumetric meshes

    Parameters:
        magnets (dict): dictionary of magnet parameters including
            {
                'file': path to magnet coil point-locus data file (str),
                'cross_section': coil cross-section definition (list),
                'start': starting line index for data in file (int),
                'sample': sampling modifier for filament points (int). For a
                    user-supplied value of n, sample every n points in list of
                    points in each filament,
                'name': name to use for STEP export (str),
                'h5m_tag': material tag to use in H5M neutronics model (str)
            }
            For the list defining the coil cross-section, the cross-section
            shape must be either a circle or rectangle. For a circular
            cross-section, the list format is
            ['circle' (str), radius (float, cm)]
            For a rectangular cross-section, the list format is
            ['rectangle' (str), width (float, cm), thickness (float, cm)]
        toroidal_extent (float): desired toroidal extent of the model (degrees)
        export_dir (str): directory to which to export output files.
        logger (object): logger object (defaults to None). If no logger is
            supplied, a default logger will be instantiated.
    '''

    def __init__(
        self,
        magnets,
        toroidal_extent,
        export_dir,
        logger

    ):
        self.coils_file_path = magnets['file']
        self.start = magnets['start']
        self.sample = magnets['sample']
        self.toroidal_extent = np.deg2rad(toroidal_extent)
        self.cross_section = magnets['cross_section']
        self.export_dir = export_dir

        if logger == None or not logger.hasHandlers():
            self.logger = log.init()
        else:
            self.logger = logger

        self.extract_filaments()
        self.set_average_radial_distance()
        self.extract_cross_section()
        self.set_filtered_filaments()

    def extract_filaments(self):
        # set self.filaments to np array, index filament, loci of filament,
        # xyz of loci
        # each filament is a list consisting of lists of xyz

        with open(self.coils_file_path, 'r') as file:
            data = file.readlines()[self.start:]

        # create lists of [x,y,z] points for each filament

        coords = []
        filaments = []

        # ensure that sampling always starts on the first line of each filament
        sample_counter = 0

        for line in data:

            columns = line.strip().split()

            if columns[0] == 'end':
                break

            x = float(columns[0])*m2cm
            y = float(columns[1])*m2cm
            z = float(columns[2])*m2cm

            # Coil Current
            s = float(columns[3])

            # s==0 signals end of filament
            if s != 0:
                if sample_counter % self.sample == 0:
                    coords.append([x, y, z])
                sample_counter += 1
            else:
                coords.append([x, y, z])
                filaments.append(coords)
                sample_counter = 0
                coords = []

        self.filaments = np.array(filaments)

    def set_average_radial_distance(self):
        """Computes average radial distance of filament points.

        Parameters:
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

    def extract_cross_section(self):
        """Extract coil cross-section parameters

        Parameters:
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
        shape = self.cross_section[0]

        # Conditionally extract parameters for circular cross-section
        if shape == 'circle':
            # Check that list format is correct
            if len(self.cross_section) == 1:
                raise ValueError(
                    'Format of list defining circular cross-section must be\n'
                    '[\'circle\' (str), radius (float, cm)]'
                )
            elif len(self.cross_section) > 2:
                self.logger.warning(
                    'More than one length dimension has been defined for '
                    'cross_section. Interpreting the first as the circle\'s'
                    'radius;'
                    ' did you mean to use \'rectangle\'?'
                )
            # Extract parameters
            mag_len = self.cross_section[1]
            # Define string to pass to Cubit for cross-section generation
            shape_str = f'{shape} radius {mag_len}'
        # Conditinally extract parameters for rectangular cross-section
        elif shape == 'rectangle':
            # Check that list format is correct
            if len(self.cross_section) != 3:
                raise ValueError(
                    'Format of list defining rectangular cross-section must \n'
                    'be [\'rectangle\' (str), width (float, cm), thickness '
                    '(float, cm)]'
                )
            # Extract parameters
            width = self.cross_section[1]
            thickness = self.cross_section[2]
            # Detemine largest parameter
            mag_len = max(width, thickness)
            # Define string to pass to Cubit for cross-section generation
            shape_str = f'{shape} width {thickness} height {width}'
        # Otherwise, if input string is neither 'circle' nor 'rectangle',
        #  raise an exception
        else:
            raise ValueError(
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

        self.shape = shape
        self.shape_str = shape_str
        self.mag_len = mag_len

    def set_filtered_filaments(self):
        """Cleans filament data such that only filaments within the toroidal
        extent of the model are included and filaments are sorted by toroidal
        angle.

        Parameters:
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
        max_rad = self.toroidal_extent + tol

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

    def create_magnet_coils(self):
        """
        Creates a list of MagnetCoil objects within the toroidal extent.

        sets:
            self.magnet_coils (list of MagnetCoil objects): List of objects 
                representing individual magnet filaments
        """

        magnet_coils = []

        for filament in self.filtered_filaments:
            magnet_coil = MagnetCoil(filament,
                                     self.shape,
                                     self.shape_str)
            magnet_coils.append(magnet_coil)

        self.magnet_coils = magnet_coils

        return magnet_coils

    def _cut_magnets(self, volume_ids):
        """
        Cleanly cuts the magnets at the planes defining the toriodal extent

        (Internal function not intended to be called externally)

        Parameters:
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
             f'{np.rad2deg(self.toroidal_extent)}')
        )
        sweep_id = cubit.get_last_id("volume")

        # Remove magnets and magnet portions not within toroidal extent
        cubit.cmd(
            'intersect volume ' + ' '.join(str(i) for i in volume_ids)
            + f' {sweep_id}'
        )

        # renumber volume ids from 1 to N
        cubit.cmd('compress all')

        # extract new volume ids
        volume_ids = cubit.get_entities('volume')

        return volume_ids

    def build_magnet_coils(self):
        """
        Builds each filament in self.filtered_filaments in cubit, then
        cuts to the toroidal extent using self._cut_magnets()
        """

        self.create_magnet_coils()

        volume_ids = []

        for magnet_coil in self.magnet_coils:
            volume_id = magnet_coil.create_magnet()
            volume_ids.append(volume_id)

        volume_ids = self._cut_magnets(volume_ids)

        self.volume_ids = volume_ids

    def mesh_magnets(self):
        """Creates tetrahedral mesh of magnet volumes, exports to
        exodus and converts to h5m
        """
        # Mesh magnet volumes
        for vol in self.volume_ids:
            cubit.cmd(f'volume {vol} scheme tetmesh')
            cubit.cmd(f'mesh volume {vol}')

        self.logger.info('Exporting coil mesh...')

        # Define export paths
        exo_path = Path(self.export_dir) / 'coil_mesh.exo'
        h5m_path = Path(self.export_dir) / 'coil_mesh.h5m'

        # EXODUS export
        cubit.cmd(f'export mesh "{exo_path}" overwrite')

        # Convert EXODUS to H5M
        subprocess.run(f'mbconvert {exo_path} {h5m_path}', shell=True)


class MagnetCoil(object):
    def __init__(
        self,
        filament_,
        shape_,
        shape_str_
    ):
        self.filament = filament_
        self.shape = shape_
        self.shape_str = shape_str_

    def orient_rectangle(self, path_origin, surf_id, t_vec, norm, rot_axis,
                         rot_ang_norm):
        """Orients rectangular cross-section in the normal plane such that its
        thickness direction faces the origin.

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

        # Re-orient rotated cross-section such that thickness vector faces origin
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
            self.orient_rectangle(
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
    """
    Parser for running as a script
    """
    parser = argparse.ArgumentParser(prog='generateMagnetSet')

    parser.add_argument('filename', help='YAML file defining this case')

    return parser.parse_args()

def read_yaml_magnets(filename):
    """
    Read YAML file describing this case and extract data relevant for magnet
    definition.
    """
    with open(filename) as yaml_file:
        all_data =  yaml.safe_load(yaml_file)

    magnets = all_data['magnets']
    toroidal_extent = all_data['toroidal_extent']
    export_dir = all_data['export_dir']
    logger = all_data['logger']
    
    return magnets, toroidal_extent, export_dir

def generate_magnet_set():
    """
    Main method when run as cmd line script
    """
    args = parse_args()

    magnets, toroidal_extent, export_dir = read_yaml_magnets(args.filename)

    magnet_set = MagnetSet(magnets,toroidal_extent,export_dir)

    magnet_set.build_magnet_coils()
    magnet_set.mesh_magnets()

if __name__ == '__main__': generate_magnet_set()
