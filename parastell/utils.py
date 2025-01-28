import yaml
import tempfile
from pathlib import Path

import numpy as np
import math
from scipy.ndimage import gaussian_filter
from pymoab import core, types


m2cm = 100
m3tocm3 = m2cm * m2cm * m2cm


def downsample_loop(list, sample_mod):
    """Downsamples a list representing a closed loop.

    Arguments:
        list (iterable): closed loop.
        sample_mod (int): sampling modifier.

    Returns:
        (iterable): downsampled closed loop
    """
    return np.append(list[:-1:sample_mod], [list[0]], axis=0)


def enforce_helical_symmetry(matrix):
    """Ensures that a matrix is helically symmetric according to stellarator
    geometry by overwriting certain matrix elements.

    Assumes several qualities about the input matrix:
        - Regular spacing between angles defining matrix
        - Matrix represents a full stellarator period
        - The first row corresponds to a toroidal angle at the beginning of a
          period (i.e., poloidal symmetry is expected)

    Arguments:
        matrix (2-D iterable of float): matrix to be made helically symmetric.

    Returns:
        matrix (2-D iterable of float): helically symmetric matrix.
    """
    num_rows, num_columns = matrix.shape

    # Ensure rows represent closed loops
    matrix[:, -1] = matrix[:, 0]

    # Ensure poloidal symmetry at beginning of period
    matrix[0] = np.concatenate(
        [
            # Ceil and floor ensure middle element of odd sized array is
            # included only once
            matrix[0, : math.ceil(num_columns / 2)],
            np.flip(matrix[0, : math.floor(num_columns / 2)]),
        ]
    )

    # Ensure helical symmetry toroidally and poloidally by mirroring the period
    # about both matrix axes
    flattened_matrix = matrix.flatten()
    flattened_length = len(flattened_matrix)

    first_half = flattened_matrix[: math.ceil(flattened_length / 2)]
    last_half = np.flip(flattened_matrix[: math.floor(flattened_length / 2)])
    flattened_matrix = np.concatenate([first_half, last_half])

    matrix = flattened_matrix.reshape((num_rows, num_columns))

    return matrix


def expand_list(list, num):
    """Expands a list of ordered floats to a total number of entries by
    linearly interpolating between entries, inserting a proportional number of
    new entries between original entries.

    Arguments:
        list (iterable of float): list to be expanded.
        num (int): desired number of entries in expanded list.

    Returns:
        list_exp (iterable of float): expanded list.
    """
    list_exp = []

    init_entry = list[0]
    final_entry = list[-1]
    extent = final_entry - init_entry

    avg_diff = extent / (num - 1)

    for entry, next_entry in zip(list[:-1], list[1:]):
        num_new_entries = int(round((next_entry - entry) / avg_diff))

        # Don't append the last entry in the created linspace to avoid adding
        # it twice when the next created linspace is appended
        list_exp = np.append(
            list_exp,
            np.linspace(entry, next_entry, num=num_new_entries + 1)[:-1],
        )

    list_exp = np.append(list_exp, final_entry)

    return list_exp


def filter_kwargs(
    dict, allowed_kwargs, all_kwargs=False, fn_name=None, logger=None
):
    """Constructs a dictionary of keyword arguments with corresponding values
    for a class method from an input dictionary based on a list of allowable
    keyword argument argument names. Conditionally raises an exception if the
    user flags that the supplied list of allowable arguments should represent
    all keys present in the input dictionary of arguments.

    Arguments:
        dict (dict): dictionary of arguments and corresponding values.
        allowed_kwargs (list of str): list of allowed keyword argument names.
        all_kwargs (bool): flag to indicate whether 'allowed_kwargs' should
            represent all keys present in 'dict' (optional, defaults to False).
        fn_name (str): name of class method (optional, defaults to None). If
            'all_kwargs' is True, a method name should be supplied.
        logger (object): logger object (optional, defaults to None). If
            'all_kwargs' is True, a logger object should be supplied.

    Returns:
        kwarg_dict (dict): dictionary of keyword arguments and values.
    """
    allowed_keys = dict.keys() & allowed_kwargs
    extra_keys = dict.keys() - allowed_kwargs

    if all_kwargs and extra_keys:
        e = ValueError(
            f"{extra_keys} not supported keyword argument(s) of "
            f'"{fn_name}"'
        )
        logger.error(e.args[0])
        raise e

    return {name: dict[name] for name in allowed_keys}


def normalize(vec_list):
    """Normalizes a set of vectors.

    Arguments:
        vec_list (1 or 2D np array): single 1D vector or array of 1D vectors
            to be normalized
    Returns:
        vec_list (np array of same shape as input): single 1D normalized vector
            or array of normalized 1D vectors
    """
    if len(vec_list.shape) == 1:
        return vec_list / np.linalg.norm(vec_list)
    elif len(vec_list.shape) == 2:
        return vec_list / np.linalg.norm(vec_list, axis=1)[:, np.newaxis]
    else:
        print('Input "vec_list" must be 1-D or 2-D NumPy array')


def read_yaml_config(filename):
    """Read YAML file describing ParaStell configuration and extract all data."""
    with open(filename) as yaml_file:
        all_data = yaml.safe_load(yaml_file)

    return all_data


def reorder_loop(list, index):
    """Reorders a list representing a closed loop.

    Arguments:
        list (iterable): closed loop.
        index (int): list index about which to reorder loop.

    Returns:
        (iterable): reordered closed loop.
    """
    return np.concatenate([list[index:], list[1 : index + 1]])


def smooth_matrix(matrix, steps, sigma):
    """Smooths a matrix via Gaussian filtering, without allowing matrix
    elements to increase in value.

    Arguments:
        matrix (2-D iterable of float): matrix to be smoothed.
        steps (int): number of smoothing steps.
        sigma (float): standard deviation for Gaussian kernel.

    Returns:
        smoothed_matrix (2-D iterable of float): smoothed matrix.
    """
    previous_matrix = matrix

    for step in range(steps):
        smoothed_matrix = np.minimum(
            previous_matrix,
            gaussian_filter(
                previous_matrix,
                sigma=sigma,
                mode="wrap",
            ),
        )
        previous_matrix = smoothed_matrix

    return smoothed_matrix


class DAGMCMerger:
    """Class to facilitate renumbering of entities to merge dagmc models

    Arguments:
        mb (PyMOAB Core): PyMOAB core with a DAGMC model loaded. Optional.
    """

    def __init__(self, mb=None):
        self.mb = mb
        if mb is None:
            self.mb = core.Core()
        self.geom_dim_tag = self.mb.tag_get_handle(
            "GEOM_DIMENSION", 1, types.MB_TYPE_INTEGER, types.MB_TAG_DENSE
        )
        self.global_id_tag = self.mb.tag_get_handle(
            "GLOBAL_ID", 1, types.MB_TYPE_INTEGER, types.MB_TAG_DENSE
        )

    def load_file(self, filename):
        self.mb.load_file(filename)

    def get_max_ids(self):
        """Get the maximum id in the dagmc model for each entity type

        returns:
            max_ids (dict): {entity dimension (int): max_id (int)}
        """
        # dim 0: vertex
        # dim 1: edge
        # dim 2: surface
        # dim 3: volume
        # dim 4: group
        max_ids = {dim: 0 for dim in range(5)}
        root_set = self.mb.get_root_set()
        geom_sets = self.mb.get_entities_by_type_and_tag(
            root_set, types.MBENTITYSET, self.geom_dim_tag, [None]
        )

        for entity_set in geom_sets:
            dim = self.mb.tag_get_data(self.geom_dim_tag, entity_set)[0][0]
            gid = self.mb.tag_get_data(self.global_id_tag, entity_set)[0][0]
            max_ids[dim] = max(max_ids[dim], gid)

        return max_ids

    def reassign_global_ids(self, offsets):
        """Renumber entities, starting from the offset value for that dimension

        Arguments:
            offsets (dict): {entity dimension (int): id_offset (int)}
        """
        root_set = self.mb.get_root_set()
        geom_sets = self.mb.get_entities_by_type_and_tag(
            root_set, types.MBENTITYSET, self.geom_dim_tag, [None]
        )

        for entity_set in geom_sets:
            dim = self.mb.tag_get_data(self.geom_dim_tag, entity_set)[0][0]
            current_gid = self.mb.tag_get_data(self.global_id_tag, entity_set)[
                0
            ][0]
            self.mb.tag_set_data(
                self.global_id_tag,
                entity_set,
                current_gid + offsets.get(dim, 0),
            )

    def write_file(self, filename):
        """Save to file

        Arguments:
            filename (str): Path to save the file to. All filetypes supported
                by MOAB are available. Write to h5m for use with DAGMC.
        """
        self.mb.write_file(filename)


def merge_dagmc_files(models_to_merge):
    """Takes a list of dagmc models, and renumbers entity ids such that they
    will no longer clash, allowing the models to be loaded into the same
    PyMOAB core instance, and saved to a single model.

    Arguments:
        models_to_merge (list of PyMOAB core): List of dagmc models to be
            merged.
    Returns:
        merged_mbc (PyMOAB Core): PyMOAB core instance containing the merged
            DAGMC models.
    """
    temp_files = []
    max_ids = {dim: 0 for dim in range(5)}
    for model in models_to_merge:
        merger = DAGMCMerger(model)
        merger.reassign_global_ids(max_ids)
        max_ids = merger.get_max_ids()
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".h5m"
        ) as temp_file:
            temp_filename = temp_file.name
        merger.write_file(temp_filename)
        temp_files.append(temp_filename)

    merged_mbc = core.Core()
    for file in temp_files:
        merged_mbc.load_file(file)

    Path.unlink(temp_filename)
    return merged_mbc
