import math

import numpy as np

m2cm = 100

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
        print('Input \'vec_list\' must be 1-D or 2-D NumPy array')


def expand_ang_list(ang_list, num_ang):
    """Expands list of angles by linearly interpolating according to specified
    number to include in stellarator build.

    Arguments:
        ang_list (list of double): user-supplied list of toroidal or poloidal
            angles (rad).
        num_ang (int): number of angles to include in stellarator build.

    Returns:
        ang_list_exp (list of double): interpolated list of angles (rad).
    """
    ang_list = np.deg2rad(ang_list)

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
