import yaml
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
        print('Input "vec_list" must be 1-D or 2-D NumPy array')


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


def read_yaml_config(filename):
    """Read YAML file describing ParaStell configuration and extract all data.
    """
    with open(filename) as yaml_file:
        all_data = yaml.safe_load(yaml_file)

    return all_data



def construct_kwargs_from_dict(
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
    kwarg_dict = {}
    for name, value in dict.items():
        if name in allowed_kwargs:
            kwarg_dict.update({name: value})
        elif all_kwargs:
            e = ValueError(
                f'{name} is not a supported keyword argument of '
                f'"{fn_name}"'
            )
            logger.error(e.args[0])
            raise e

    return kwarg_dict


def set_kwarg_attrs(
    class_obj, kwargs, allowed_kwargs
):
    """Sets the attributes of a given class object according to a dictionary of
    keyword argument names and corresponding values.

    Arguments:
        class_obj (object): class object.
        kwargs (dict): dictionary of keyword arguments and corresponding values.
        allowed_kwargs (list of str): list of allowed keyword argument names.
    """
    for name, value in kwargs.items():
        if name in allowed_kwargs:
            class_obj.__setattr__(name, value)
        else:
            e = ValueError(
                f'{name} is not a supported keyword argument of '
                f'"{type(class_obj).__name__}"'
            )
            class_obj._logger.error(e.args[0])
            raise e
