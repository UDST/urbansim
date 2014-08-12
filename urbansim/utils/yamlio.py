"""
Utilities for doing IO to YAML files.

"""
import itertools
import os
import numpy as np

import yaml


def series_to_yaml_safe(series):
    """
    Convert a pandas Series to a dict that will survive YAML serialization
    and re-conversion back to a Series.

    Parameters
    ----------
    series : pandas.Series

    Returns
    -------
    safe : dict

    """
    index = series.index.to_native_types()
    values = series.values.tolist()

    return {i: v for i, v in itertools.izip(index, values)}


def frame_to_yaml_safe(frame):
    """
    Convert a pandas DataFrame to a dictionary that will survive
    YAML serialization and re-conversion back to a DataFrame.

    Parameters
    ----------
    frame : pandas.DataFrame

    Returns
    -------
    safe : dict

    """
    return {col: series_to_yaml_safe(series)
            for col, series in frame.iteritems()}


def to_scalar_safe(obj):
    """
    Convert a numpy data type to a standard python scalar.
    """
    try:
        return np.asscalar(obj)
    except:
        return obj


def ordered_yaml(cfg):
    """
    Convert a dictionary to a YAML string with preferential ordering
    for some keys. Converted string is meant to be fairly human readable.

    Parameters
    ----------
    cfg : dict
        Dictionary to convert to a YAML string.

    Returns
    -------
    str
        Nicely formatted YAML string.

    """
    order = ['name', 'model_type', 'segmentation_col', 'fit_filters',
             'predict_filters',
             'choosers_fit_filters', 'choosers_predict_filters',
             'alts_fit_filters', 'alts_predict_filters',
             'interaction_predict_filters',
             'choice_column', 'sample_size', 'estimation_sample_size',
             'model_expression', 'ytransform', 'min_segment_size',
             'default_config', 'models', 'coefficients', 'fitted']

    s = []
    for key in order:
        if key not in cfg:
            continue
        s.append(
            yaml.dump({key: cfg[key]}, default_flow_style=False, indent=4))

    for key in cfg:
        if key in order:
            continue
        s.append(
            yaml.dump({key: cfg[key]}, default_flow_style=False, indent=4))

    return os.linesep.join(s)


def convert_to_yaml(cfg, str_or_buffer):
    """
    Convert a dictionary to YAML and return the string or write it out
    depending on the type of `str_or_buffer`.

    Parameters
    ----------
    cfg : dict
        Dictionary to convert.
    str_or_buffer : None, str, or buffer
        If None: the YAML string will be returned.
        If string: YAML will be saved to a file.
        If buffer: YAML will be written to buffer using the ``.write`` method.

    Returns
    -------
    str or None
        YAML string if `str_or_buffer` is None, otherwise None since YAML
        is written out to a separate destination.

    """
    s = ordered_yaml(cfg)

    if not str_or_buffer:
        return s
    elif isinstance(str_or_buffer, str):
        with open(str_or_buffer, 'w') as f:
            f.write(s)
    else:
        str_or_buffer.write(s)


def yaml_to_dict(yaml_str=None, str_or_buffer=None):
    """
    Load YAML from a string, file, or buffer (an object with a .read method).
    Parameters are mutually exclusive.

    Parameters
    ----------
    yaml_str : str, optional
        A string of YAML.
    str_or_buffer : str or file like, optional
        File name or buffer from which to load YAML.

    Returns
    -------
    dict
        Conversion from YAML.

    """
    if not yaml_str and not str_or_buffer:
        raise ValueError('One of yaml_str or str_or_buffer is required.')

    if yaml_str:
        d = yaml.load(yaml_str)
    elif isinstance(str_or_buffer, str):
        with open(str_or_buffer) as f:
            d = yaml.load(f)
    else:
        d = yaml.load(str_or_buffer)

    return d
