"""
Utilities for doing IO to YAML files.

"""
import os

import numpy as np

import yaml
from collections import OrderedDict


# default key ordering for human-readable model configuration dumps
_DEFAULT_ORDER = [
    'name', 'model_type', 'segmentation_col', 'fit_filters',
    'predict_filters',
    'choosers_fit_filters', 'choosers_predict_filters',
    'alts_fit_filters', 'alts_predict_filters',
    'interaction_predict_filters',
    'choice_column', 'sample_size', 'estimation_sample_size',
    'prediction_sample_size',
    'model_expression', 'ytransform', 'min_segment_size',
    'default_config', 'models', 'coefficients', 'fitted']


def series_to_yaml_safe(series, ordered=False):
    """
    Convert a pandas Series to a dict that will survive YAML serialization
    and re-conversion back to a Series.

    Parameters
    ----------
    series : pandas.Series
    ordered: bool, optional, default False
        If True, an OrderedDict is returned.

    Returns
    -------
    safe : dict or OrderedDict

    """
    index = [to_scalar_safe(value) for value in series.index]
    values = series.values.tolist()

    if ordered:
        return OrderedDict(
            tuple((k, v)) for k, v in zip(index, values))
    else:
        return {i: v for i, v in zip(index, values)}


def frame_to_yaml_safe(frame, ordered=False):
    """
    Convert a pandas DataFrame to a dictionary that will survive
    YAML serialization and re-conversion back to a DataFrame.

    Parameters
    ----------
    frame : pandas.DataFrame
    ordered: bool, optional, default False
        If True, an OrderedDict is returned.

    Returns
    -------
    safe : dict or OrderedDict

    """
    if ordered:
        return OrderedDict(tuple((col, series_to_yaml_safe(series, True))
                                 for col, series in frame.items()))
    else:
        return {col: series_to_yaml_safe(series)
                for col, series in frame.items()}


def to_scalar_safe(obj):
    """
    Convert a numpy data type to a standard python scalar.
    """
    try:
        return obj.item()
    except Exception:
        return obj


def _to_plain(obj):
    """
    Recursively convert OrderedDict (and dict) to plain dict so that
    ``yaml.safe_dump``/``yaml.safe_load`` round-trip without python-specific
    tags. Plain dicts preserve insertion order on Python 3.7+.
    """
    if isinstance(obj, (OrderedDict, dict)):
        return {k: _to_plain(v) for k, v in obj.items()}
    return obj


def ordered_yaml(cfg, order=None):
    """
    Convert a dictionary to a YAML string with preferential ordering
    for some keys. Converted string is meant to be fairly human readable.

    Uses ``yaml.dump(sort_keys=False)`` after arranging keys in the
    requested order, so dict insertion order is preserved on output.

    Parameters
    ----------
    cfg : dict
        Dictionary to convert to a YAML string.
    order: list, optional
        If provided, overrides the default key ordering. An empty list
        preserves the insertion order of ``cfg``.

    Returns
    -------
    str
        Nicely formatted YAML string.

    """
    if order is None:
        order = _DEFAULT_ORDER

    built = {}
    for key in order:
        if key in cfg:
            built[key] = cfg[key]
    for key in cfg:
        if key not in built:
            built[key] = cfg[key]

    return yaml.dump(_to_plain(built), sort_keys=False,
                     default_flow_style=False, indent=4)


def convert_to_yaml(cfg, str_or_buffer):
    """
    Convert a dictionary to YAML and return the string or write it out
    depending on the type of `str_or_buffer`.

    Parameters
    ----------
    cfg : dict or OrderedDict
        Dictionary or OrderedDict to convert.
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
    order = None
    if isinstance(cfg, OrderedDict):
        order = []

    s = ordered_yaml(cfg, order)

    if not str_or_buffer:
        return s
    elif isinstance(str_or_buffer, str):
        with open(str_or_buffer, 'w') as f:
            f.write(s)
    else:
        str_or_buffer.write(s)


def yaml_to_dict(yaml_str=None, str_or_buffer=None, ordered=False):
    """
    Load YAML from a string, file, or buffer (an object with a .read method).
    Parameters are mutually exclusive.

    Parameters
    ----------
    yaml_str : str, optional
        A string of YAML.
    str_or_buffer : str or file like, optional
        File name or buffer from which to load YAML.
    ordered: bool, optional, default False
        If True, an OrderedDict is returned.

    Returns
    -------
    dict
        Conversion from YAML.

    """
    if not yaml_str and not str_or_buffer:
        raise ValueError('One of yaml_str or str_or_buffer is required.')

    if yaml_str:
        d = yaml.safe_load(yaml_str)
    elif isinstance(str_or_buffer, str):
        with open(str_or_buffer) as f:
            d = yaml.safe_load(f)
    else:
        d = yaml.safe_load(str_or_buffer)

    if ordered:
        d = OrderedDict(d) if d is not None else OrderedDict()
    return d
