"""
Utilities for doing IO to YAML files.

"""
try:
    from itertools import izip as zip
except ImportError:
    pass
import os
import sys
import numpy as np

import yaml
from collections import OrderedDict


if sys.version_info[0] < 3:

    def __represent_long(dumper, data):
        """
        Strips away extraneous long format text. Only applicable
        for py27.

        e.g. !!python/long '14' will be formatted as 14

        """
        return dumper.represent_int(data)

    yaml.add_representer(long, __represent_long)


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
    index = series.index.to_native_types(quoting=True)
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
                                 for col, series in frame.iteritems()))
    else:
        return {col: series_to_yaml_safe(series)
                for col, series in frame.iteritems()}


def to_scalar_safe(obj):
    """
    Convert a numpy data type to a standard python scalar.
    """
    try:
        return obj.item()
    except Exception:
        return obj


def ordered_yaml(cfg, order=None):
    """
    Convert a dictionary to a YAML string with preferential ordering
    for some keys. Converted string is meant to be fairly human readable.

    Parameters
    ----------
    cfg : dict
        Dictionary to convert to a YAML string.
    order: list
        If provided, overrides the default key ordering.

    Returns
    -------
    str
        Nicely formatted YAML string.

    """
    if order is None:
        order = ['name', 'model_type', 'segmentation_col', 'fit_filters',
                 'predict_filters',
                 'choosers_fit_filters', 'choosers_predict_filters',
                 'alts_fit_filters', 'alts_predict_filters',
                 'interaction_predict_filters',
                 'choice_column', 'sample_size', 'estimation_sample_size',
                 'prediction_sample_size',
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

    return '\n'.join(s)


def __represent_ordereddict(dumper, data):
    """
    Allows for OrderedDict to be written out to yaml.

    References:
        https://codedump.io/share/2MLFLtw3wnX7/1/can-pyyaml-dump-dict-items-in-non-alphabetical-order
        http://stackoverflow.com/questions/16782112/can-pyyaml-dump-dict-items-in-non-alphabetical-order

    """
    value = []

    for item_key, item_value in data.items():
        node_key = dumper.represent_data(item_key)
        node_value = dumper.represent_data(item_value)

        value.append((node_key, node_value))

    return yaml.nodes.MappingNode(u'tag:yaml.org,2002:map', value)


yaml.add_representer(OrderedDict, __represent_ordereddict)


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

    # determine which load method to use
    if ordered:
        loader = __ordered_load
    else:
        loader = yaml.safe_load

    if yaml_str:
        d = loader(yaml_str)
    elif isinstance(str_or_buffer, str):
        with open(str_or_buffer) as f:
            d = loader(f)
    else:
        d = loader(str_or_buffer)

    return d


def __ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    """
    Loads yaml into an OrderedDict.

    From:
    https://stackoverflow.com/questions/5121931/in-python-how-can-you-load-yaml-mappings-as-ordereddicts

    """

    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))

    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)
