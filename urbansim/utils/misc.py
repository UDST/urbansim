"""
Utilities used within urbansim that don't yet have a better home.

"""
from __future__ import print_function

import os
from functools import reduce

import numpy as np
import pandas as pd


def _mkifnotexists(folder):
    d = os.path.join(os.getenv('DATA_HOME', "."), folder)
    if not os.path.exists(d):
        os.makedirs(d)
    return d


def data_dir():
    """
    Return the directory for the input data.
    """
    return _mkifnotexists("data")


def configs_dir():
    """
    Return the directory for the model configuration files.
    """
    return _mkifnotexists("configs")


def config(fname):
    """
    Return the config path for the file with the given filename.
    """
    return os.path.join(configs_dir(), fname)


def get_run_number():
    """
    Get a run number for this execution of the model system, for
    identifying the output hdf5 files).

    Returns
    -------
    The integer number for this run of the model system.
    """
    try:
        f = open(os.path.join(os.getenv('DATA_HOME', "."), 'RUNNUM'), 'r')
        num = int(f.read())
        f.close()
    except Exception:
        num = 1
    f = open(os.path.join(os.getenv('DATA_HOME', "."), 'RUNNUM'), 'w')
    f.write(str(num + 1))
    f.close()
    return num


def compute_range(travel_data, attr, travel_time_attr, dist, agg=np.sum):
    """
    Compute a zone-based accessibility query using the urbansim format
    travel data dataframe.

    Parameters
    ----------
    travel_data : dataframe
        The dataframe of urbansim format travel data.  Has from_zone_id as
        first index, to_zone_id as second index, and different impedances
        between zones as columns.
    attr : series
        The attr to aggregate.  Should be indexed by zone_id and the values
        will be aggregated.
    travel_time_attr : string
        The column name in travel_data to use as the impedance.
    dist : float
        The max distance to aggregate up to
    agg : function, optional, np.sum by default
        The numpy function to use for aggregation
    """
    travel_data = travel_data.reset_index(level=1)
    travel_data = travel_data[travel_data[travel_time_attr] < dist]
    travel_data["attr"] = attr.reindex(travel_data.to_zone_id, fill_value=0).values
    return travel_data.groupby(level=0).attr.apply(agg)


def fidx(right, left, left_fk=None):
    """
    Re-indexes a series or data frame (right) to align with
    another (left) series or data frame via foreign key relationship.
    The index of the right must be unique.

    Allows for data frame re-indexes and supports re-indexing data
    frames or series with a multi-index.

    Parameters:
    -----------
    right: pandas.DataFrame or pandas.Series
        Series or data frame to re-index from.
    left: pandas.Series or pandas.DataFrame
        Series or data frame to re-index to.
        If a series is provided, its values serve as the foreign keys.
        If a data frame is provided, one or more columns may be used
        as foreign keys, must specify the ``left_fk`` argument to
        specify which column(s) will serve as keys.
    left_fk: optional, str or list of str
        Used when the left is a data frame, specifies the column(s) in
        the left to serve as foreign keys. The specified columns' ordering
        must match the order of the multi-index in the right.

    Returns:
    --------
    pandas.Series or pandas.DataFrame with column(s) from
    right aligned with the left.

    """
    # ensure that we can align correctly
    if not right.index.is_unique:
        raise ValueError("The right's index must be unique!")

    # simpler case:
    # if the left (target) is a single series then just re-index to it
    if isinstance(left_fk, str):
        left = left[left_fk]

    if isinstance(left, pd.Series):
        a = right.reindex(left)
        a.index = left.index
        return a

    # when reindexing using multiple columns (composite foreign key)
    # i.e. the right has a multindex

    # if a series for the right provided, convert to a data frame
    if isinstance(right, pd.Series):
        right = right.to_frame('right')
        right_cols = 'right'
    else:
        right_cols = right.columns

    # do the merge
    return pd.merge(
        left=left,
        right=right,
        left_on=left_fk,
        right_index=True,
        how='left'
    )[right_cols]


def signif(val):
    """
    Convert a statistical significance to its ascii representation - this
    should be the same representation created in R.
    """
    val = abs(val)
    if val > 3.1:
        return '***'
    elif val > 2.33:
        return '**'
    elif val > 1.64:
        return '*'
    elif val > 1.28:
        return '.'
    return ''


def column_map(tables, columns):
    """
    Take a list of tables and a list of column names and resolve which
    columns come from which table.

    Parameters
    ----------
    tables : sequence of _DataFrameWrapper or _TableFuncWrapper
        Could also be sequence of modified pandas.DataFrames, the important
        thing is that they have ``.name`` and ``.columns`` attributes.
    columns : sequence of str
        The column names of interest.

    Returns
    -------
    col_map : dict
        Maps table names to lists of column names.

    """
    if not columns:
        return {t.name: None for t in tables}

    columns = set(columns)
    colmap = {t.name: list(set(t.columns).intersection(columns)) for t in tables}
    foundcols = reduce(lambda x, y: x.union(y), (set(v) for v in colmap.values()))
    if foundcols != columns:
        raise RuntimeError('Not all required columns were found. '
                           'Missing: {}'.format(list(columns - foundcols)))
    return colmap


def column_list(tables, columns):
    """
    Take a list of tables and a list of column names and return the columns
    that are present in the tables.

    Parameters
    ----------
    tables : sequence of _DataFrameWrapper or _TableFuncWrapper
        Could also be sequence of modified pandas.DataFrames, the important
        thing is that they have ``.name`` and ``.columns`` attributes.
    columns : sequence of str
        The column names of interest.

    Returns
    -------
    cols : list
        Lists of column names available in the tables.

    """
    columns = set(columns)
    foundcols = reduce(lambda x, y: x.union(y), (set(t.columns) for t in tables))
    return list(columns.intersection(foundcols))
