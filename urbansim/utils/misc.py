"""
Utilities used within urbansim that don't yet have a better home.

"""
from __future__ import print_function

import os

import numpy as np
import pandas as pd
import toolz


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


def runs_dir():
    """
    Return the directory for the run output.
    """
    return _mkifnotexists("runs")


def models_dir():
    """
    Return the directory for the model configuration files (used by the
    website).
    """
    return _mkifnotexists("configs")


def charts_dir():
    """
    Return the directory for the chart configuration files (used by the
    website).
    """
    return _mkifnotexists("web/charts")


def maps_dir():
    """
    Return the directory for the map configuration files (used by the
    website).
    """
    return _mkifnotexists("web/maps")


def simulations_dir():
    """
    Return the directory for the simulation configuration files (used by the
    website).
    """
    return _mkifnotexists("web/simulations")


def reports_dir():
    """
    Return the directory for the report configuration files (used by the
    website).
    """
    return _mkifnotexists("web/reports")


def edits_dir():
    """
    Return the directory for the editable files (used by the
    website).
    """
    return _mkifnotexists("")


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
    except:
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
    travel_data["attr"] = attr[travel_data.to_zone_id].values
    return travel_data.groupby(level=0).attr.apply(agg)


def reindex(series1, series2):
    """
    This reindexes the first series by the second series.  This is an extremely
    common operation that does not appear to  be in Pandas at this time.
    If anyone knows of an easier way to do this in Pandas, please inform the
    UrbanSim developers.

    The canonical example would be a parcel series which has an index which is
    parcel_ids and a value which you want to fetch, let's say it's land_area.
    Another dataset, let's say of buildings has a series which indicate the
    parcel_ids that the buildings are located on, but which does not have
    land_area.  If you pass parcels.land_area as the first series and
    buildings.parcel_id as the second series, this function returns a series
    which is indexed by buildings and has land_area as values and can be
    added to the buildings dataset.

    In short, this is a join on to a different table using a foreign key
    stored in the current table, but with only one attribute rather than
    for a full dataset.

    This is very similar to the pandas "loc" function or "reindex" function,
    but neither of those functions return the series indexed on the current
    table.  In both of those cases, the series would be indexed on the foreign
    table and would require a second step to change the index.
    """

    # turns out the merge is much faster than the .loc below
    df = pd.merge(pd.DataFrame({"left": series2}),
                  pd.DataFrame({"right": series1}),
                  left_on="left",
                  right_index=True,
                  how="left")
    return df.right

    # return pd.Series(series1.loc[series2.values].values, index=series2.index)


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


naics_d = {
    11: 'Agriculture',
    21: 'Mining',
    22: 'Utilities',
    23: 'Construction',
    31: 'Manufacturing1',
    32: 'Manufacturing2',
    33: 'Manufacturing3',
    42: 'Wholesale',
    44: 'Retail1',
    45: 'Retail2',
    48: 'Transportation',
    49: 'Warehousing',
    51: 'Information',
    52: 'Finance and Insurance',
    53: 'Real Estate',
    54: 'Professional',
    55: 'Management',
    56: 'Administrative',
    61: 'Educational',
    62: 'Health Care',
    71: 'Arts',
    72: 'Accomodation and Food',
    81: 'Other',
    92: 'Public',
    99: 'Unknown'
}


def naicsname(val):
    """
    This function maps NAICS (job codes) from number to name.
    """
    return naics_d[val]


def numpymat2df(mat):
    """
    Sometimes (though not very often) it is useful to convert a numpy matrix
    which has no column names to a Pandas dataframe for use of the Pandas
    functions.  This method converts a 2D numpy matrix to Pandas dataframe
    with default column headers.

    Parameters
    ----------
    mat : The numpy matrix

    Returns
    -------
    A pandas dataframe with the same data as the input matrix but with columns
    named x0,  x1, ... x[n-1] for the number of columns.
    """
    return pd.DataFrame(
        dict(('x%d' % i, mat[:, i]) for i in range(mat.shape[1])))


def df64bitto32bit(tbl):
    """
    Convert a Pandas dataframe from 64 bit types to 32 bit types to save
    memory or disk space.

    Parameters
    ----------
    tbl : The dataframe to convert

    Returns
    -------
    The converted dataframe
    """
    newtbl = pd.DataFrame(index=tbl.index)
    for colname in tbl.columns:
        newtbl[colname] = series64bitto32bit(tbl[colname])
    return newtbl


def series64bitto32bit(s):
    """
    Convert a Pandas series from 64 bit types to 32 bit types to save
    memory or disk space.

    Parameters
    ----------
    s : The series to convert

    Returns
    -------
    The converted series
    """
    if s.dtype == np.float64:
        return s.astype('float32')
    elif s.dtype == np.int64:
        return s.astype('int32')
    return s


def _pandassummarytojson(v, ndigits=3):
    return {i: round(float(v.ix[i]), ndigits) for i in v.index}


def pandasdfsummarytojson(df, ndigits=3):
    """
    Convert the result of a

    Parameters
    ----------
    df : The result of a Pandas describe operation.
    ndigits : int, optional - The number of significant digits to round to.

    Returns
    -------
    A json object which captures the describe.  Keys are field names and
    values are dictionaries with all of the indexes returned by the Pandas
    describe.
    """
    df = df.transpose()
    return {k: _pandassummarytojson(v, ndigits) for k, v in df.iterrows()}


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
    foundcols = toolz.reduce(lambda x, y: x.union(y), (set(v) for v in colmap.values()))
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
    foundcols = toolz.reduce(lambda x, y: x.union(y), (set(t.columns) for t in tables))
    return list(columns.intersection(foundcols))
