"""
Utilities used within urbansim that don't yet have a better home.

"""
from __future__ import print_function

import os
import numpy as np
import pandas as pd


def _mkifnotexists(folder):
    d = os.path.join(os.getenv('DATA_HOME', "."), folder)
    if not os.path.exists(d):
        os.mkdir(d)
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


def reports_dir():
    """
    Return the directory for the report configuration files (used by the
    website).
    """
    return _mkifnotexists("web/reports")


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
    31: 'Manufacturing',
    32: 'Manufacturing',
    33: 'Manufacturing',
    42: 'Wholesale',
    44: 'Retail',
    45: 'Retail',
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
