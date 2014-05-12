import numbers

import numpy as np
import pandas as pd


def apply_filter_query(df, filters=None):
    """
    Use the DataFrame.query method to filter a table down to the
    desired rows.

    Parameters
    ----------
    df : pandas.DataFrame
    filters : list of str, optional
        List of filters to apply. Will be joined together with
        ' and ' and passed to DataFrame.query.
        If not supplied no filtering will be done.

    Returns
    -------
    filtered_df : pandas.DataFrame

    """
    if filters:
        query = ' and '.join(filters)
        return df.query(query)
    else:
        return df


def _filterize(name, value):
    """
    Turn a `name` and `value` into a string expression compatible
    the ``DataFrame.query`` method.

    Parameters
    ----------
    name : str
        Should be the name of a column in the table to which the
        filter will be applied.

        A suffix of '_max' will result in a "less than" filter,
        a suffix of '_min' will result in a "greater than or equal to" filter,
        and no recognized suffix will result in an "equal to" filter.
    value : any
        Value side of filter for comparison to column values.

    Returns
    -------
    filter_exp : str

    """
    if name.endswith('_min'):
        name = name[:-4]
        comp = '>='
    elif name.endswith('_max'):
        name = name[:-4]
        comp = '<'
    else:
        comp = '=='

    return '{} {} {!r}'.format(name, comp, value)


def filter_table(table, filter_series, ignore=None):
    """
    Filter a table based on a set of restrictions given in
    Series of column name / filter parameter pairs. The column
    names can have suffixes `_min` and `_max` to indicate
    "less than" and "greater than" constraints.

    Parameters
    ----------
    table : pandas.DataFrame
        Table to filter.
    filter_series : pandas.Series
        Series of column name / value pairs of filter constraints.
        Columns that ends with '_max' will be used to create
        a "less than" filters, columns that end with '_min' will be
        used to create "greater than or equal to" filters.
        A column with no suffix will be used to make an 'equal to' filter.
    ignore : sequence of str, optional
        List of column names that should not be used for filtering.

    Returns
    -------
    filtered : pandas.DataFrame

    """
    ignore = ignore if ignore else set()

    filters = [_filterize(name, val)
               for name, val in filter_series.iteritems()
               if not (name in ignore or
                       (isinstance(val, numbers.Number) and
                        np.isnan(val)))]
    return apply_filter_query(table, filters)


def concat_indexes(indexes):
    """
    Concatenate a sequence of pandas Indexes.

    Parameters
    ----------
    indexes : sequence of pandas.Index

    Returns
    -------
    pandas.Index

    """
    return pd.Index(np.concatenate(indexes))
