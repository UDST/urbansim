"""
Utilities used within the ``urbansim.models`` package.

"""
import collections
import logging
import numbers
from StringIO import StringIO
from tokenize import generate_tokens, NAME

import numpy as np
import pandas as pd
import patsy
import toolz

from ..utils.logutil import log_start_finish

logger = logging.getLogger(__name__)


def apply_filter_query(df, filters=None):
    """
    Use the DataFrame.query method to filter a table down to the
    desired rows.

    Parameters
    ----------
    df : pandas.DataFrame
    filters : list of str or str, optional
        List of filters to apply. Will be joined together with
        ' and ' and passed to DataFrame.query. A string will be passed
        straight to DataFrame.query.
        If not supplied no filtering will be done.

    Returns
    -------
    filtered_df : pandas.DataFrame

    """
    with log_start_finish('apply filter query: {!r}'.format(filters), logger):
        if filters:
            if isinstance(filters, str):
                query = filters
            else:
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

    result = '{} {} {!r}'.format(name, comp, value)
    logger.debug(
        'converted name={} and value={} to filter {}'.format(
            name, value, result))
    return result


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
    with log_start_finish('filter table', logger):
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


def has_constant_expr(expr):
    """
    Report whether a model expression has constant specific term.
    That is, a term explicitly specying whether the model should or
    should not include a constant. (e.g. '+ 1' or '- 1'.)

    Parameters
    ----------
    expr : str
        Model expression to check.

    Returns
    -------
    has_constant : bool

    """
    def has_constant(node):
        if node.type == 'ONE':
            return True

        for n in node.args:
            if has_constant(n):
                return True

        return False

    return has_constant(patsy.parse_formula.parse_formula(expr))


def str_model_expression(expr, add_constant=True):
    """
    We support specifying model expressions as strings, lists, or dicts;
    but for use with patsy and statsmodels we need a string.
    This function will take any of those as input and return a string.

    Parameters
    ----------
    expr : str, iterable, or dict
        A string will be returned unmodified except to add or remove
        a constant.
        An iterable sequence will be joined together with ' + '.
        A dictionary should have ``right_side`` and, optionally,
        ``left_side`` keys. The ``right_side`` can be a list or a string
        and will be handled as above. If ``left_side`` is present it will
        be joined with ``right_side`` with ' ~ '.
    add_constant : bool, optional
        Whether to add a ' + 1' (if True) or ' - 1' (if False) to the model.
        If the expression already has a '+ 1' or '- 1' this option will be
        ignored.

    Returns
    -------
    model_expression : str
        A string model expression suitable for use with statsmodels and patsy.

    """
    if not isinstance(expr, str):
        if isinstance(expr, collections.Mapping):
            left_side = expr.get('left_side')
            right_side = str_model_expression(expr['right_side'], add_constant)
        else:
            # some kind of iterable like a list
            left_side = None
            right_side = ' + '.join(expr)

        if left_side:
            model_expression = ' ~ '.join((left_side, right_side))
        else:
            model_expression = right_side

    else:
        model_expression = expr

    if not has_constant_expr(model_expression):
        if add_constant:
            model_expression += ' + 1'
        else:
            model_expression += ' - 1'

    logger.debug(
        'converted expression: {!r} to model: {!r}'.format(
            expr, model_expression))
    return model_expression


def sorted_groupby(df, groupby):
    """
    Perform a groupby on a DataFrame using a specific column
    and assuming that that column is sorted.

    Parameters
    ----------
    df : pandas.DataFrame
    groupby : object
        Column name on which to groupby. This column must be sorted.

    Returns
    -------
    generator
        Yields pairs of group_name, DataFrame.

    """
    start = 0
    prev = df[groupby].iloc[start]
    for i, x in enumerate(df[groupby]):
        if x != prev:
            yield prev, df.iloc[start:i]
            prev = x
            start = i
    # need to send back the last group
    yield prev, df.iloc[start:]


def columns_in_filters(filters):
    """
    Returns a list of the columns used in a set of query filters.

    Parameters
    ----------
    filters : list of str or str
        List of the filters as passed passed to ``apply_filter_query``.

    Returns
    -------
    columns : list of str
        List of all the strings mentioned in the filters.

    """
    if not filters:
        return []

    if not isinstance(filters, str):
        filters = ' '.join(filters)

    columns = []
    reserved = {'and', 'or', 'in', 'not'}

    for toknum, tokval, _, _, _ in generate_tokens(StringIO(filters).readline):
        if toknum == NAME and tokval not in reserved:
            columns.append(tokval)

    return list(toolz.unique(columns))


def _tokens_from_patsy(node):
    """
    Yields all the individual tokens from within a patsy formula
    as parsed by patsy.parse_formula.parse_formula.

    Parameters
    ----------
    node : patsy.parse_formula.ParseNode

    """
    for n in node.args:
        for t in _tokens_from_patsy(n):
            yield t

    if node.token:
        yield node.token


def columns_in_formula(formula):
    """
    Returns the names of all the columns used in a patsy formula.

    Parameters
    ----------
    formula : str, iterable, or dict
        Any formula construction supported by ``str_model_expression``.

    Returns
    -------
    columns : list of str

    """
    if formula is None:
        return []

    formula = str_model_expression(formula, add_constant=False)
    columns = []

    tokens = map(
        lambda x: x.extra,
        toolz.remove(
            lambda x: x.extra is None,
            _tokens_from_patsy(patsy.parse_formula.parse_formula(formula))))

    for tok in tokens:
        # if there are parentheses in the expression we
        # want to drop them and everything outside
        # and start again from the top
        if '(' in tok:
            start = tok.find('(') + 1
            fin = tok.rfind(')')
            columns.extend(columns_in_formula(tok[start:fin]))
        else:
            for toknum, tokval, _, _, _ in generate_tokens(
                    StringIO(tok).readline):
                if toknum == NAME:
                    columns.append(tokval)

    return list(toolz.unique(columns))
