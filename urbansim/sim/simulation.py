from __future__ import print_function

import inspect
from collections import Callable

import pandas as pd
import toolz
import time
import logging

logger = logging.getLogger(__name__)

_TABLES = {}
_COLUMNS = {}
_MODELS = {}


def clear_sim():
    """
    Clear any stored state from the simulation.

    """
    _TABLES.clear()
    _COLUMNS.clear()
    _MODELS.clear()


class _DataFrameWrapper(object):
    """
    Wraps a DataFrame so it can provide certain columns and handle
    computed columns.

    Parameters
    ----------
    name : str
        Name for the table.
    frame : pandas.DataFrame

    """
    def __init__(self, name, frame):
        self.name = name
        self._frame = frame

    @property
    def columns(self):
        """
        Columns in this table.

        """
        return list(self._frame.columns) + _list_columns_for_table(self.name)

    @property
    def local_columns(self):
        """
        Columns in this table.

        """
        return list(self._frame.columns)

    @property
    def index(self):
        """
        Table index.

        """
        return self._frame.index

    def to_frame(self, columns=None):
        """
        Make a DataFrame with the given columns.

        Parameters
        ----------
        columns : sequence, optional
            Sequence of the column names desired in the DataFrame.
            If None all columns are returned, including registered columns.

        Returns
        -------
        frame : pandas.DataFrame

        """
        extra_cols = _columns_for_table(self.name)

        if columns:
            local_cols = [c for c in self._frame.columns
                          if c in columns and c not in extra_cols]
            extra_cols = toolz.keyfilter(lambda c: c in columns, extra_cols)
            df = self._frame[local_cols].copy()
        else:
            df = self._frame.copy()

        for name, col in extra_cols.items():
            df[name] = col()

        return df

    def update_col(self, column_name, series):
        """
        Add or replace a column in the underlying DataFrame.

        Parameters
        ----------
        column_name : str
            Column to add or replace.
        series : pandas.Series or sequence
            Column data.

        """
        self._frame[column_name] = series

    def __setitem__(self, key, value):
        return self.update_col(key, value)

    def get_column(self, column_name):
        """
        Returns a column as a Series.

        Parameters
        ----------
        column_name : str

        Returns
        -------
        column : pandas.Series

        """
        return self.to_frame(columns=[column_name])[column_name]

    def __getitem__(self, key):
        return self.get_column(key)

    def __getattr__(self, key):
        return self.get_column(key)

    def __len__(self):
        return len(self._frame)


class _TableFuncWrapper(object):
    """
    Wrap a function that provides a DataFrame.

    Parameters
    ----------
    name : str
        Name for the table.
    func : callable
        Callable that returns a DataFrame.

    """
    def __init__(self, name, func):
        self.name = name
        self._func = func
        self._arg_list = set(inspect.getargspec(func).args)
        self._columns = []
        self._index = None
        self._len = 0

    @property
    def columns(self):
        """
        Columns in this table. (May often be out of date.)

        """
        return self._columns + _list_columns_for_table(self.name)

    @property
    def index(self):
        """
        Index of the underlying table. Will be None if that index is
        unknown.

        """
        return self._index

    def to_frame(self, columns=None):
        """
        Make a DataFrame with the given columns.

        Parameters
        ----------
        columns : sequence, optional
            Sequence of the column names desired in the DataFrame.
            If None all columns are returned.

        Returns
        -------
        frame : pandas.DataFrame

        """
        kwargs = {t: get_table(t) for t in self._arg_list}
        frame = self._func(**kwargs)
        self._columns = list(frame.columns)
        self._index = frame.index
        self._len = len(frame)
        return _DataFrameWrapper(self.name, frame).to_frame(columns)

    def get_column(self, column_name):
        """
        Returns a column as a Series.

        Parameters
        ----------
        column_name : str

        Returns
        -------
        column : pandas.Series

        """
        return self.to_frame(columns=[column_name])[column_name]

    def __getitem__(self, key):
        return self.get_column(key)

    def __getattr__(self, key):
        return self.get_column(key)

    def __len__(self):
        return self._len


class _ColumnFuncWrapper(object):
    """
    Wrap a function that returns a Series.

    Parameters
    ----------
    table_name : str
        Table with which the column will be associated.
    column_name : str
        Name for the column.
    func : callable
        Should return a Series that has an
        index matching the table to which it is being added.

    """
    def __init__(self, table_name, column_name, func):
        self.table_name = table_name
        self.name = column_name
        self._func = func
        self._arg_list = set(inspect.getargspec(func).args)

    def __call__(self):
        kwargs = {t: get_table(t) for t in self._arg_list}
        return self._func(**kwargs)


class _SeriesWrapper(object):
    """
    Wrap a Series for the purpose of giving it the same interface as a
    `_ColumnFuncWrapper`.

    Parameters
    ----------
    table_name : str
        Table with which the column will be associated.
    column_name : str
        Name for the column.
    func : callable
        Should return a Series that has an
        index matching the table to which it is being added.

    """
    def __init__(self, table_name, column_name, series):
        self.table_name = table_name
        self.name = column_name
        self._column = series

    def __call__(self):
        return self._column


class _ModelFuncWrapper(object):
    """
    Wrap a model function for dependency injection.

    Parameters
    ----------
    model_name : str
    func : callable

    """
    def __init__(self, model_name, func):
        self.name = model_name
        self._func = func
        self._arg_list = set(inspect.getargspec(func).args)

    def __call__(self, year=None):
        kwargs = {t: get_table(t) for t in self._arg_list if t != 'year'}
        if 'year' in self._arg_list:
            kwargs['year'] = year
        return self._func(**kwargs)


def add_table(table_name, table):
    """
    Register a table with the simulation.

    Parameters
    ----------
    table_name : str
        Should be globally unique to this table.
    table : pandas.DataFrame or function
        If a function it should return a DataFrame. Function argument
        names will be matched to known tables, which will be injected
        when this function is called.

    """
    if isinstance(table, pd.DataFrame):
        table = _DataFrameWrapper(table_name, table)
    elif isinstance(table, Callable):
        table = _TableFuncWrapper(table_name, table)
    else:
        raise TypeError('table must be DataFrame or function.')

    _TABLES[table_name] = table


def table(table_name):
    """
    Decorator version of `add_table` used for decorating functions
    that return DataFrames.

    Decorated function argument names will be matched to known tables,
    which will be injected when this function is called.

    """
    def decorator(func):
        add_table(table_name, func)
        return func
    return decorator


def get_table(table_name):
    """
    Get a registered table.

    Parameters
    ----------
    table_name : str

    Returns
    -------
    table : _DataFrameWrapper or _TableFuncWrapper

    """
    if table_name in _TABLES:
        return _TABLES[table_name]
    else:
        raise KeyError('table not found: {}'.format(table_name))


def list_tables():
    """
    List of table names.

    """
    return list(_TABLES.keys())


def add_column(table_name, column_name, column):
    """
    Add a new column to a table from a Series or callable.

    Parameters
    ----------
    table_name : str
        Table with which the column will be associated.
    column_name : str
        Name for the column.
    column : pandas.Series or callable
        If a callable it should return a Series. Any Series should have an
        index matching the table to which it is being added.

    """
    if isinstance(column, pd.Series):
        column = _SeriesWrapper(table_name, column_name, column)
    elif isinstance(column, Callable):
        column = \
            _ColumnFuncWrapper(table_name, column_name, column)
    else:
        raise TypeError('Only Series or callable allowed for column.')

    _COLUMNS[(table_name, column_name)] = column


def column(table_name, column_name):
    """
    Decorator version of `add_column` used for decorating functions
    that return a Series with an index matching the named table.

    The argument names of the function should match known tables, which
    will be injected.

    """
    def decorator(func):
        add_column(table_name, column_name, func)
        return func
    return decorator


def _list_columns_for_table(table_name):
    """
    Return a list of all the extra columns registered for a given table.

    Parameters
    ----------
    table_name : str

    Returns
    -------
    columns : list of str

    """
    return [cname for tname, cname in _COLUMNS.keys() if tname == table_name]


def _columns_for_table(table_name):
    """
    Return all of the columns registered for a given table.

    Parameters
    ----------
    table_name : str

    Returns
    -------
    columns : dict of column wrappers
        Keys will be column names.

    """
    return {cname: col
            for (tname, cname), col in _COLUMNS.items()
            if tname == table_name}


def add_model(model_name, func):
    """
    Add a model function to the simulation.

    Model argument names are used for injecting known tables of the same name.
    The argument name "year" may be used to have the current simulation
    year injected.

    Parameters
    ----------
    model_name : str
    func : callable

    """
    if isinstance(func, Callable):
        _MODELS[model_name] = _ModelFuncWrapper(model_name, func)
    else:
        raise TypeError('func must be a callable')


def model(model_name):
    """
    Decorator version of `add_model`, used to decorate a function that
    will require injection of tables and that can be run by the
    `run` function.

    """
    def decorator(func):
        add_model(model_name, func)
        return func
    return decorator


def get_model(model_name):
    """
    Get a wrapped model by name.

    Parameters
    ----------

    """
    if model_name in _MODELS:
        return _MODELS[model_name]
    else:
        raise KeyError('no model named {}'.format(model_name))


def run(models, years=None):
    """
    Run models in series, optionally repeatedly over some years.

    Parameters
    ----------
    models : list of str
        List of models to run identified by their name.

    """
    years = years or [None]

    for year in years:
        if year:
            print('Running year {}'.format(year))
        for model_name in models:
            print('Running model {}'.format(model_name))
            model = get_model(model_name)
            t1 = time.time()
            model(year=year)
            logger.debug("Time to execute model = %.3fs" % (time.time()-t1))
