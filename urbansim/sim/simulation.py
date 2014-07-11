import inspect
from collections import Callable

import pandas as pd

_TABLES = {}


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
        return self._frame.columns

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
        if columns:
            return self._frame[list(columns)]
        else:
            return self._frame


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
        self._arg_list = inspect.getargspec(func).args
        self._columns = self.to_frame().columns

    @property
    def columns(self):
        """
        Columns in this table.

        """
        return self._columns

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
        self._columns = frame.columns
        return _DataFrameWrapper(self.name, frame).to_frame(columns)


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
