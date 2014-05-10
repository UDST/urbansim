from __future__ import division

import numpy as np
import pandas as pd

from . import util


def _empty_index():
    return pd.Index([])


def add_rows(data, nrows, starting_index=None):
    """
    Add rows to data table according to a given nrows.
    New rows will have their IDs set to NaN.

    Parameters
    ----------
    data : pandas.DataFrame
    nrows : int
        Number of rows to add.
    starting_index : int, optional
        The starting index from which to calculate indexes for the new
        rows. If not given the max + 1 of the index of `data` will be used.

    Returns
    -------
    updated : pandas.DataFrame
        Table with rows added. New rows will have their index values
        set to NaN.
    added : pandas.Index
        New indexes of the rows that were added.
    copied : pandas.Index
        Indexes of rows that were copied. A row copied multiple times
        will have multiple entries.

    """
    if nrows == 0:
        return data, _empty_index(), _empty_index()

    if not starting_index:
        starting_index = data.index.values.max() + 1

    i_to_copy = np.random.choice(data.index.values, nrows)
    new_rows = data.loc[i_to_copy].copy()
    added_index = pd.Index(np.arange(
        starting_index, starting_index + nrows, dtype=np.int))
    new_rows.index = added_index

    return pd.concat([data, new_rows]), added_index, pd.Index(i_to_copy)


def remove_rows(data, nrows):
    """
    Remove a random `nrows` number of rows from a table.

    Parameters
    ----------
    data : DataFrame
    nrows : float
        Number of rows to remove.

    Returns
    -------
    updated : pandas.DataFrame
        Table with random rows removed.
    removed : pandas.Index
        Indexes of the rows removed from the table.

    """
    nrows = abs(nrows)  # in case a negative number came in
    if nrows == 0:
        return data, _empty_index()
    elif nrows >= len(data):
        raise ValueError('Operation would remove entire table.')

    i_to_keep = np.random.choice(
        data.index.values, len(data) - nrows, replace=False)

    return data.loc[i_to_keep], data.index.diff(i_to_keep)


def add_or_remove_rows(data, nrows, starting_index=None):
    """
    Add or remove rows to/from a table. Rows are added
    for positive `nrows` and removed for negative `nrows`.

    Parameters
    ----------
    data : DataFrame
    nrows : float
        Number of rows to add or remove.
    starting_index : int, optional
        The starting index from which to calculate indexes for new rows.
        If not given the max + 1 of the index of `data` will be used.
        (Not applicable if rows are being removed.)

    Returns
    -------
    updated : pandas.DataFrame
        Table with random rows removed.
    added : pandas.Index
        New indexes of the rows that were added.
    copied : pandas.Index
        Indexes of rows that were copied. A row copied multiple times
        will have multiple entries.
    removed : pandas.Index
        Index of rows that were removed.

    """
    if nrows > 0:
        updated, added, copied = add_rows(data, nrows, starting_index)
        removed = _empty_index()

    elif nrows < 0:
        updated, removed = remove_rows(data, nrows)
        added, copied = _empty_index(), _empty_index()

    else:
        updated, added, copied, removed = \
            data, _empty_index(), _empty_index(), _empty_index()

    return updated, added, copied, removed


class GrowthRateTransition(object):
    """
    Transition given tables using a simple growth rate.

    Parameters
    ----------
    growth_rate : float

    """
    def __init__(self, growth_rate):
        self.growth_rate = growth_rate

    def transition(self, data, year):
        """
        Add or remove rows to/from a table according to the prescribed
        growth rate for this model.

        Parameters
        ----------
        data : pandas.DataFrame
            Rows will be removed from or added to this table.
        year : None, optional
            Here for compatibility with other transition models,
            but ignored.

        Returns
        -------
        updated : pandas.DataFrame
            Table with rows removed or added.
        added : pandas.Index
            New indexes of the rows that were added.
        copied : pandas.Index
            Indexes of rows that were copied. A row copied multiple times
            will have multiple entries.
        removed : pandas.Index
            Index of rows that were removed.

        """
        nrows = int(round(len(data) * self.growth_rate))
        return add_or_remove_rows(data, nrows)

    def __call__(self, data, year):
        """
        Call `self.transition` with inputs.

        """
        return self.transition(data, year)


class TabularGrowthRateTransition(object):
    """
    Growth rate based transitions where the rates are stored in
    a table indexed by year with optional segmentation.

    Parameters
    ----------
    growth_rates : pandas.DataFrame
    rates_column : str
        Name of the column in `growth_rates` that contains the rates.

    """
    def __init__(self, growth_rates, rates_column):
        self.growth_rates = growth_rates
        self.rates_column = rates_column

    @property
    def _config_table(self):
        """
        Table that has transition configuration.

        """
        return self.growth_rates

    @property
    def _config_column(self):
        """
        Non-filter column in config table.

        """
        return self.rates_column

    def _calc_nrows(self, len_data, growth_rate):
        """
        Calculate the number of rows to add to or remove from some data.

        Parameters
        ----------
        len_data : int
            The current number of rows in the data table.
        growth_rate : float
            Growth rate as a fraction. Positive for growth, negative
            for removing rows.

        """
        return int(round(len_data * growth_rate))

    def transition(self, data, year):
        """
        Add or remove rows to/from a table according to the prescribed
        growth rate for this model and year.

        Parameters
        ----------
        data : pandas.DataFrame
            Rows will be removed from or added to this table.
        year : None, optional
            Here for compatibility with other transition models,
            but ignored.

        Returns
        -------
        updated : pandas.DataFrame
            Table with rows removed or added.
        added : pandas.Index
            New indexes of the rows that were added.
        copied : pandas.Index
            Indexes of rows that were copied. A row copied multiple times
            will have multiple entries.
        removed : pandas.Index
            Index of rows that were removed.

        """
        if year not in self._config_table.index:
            raise ValueError('No targets for given year: {}'.format(year))

        # want this to be a DataFrame
        year_config = self._config_table.loc[[year]]

        segments = []
        added_indexes = []
        copied_indexes = []
        removed_indexes = []

        # since we're looping over descrete segments we need to track
        # out here where their new indexes will begin
        starting_index = data.index.values.max() + 1

        for _, row in year_config.iterrows():
            subset = util.filter_table(data, row, ignore={self._config_column})
            nrows = self._calc_nrows(len(subset), row[self._config_column])
            updated, added, copied, removed = \
                add_or_remove_rows(subset, nrows, starting_index)
            starting_index = starting_index + nrows + 1
            segments.append(updated)
            added_indexes.append(added)
            copied_indexes.append(copied)
            removed_indexes.append(removed)

        updated = pd.concat(segments)
        added_indexes = util.concat_indexes(added_indexes)
        copied_indexes = util.concat_indexes(copied_indexes)
        removed_indexes = util.concat_indexes(removed_indexes)

        return updated, added_indexes, copied_indexes, removed_indexes

    def __call__(self, data, year):
        """
        Call `self.transition` with inputs.

        """
        return self.transition(data, year)


class TabularTotalsTransition(TabularGrowthRateTransition):
    """
    Transition data via control totals in pandas DataFrame with
    optional segmentation.

    Parameters
    ----------
    targets : pandas.DataFrame
    totals_column : str
        Name of the column in `targets` that contains the control totals.

    """
    def __init__(self, targets, totals_column):
        self.targets = targets
        self.totals_column = totals_column

    @property
    def _config_table(self):
        """
        Table that has transition configuration.

        """
        return self.targets

    @property
    def _config_column(self):
        """
        Non-filter column in config table.

        """
        return self.totals_column

    def _calc_nrows(self, len_data, target_pop):
        """
        Calculate the number of rows to add to or remove from some data.

        Parameters
        ----------
        len_data : int
            The current number of rows in the data table.
        target_pop : int
            Target population.

        """
        return target_pop - len_data

    def transition(self, data, year):
        """
        Add or remove rows to/from a table according to the prescribed
        totals for this model and year.

        Parameters
        ----------
        data : pandas.DataFrame
            Rows will be removed from or added to this table.
        year : None, optional
            Here for compatibility with other transition models,
            but ignored.

        Returns
        -------
        updated : pandas.DataFrame
            Table with rows removed or added.
        added : pandas.Index
            New indexes of the rows that were added.
        copied : pandas.Index
            Indexes of rows that were copied. A row copied multiple times
            will have multiple entries.
        removed : pandas.Index
            Index of rows that were removed.

        """
        return super(TabularTotalsTransition, self).transition(data, year)


def _update_linked_table(table, col_name, added, copied, removed):
    """
    Copy and update rows in a table that has a column referencing another
    table that has had rows added via copying.

    Parameters
    ----------
    table : pandas.DataFrame
        Table to update with new or removed rows.
    col_name : str
        Name of column in `table` that corresponds to the index values
        in `copied` and `removed`.
    added : pandas.Index
        Indexes of rows that are new in the linked table.
    copied : pandas.Index
        Indexes of rows that were copied to make new rows in linked table.
    removed : pandas.Index
        Indexes of rows that were removed from the linked table.

    Returns
    -------
    updated : pandas.DataFrame

    """
    table = table.loc[~table[col_name].isin(removed)]

    id_map = added.groupby(copied)
    new_rows = []

    for copied_id, new_ids in id_map.items():
        rows = table.query('{} == {}'.format(col_name, copied_id))
        # number of times we'll need to duplicate new_ids
        n_matching_rows = len(rows)
        rows = rows.loc[rows.index.repeat(len(new_ids))]
        rows[col_name] = new_ids * n_matching_rows
        new_rows.append(rows)

    new_rows = pd.concat(new_rows)

    starting_index = table.index.values.max() + 1
    new_rows.index = np.arange(
        starting_index, starting_index + len(new_rows), dtype=np.int)

    return pd.concat([table, new_rows])


class TransitionModel(object):
    """
    Models things moving into or out of a region.

    Parameters
    ----------
    transitioner : callable
        A callable that takes a data table and a year number and returns
        and new data table, the indexes of rows added, the indexes
        of rows copied, and the indexes of rows removed.

    """
    def __init__(self, transitioner):
        self.transitioner = transitioner

    def transition(self, data, year, linked_tables=None):
        """
        Add or remove rows from a table based on population targets.

        Parameters
        ----------
        data : pandas.DataFrame
            Rows will be removed from or added to this table.
        year : int
            Year number that will be passed to `transitioner`.
        linked_tables : dict of tuple, optional
            Dictionary of (table, 'column name') pairs. The column name
            should match the index of `data`. Indexes in `data` that
            are copied or removed will also be copied and removed in
            linked tables. They dictionary keys are used in the
            returned `updated_links`.

        Returns
        -------
        updated : pandas.DataFrame
            Table with rows removed or added.
        updated_links : dict of pandas.DataFrame

        """
        linked_tables = linked_tables or {}
        updated_links = {}

        updated, added, copied, removed = self.transitioner(data, year)

        for table_name, (table, col) in linked_tables.items():
            updated_links[table_name] = \
                _update_linked_table(table, col, added, copied, removed)

        return updated, updated_links
