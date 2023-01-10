"""
Use the ``TransitionModel`` class with the different transitioners to
add or remove agents based on growth rates or target totals.

"""
from __future__ import division

import logging

import numpy as np
import pandas as pd

from . import util
from ..utils.logutil import log_start_finish
from ..utils.sampling import sample_rows

logger = logging.getLogger(__name__)


def _empty_index():
    return pd.Index([])


def add_rows(data, nrows, starting_index=None, accounting_column=None):
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
    accounting_column: string, optional
        Name of column with accounting totals/quanties to apply towards the control. If not provided
        then row counts will be used for accounting.

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
    logger.debug('start: adding {} rows in transition model'.format(nrows))
    if nrows == 0:
        return data, _empty_index(), _empty_index()

    if not starting_index:
        starting_index = data.index.values.max() + 1

    new_rows = sample_rows(nrows, data, accounting_column=accounting_column)
    copied_index = new_rows.index
    added_index = pd.Index(np.arange(
        starting_index, starting_index + len(new_rows.index), dtype=int))
    new_rows.index = added_index

    logger.debug(
        'finish: added {} rows in transition model'.format(len(new_rows)))
    return pd.concat([data, new_rows]), added_index, copied_index


def remove_rows(data, nrows, accounting_column=None):
    """
    Remove a random `nrows` number of rows from a table.

    Parameters
    ----------
    data : DataFrame
    nrows : float
        Number of rows to remove.
    accounting_column: string, optional
        Name of column with accounting totals/quanties to apply towards the control. If not provided
        then row counts will be used for accounting.

    Returns
    -------
    updated : pandas.DataFrame
        Table with random rows removed.
    removed : pandas.Index
        Indexes of the rows removed from the table.

    """
    logger.debug('start: removing {} rows in transition model'.format(nrows))
    nrows = abs(nrows)  # in case a negative number came in
    unit_check = data[accounting_column].sum() if accounting_column else len(data)
    if nrows == 0:
        return data, _empty_index()
    elif nrows > unit_check:
        raise ValueError('Number of rows to remove exceeds number of records in table.')

    remove_rows = sample_rows(nrows, data, accounting_column=accounting_column, replace=False)
    remove_index = remove_rows.index

    logger.debug('finish: removed {} rows in transition model'.format(nrows))
    return data.loc[data.index.difference(remove_index)], remove_index


def add_or_remove_rows(data, nrows, starting_index=None, accounting_column=None):
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
        updated, added, copied = add_rows(
            data, nrows, starting_index,
            accounting_column=accounting_column)
        removed = _empty_index()

    elif nrows < 0:
        updated, removed = remove_rows(data, nrows, accounting_column=accounting_column)
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
    accounting_column: string, optional
        Name of column with accounting totals/quanties to apply towards the control. If not provided
        then row counts will be used for accounting.
    """
    def __init__(self, growth_rate, accounting_column=None):
        self.growth_rate = growth_rate
        self.accounting_column = accounting_column

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
        if self.accounting_column is None:
            nrows = int(round(len(data) * self.growth_rate))
        else:
            nrows = int(round(data[self.accounting_column].sum() * self.growth_rate))
        with log_start_finish(
                'adding {} rows via growth rate ({}) transition'.format(
                    nrows, self.growth_rate),
                logger):
            return add_or_remove_rows(data, nrows, accounting_column=self.accounting_column)

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
    accounting_column: string, optional
        Name of column with accounting totals/quanties to apply towards the control. If not provided
        then row counts will be used for accounting.
    """
    def __init__(self, growth_rates, rates_column, accounting_column=None):
        self.growth_rates = growth_rates
        self.rates_column = rates_column
        self.accounting_column = accounting_column

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
        logger.debug('start: tabular transition')
        if year not in self._config_table.index:
            raise ValueError('No targets for given year: {}'.format(year))

        # want this to be a DataFrame
        year_config = self._config_table.loc[[year]]
        logger.debug('transitioning {} segments'.format(len(year_config)))

        segments = []
        added_indexes = []
        copied_indexes = []
        removed_indexes = []

        # since we're looping over discrete segments we need to track
        # out here where their new indexes will begin
        starting_index = data.index.values.max() + 1

        for _, row in year_config.iterrows():
            subset = util.filter_table(data, row, ignore={self._config_column})

            # Do not run on segment if it is empty
            if len(subset) == 0:
                logger.debug('empty segment encountered')
                continue

            if self.accounting_column is None:
                nrows = self._calc_nrows(len(subset), row[self._config_column])
            else:
                nrows = self._calc_nrows(
                    subset[self.accounting_column].sum(),
                    row[self._config_column])

            updated, added, copied, removed = \
                add_or_remove_rows(subset, nrows, starting_index, self.accounting_column)
            if nrows > 0:
                # only update the starting index if rows were added
                starting_index = starting_index + nrows
            segments.append(updated)
            added_indexes.append(added)
            copied_indexes.append(copied)
            removed_indexes.append(removed)

        updated = pd.concat(segments)
        added_indexes = util.concat_indexes(added_indexes)
        copied_indexes = util.concat_indexes(copied_indexes)
        removed_indexes = util.concat_indexes(removed_indexes)

        logger.debug('finish: tabular transition')
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
    accounting_column: string, optional
        Name of column with accounting totals/quanties to apply towards the control. If not provided
        then row counts will be used for accounting.
    """
    def __init__(self, targets, totals_column, accounting_column=None):
        self.targets = targets
        self.totals_column = totals_column
        self.accounting_column = accounting_column

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
        with log_start_finish('tabular totals transition', logger):
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
    logger.debug('start: update linked table after transition')

    # handle removals
    table = table.loc[~table[col_name].isin(set(removed))]
    if (added is None or len(added) == 0):
        return table

    # map new IDs to the IDs from which they were copied
    id_map = pd.concat([pd.Series(copied, name=col_name), pd.Series(added, name='temp_id')], axis=1)

    # join to linked table and assign new id
    new_rows = id_map.merge(table, on=col_name)
    new_rows.drop(col_name, axis=1, inplace=True)
    new_rows.rename(columns={'temp_id': col_name}, inplace=True)

    # index the new rows
    starting_index = table.index.values.max() + 1
    new_rows.index = np.arange(starting_index, starting_index + len(new_rows), dtype=int)

    logger.debug('finish: update linked table after transition')
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
        added : pandas.Series
            Indexes of new rows in `updated`.
        updated_links : dict of pandas.DataFrame

        """
        logger.debug('start: transition')
        linked_tables = linked_tables or {}
        updated_links = {}

        with log_start_finish('add/remove rows', logger):
            updated, added, copied, removed = self.transitioner(data, year)

        for table_name, (table, col) in linked_tables.items():
            logger.debug('updating linked table {}'.format(table_name))
            updated_links[table_name] = \
                _update_linked_table(table, col, added, copied, removed)

        logger.debug('finish: transition')
        return updated, added, updated_links
