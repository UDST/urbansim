from __future__ import division

import numpy as np
import pandas as pd


def add_rows(data, nrows):
    """
    Add rows to data table according to a given nrows.
    New rows will have their IDs set to NaN.

    Parameters
    ----------
    data : pandas.DataFrame
    nrows : int
        Number of rows to add.

    Returns
    -------
    updated : pandas.DataFrame
        Table with rows added. New rows will have their index values
        set to NaN.

    """
    if nrows == 0:
        return data

    i_to_copy = np.random.choice(data.index.values, nrows)
    new_rows = data.loc[i_to_copy].copy()

    # the only wat to get NaNs into an index along with integers
    # seems to be to make the Index with dtype object
    new_rows.index = pd.Index([np.nan] * len(new_rows), dtype=np.object)

    return pd.concat([data, new_rows])


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

    """
    nrows = abs(nrows)  # in case a negative number came in
    if nrows == 0:
        return data
    elif nrows >= len(data):
        raise ValueError('Operation would remove entire table.')

    i_to_keep = np.random.choice(
        data.index.values, len(data) - nrows, replace=False)
    return data.loc[i_to_keep]


def fill_nan_ids(data):
    """
    Fill NaN values in the index of a table. They will be filled
    with integers larger than the largest existing ID.

    Parameters
    ----------
    data : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame

    """
    index = pd.Series(data.index)
    isnull = index.isnull()
    n_to_fill = isnull.sum()
    id_max = int(index.max())
    new_ids = range(id_max + 1, id_max + 1 + n_to_fill)
    index[isnull] = new_ids
    data.index = pd.Int64Index(index)
    return data


def _add_or_remove_rows(data, nrows, populate_ids=True):
    """
    Add or remove rows to/from a table. Rows are added
    for positive `nrows` and removed for negative `nrows`.

    Parameters
    ----------
    data : DataFrame
    nrows : float
        Number of rows to add or remove.
    populate_ids : bool, optional
        Whether to populate the index field of added rows.

    Returns
    -------
    updated : pandas.DataFrame
        Table with random rows removed.

    """
    if nrows > 0:
        updated = add_rows(data, nrows)
        if populate_ids:
            updated = fill_nan_ids(updated)
        return updated

    elif nrows < 0:
        return remove_rows(data, nrows)

    else:
        return data


class GRTransitionModel(object):
    """
    Model transitions via a simple growth rate.

    Parameters
    ----------
    growth_rate : float
        Rate of growth as a fraction. Negative numbers indicate
        population loss.

        For example, 0.05 for a five percent growth rate.
    populate_ids : bool, optional
        Whether to populate the index field of added rows.
        By default they will be populated with new IDs, but you will
        probably want to turn that off if this model is for a segment
        of a larger table so that appropriate IDs can be added after
        all the segments have had rows added.

    """
    def __init__(self, growth_rate, populate_ids=True):
        self.growth_rate = growth_rate
        self.populate_ids = populate_ids

    def transition(self, data, year=None):
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

        """
        nrows = int(round(len(data) * self.growth_rate))
        return _add_or_remove_rows(data, nrows, self.populate_ids)


class TabularTransitionModel(object):
    """
    A transition model based on yearly population targets.

    Parameters
    ----------
    targets : pandas.Series
        Pandas series containing the population targets
        indexed by year.
    populate_ids : bool, optional
        Whether to populate the index field of added rows.
        By default they will be populated with new IDs, but you will
        probably want to turn that off if this model is for a segment
        of a larger table so that appropriate IDs can be added after
        all the segments have had rows added.

    """
    def __init__(self, targets, populate_ids=True):
        self.targets = targets
        self.populate_ids = populate_ids

    def transition(self, data, year):
        """
        Add or remove rows from a table based on population targets.

        Parameters
        ----------
        data : pandas.DataFrame
            Rows will be removed from or added to this table.
        year : int
            Year number matching the index of `targets`.

        Returns
        -------
        updated : pandas.DataFrame
            Table with rows removed or added.

        """
        if year not in self.targets.index:
            raise ValueError('No targets for given year: {}'.format(year))

        nrows = self.targets[year] - len(data)
        return _add_or_remove_rows(data, nrows, self.populate_ids)
