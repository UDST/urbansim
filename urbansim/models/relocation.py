"""
Use the ``RelocationModel`` class to choose movers based on
relocation rates.

"""
import logging

import numpy as np
import pandas as pd

from . import util

logger = logging.getLogger(__name__)


def find_movers(choosers, rates, rate_column):
    """
    Returns an array of the indexes of the `choosers` that are slated
    to move.

    Parameters
    ----------
    choosers : pandas.DataFrame
        Table of agents from which to find movers.
    rates : pandas.DataFrame
        Table of relocation rates. Index is unused.

        Other columns describe filters on the `choosers`
        table so that different segments can have different relocation
        rates. Columns that ends with '_max' will be used to create
        a "less than" filters, columns that end with '_min' will be
        used to create "greater than or equal to" filters.
        A column with no suffix will be used to make an 'equal to' filter.

        An example `rates` structure:

        age_of_head_max  age_of_head_min
                    nan               65
                     65               40

        In this example the `choosers` table would need to have an
        'age_of_head' column on which to filter.

        nan should be used to flag filters that do not apply
        in a given row.
    rate_column : object
        Name of column in `rates` table that has relocation rates.

    Returns
    -------
    movers : pandas.Index
        Suitable for indexing `choosers` by index.

    """
    logger.debug('start: find movers for relocation')
    relocation_rates = pd.Series(
        np.zeros(len(choosers)), index=choosers.index)

    for _, row in rates.iterrows():
        indexes = util.filter_table(choosers, row, ignore={rate_column}).index
        relocation_rates.loc[indexes] = row[rate_column]

    movers = relocation_rates.index[
        relocation_rates > np.random.random(len(choosers))]
    logger.debug('picked {} movers for relocation'.format(len(movers)))
    logger.debug('finish: find movers for relocation')
    return movers


class RelocationModel(object):
    """
    Find movers within a population according to a table of
    relocation rates.

    Parameters
    ----------
    rates : pandas.DataFrame
        Table of relocation rates. Index is unused.

        Other columns describe filters on the `choosers`
        table so that different segments can have different relocation
        rates. Columns that ends with '_max' will be used to create
        a "less than" filters, columns that end with '_min' will be
        used to create "greater than or equal to" filters.
        A column with no suffix will be used to make an 'equal to' filter.

        An example `rates` structure:

        age_of_head_max  age_of_head_min
                    nan               65
                     65               40

        In this example the `choosers` table would need to have an
        'age_of_head' column on which to filter.

        nan should be used to flag filters that do not apply
        in a given row.
    rate_column : object, optional
        Name of column in `rates` table that contains relocation rates.
        If not given 'probability_of_relocating' is used.

    """
    def __init__(self, rates, rate_column=None):
        self.relocation_rates = rates
        self.rate_column = rate_column or 'probability_of_relocating'

    def find_movers(self, choosers):
        """
        Select movers from among a table of `choosers` according to the
        stored relocation rates.

        Parameters
        ----------
        choosers : pandas.DataFrame
            Table of agents from which to find movers.

        Returns
        -------
        movers : pandas.Index
            Suitable for indexing `choosers` by index.

        """
        return find_movers(choosers, self.relocation_rates, self.rate_column)
