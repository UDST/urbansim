import numpy as np
import pandas as pd

from . import util

PROB_COL = 'probability_of_relocating'


def find_movers(choosers, rates):
    """
    Returns an array of the indexes of the `choosers` that are slated
    to move.

    Parameters
    ----------
    choosers : pandas.DataFrame
        Table of agents from which to find movers.
    rates : pandas.DataFrame
        Table of relocation rates. Index is unused. Must have a
        'probability_of_relocating' column with fraction relocation
        rates.

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

    Returns
    -------
    movers : pandas.Index
        Suitable for indexing `choosers` by index.

    """
    relocation_rates = pd.Series(
        np.zeros(len(choosers)), index=choosers.index)

    for _, row in rates.iterrows():
        indexes = util.filter_table(choosers, row, ignore={PROB_COL}).index
        relocation_rates.loc[indexes] = row[PROB_COL]

    movers = relocation_rates.index[
        relocation_rates > np.random.random(len(choosers))]
    return movers


class RelocationModel(object):
    """
    Find movers within a population according to a table of
    relocation rates.

    Parameters
    ----------
    rates : pandas.DataFrame
        Table of relocation rates. Index is unused. Must have a
        'probability_of_relocating' column with fraction relocation
        rates.

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
    col_to_mark : str or int, optional
        Name of column to modify when making movers with nan.

    """
    def __init__(self, rates):
        self.relocation_rates = rates

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
        return find_movers(choosers, self.relocation_rates)
