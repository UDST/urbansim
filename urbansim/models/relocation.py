import numpy as np
import pandas as pd

from . import util
from ..utils import yamlio


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
    relocation_rates = pd.Series(
        np.zeros(len(choosers)), index=choosers.index)

    for _, row in rates.iterrows():
        indexes = util.filter_table(choosers, row, ignore={rate_column}).index
        relocation_rates.loc[indexes] = row[rate_column]

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

    @classmethod
    def from_yaml(cls, yaml_str=None, str_or_buffer=None):
        """
        Create a RelocationModel instance from a saved YAML configuration.
        Arguments are mutally exclusive.

        Parameters
        ----------
        yaml_str : str, optional
            A YAML string from which to load model.
        str_or_buffer : str or file like, optional
            File name or buffer from which to load YAML.

        Returns
        -------
        RelocationModel

        """
        cfg = yamlio.yaml_to_dict(yaml_str, str_or_buffer)

        return cls(
            pd.DataFrame(cfg['relocation_rates']),
            cfg.get('rate_column'))

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

    def to_dict(self):
        """
        Returns a dictionary representation of a RelocationModel instance.

        """
        return {
            'model_type': 'relocation',
            'relocation_rates': yamlio.frame_to_yaml_safe(
                self.relocation_rates),
            'rate_column': self.rate_column
        }

    def to_yaml(self, str_or_buffer=None):
        """
        Save a model respresentation to YAML.

        Parameters
        ----------
        str_or_buffer : str or file like, optional
            By default a YAML string is returned. If a string is
            given here the YAML will be written to that file.
            If an object with a ``.write`` method is given the
            YAML will be written to that object.

        Returns
        -------
        j : str
            YAML string if `str_or_buffer` is not given.

        """
        return yamlio.convert_to_yaml(self.to_dict(), str_or_buffer)
