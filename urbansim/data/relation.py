import pandas as pd


class Relation(object):
    """
    Store a join-like relationship between two tables.

    Can be used for getting columns from the `right` table with the index of
    the `left` table. This is the equivalent of a SQL inner join.

    Parameters
    ----------
    left : pandas.DataFrame
    right : pandas.DataFrame
    left_on, right_on : optional
        Column name of values on which to join.
        If None the index is used.

    """
    def __init__(self, left, right, left_on=None, right_on=None):
        self.left = left
        self.right = right

        self._merged_index = self._make_merged_index(left_on, right_on)

    def _make_merged_index(self, left_on, right_on):
        """
        Build the mapping from the left table index to the right table index.

        Parameters
        ----------
        left_on, right_on
            Column name of values on which to join.
            If None the index is used.

        Returns
        -------
        merged_idx : pandas.Series
            Index is left index values, values are right index values.

        """
        if left_on is not None:
            left_idx = self.left[left_on]
        else:
            left_idx = self.left.index.to_series()

        if right_on is not None:
            right_idx = pd.Series(self.right.index, index=self.right[right_on])
        else:
            right_idx = self.right.index.to_series()

        merged_idx = right_idx.loc[left_idx]
        merged_idx.index = self.left.index
        return merged_idx

    def __getitem__(self, key):
        if key in self.left.columns:
            return self.left[key]

        if key in self.right.columns:
            col = self.right[key].loc[self._merged_index]
            col.index = self.left.index
            return col

        raise KeyError('Unknown column: {}'.format(key))

    def __getattr__(self, attr):
        return self.__getitem__(attr)
