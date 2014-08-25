"""
Use the ``AllocationModel`` class to distribute amounts/quantities
to rows in a target table based on a weights while respecting
capacity constraints. Also might be thought of as a
'pro-rating' model.

"""
import numpy as np
import pandas as pd


class Allocation(object):
    """
    Used to allocate an amount to a set of rows,
    optionally w/ weights and capacities.

    Parameters:
    ----------
    weights_col: string, optional
        Name of the column in the target data frame used to
        weight the allocation. If not provided all
        target rows will have an equal weight.
    capacity_col: string, optional
        Name of the column in the target data frame with
        capacities that need to be respected. If not
        provided the allocation will be unconstrained.
    base_col: string, optional
        If provided, the allocation will run in 'delta'
        mode, thus assuming that the quantity/amount being
        distributed represents change. The resulting series
        will retain the original column values and will apply
        the allocated change. If not provided, we are assuming
        that the quantity/amount represents a total.
    as_integer: bool, optional, default True
        If True, allocation values will be integerized to match
        the allocation amount exactly. If False then floating
        point values will be returned.

    """
    def __init__(self,
                 weight_col=None,
                 capacity_col=None,
                 base_col=None,
                 as_integer=True):
        self.weight_col = weight_col
        self.capacity_col = capacity_col
        self.base_col = base_col
        self.as_integer = as_integer


    def allocate(self, amount, data):
        """
        Allocates the provided quantity to the target data frame.

        Parameters:
        ----------
        amount: int
            The amount to allocate.
        data: pandas.DataFrame
            Table to allocate amount to.

        Returns
        -------
        allocated: pandas.Series
            Series with allocated values, indexed to the provided
            data frame.

        """
        if data is None or len(data) == 0:
            raise ValueError('Null or empty data frame')

        # get the weight series, if not provided rows have equal weight
        if self.weight_col is not None:
            w = data[self.weight_col]
            w_sum = w.sum()
            if w_sum == 0:
                raise ValueError('All weights are zero')
        else:
            w = pd.Series(np.ones(len(data)), index=data.index)
            w_sum = len(data)

        # get the capacity series, if not provided capacities set to the amount
        if self.capacity_col is not None:
            c = data[self.capacity_col]
            if c.sum() < amount:
                raise ValueError('Amount exceeds the available capacity')
        else:
            c = pd.Series(np.ones(len(data)) * amount , index=data.index)

        # get the allocation result series
        a = pd.Series(np.zeros(len(data)), index=data.index)

        # perform the initial allocation
        while True:
            # allocate remaining amount to rows with capacity
            curr_amount = amount - a.sum()
            have_cap = a < c
            w_sum = w[have_cap].sum()
            a[have_cap] = a + (curr_amount * (w / w_sum))

            # set allocation result to capacity for overages
            over = a > c
            if len(a[over]) > 0:
                a[over] = c
            else:
                break

        # integerize using stochastic rounding
        if self.as_integer:
            # make sure the amount is an integer
            if amount % 1 != 0:
                raise ValueError('Cannot integerize and match non-integer amount')

            # round all rows with capacity down
            to_round_idx = a[a < c].index.values
            to_round_vals = a[to_round_idx].values
            floor = np.floor(to_round_vals)
            a[to_round_idx]  = floor

            # determine how many rows need to get rounded up
            to_round_up = amount - a[a == c].sum() - floor.sum()

            # randomly choose items to round up, weighted by their remainder
            print to_round_vals
            fract = to_round_vals % 1
            fract_w = fract / fract.sum()
            print fract
            print fract_w
            idx_to_round_up = np.random.choice(to_round_idx, to_round_up, False, fract_w)
            a[idx_to_round_up] += 1

        # apply deltas (if necessary) and return allocation results
        if self.base_col is not None:
            return data[self.base_col] + a
        else:
            return a
