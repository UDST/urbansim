"""
Use the ``AllocationModel`` class to distribute amounts/quantities
to rows in a target table based on a weights while respecting
capacity constraints. Also might be thought of as a
'pro-rating' model.

"""
import numpy as np
import pandas as pd
import urbansim.models.util as us_util


class AllocationModel(object):
    """
    Used to allocate quantities/amounts to
    rows in data frame.

    Parameters:
    ----------
    amounts_df: pandas.DataFrame
        Data frame containing amounts/quantities
        to allocate from. Assumed to be indexed
        by year.
    amounts_col: string
        Name of column in the amounts data frame
        to allocate from.
    target_col:
        Name of column in the targets data frame
        to allocate to.
    weight_col: string, optional
        Name of the column in the target data frame used to
        weight the allocation. If not provided all
        target rows will have an equal weight.
    capacity_col: string, optional
        Name of the column in the target data frame with
        capacities that need to be respected. If not
        provided the allocation will be unconstrained.
    as_integer: bool, optional, default True
        If True, allocation values will be integerized to match
        the allocation amount exactly. If False then floating
        point values will be returned.
    as_delta: bool optional, default False
        If True, the allocation will assume that the amount being
        distributed represents change. The resulting series
        will retain the original column values (presumably from
        the previous year) and will increment the column by
        the allocated change. If False, the allocation
        will be treated as a total and the target column will
        be entirely re-generated each year.
    compute_delta: bool option, default False
        If True, and as_delta is True, the amounts are assumed
        to represent totals and changes will be computed
        by comparing the amount with the previous year.
    segment_cols: list <string>
        List of field names that will be used for segmentation.
        Each segment will have its own allocation.

    """
    def __init__(self,
                 amounts_df,
                 amounts_col,
                 target_col,
                 weight_col=None,
                 capacity_col=None,
                 as_integer=True,
                 as_delta=False,
                 compute_delta=False,
                 segment_cols=None):
        self.amounts_df = amounts_df
        self.amounts_col = amounts_col
        self.target_col = target_col
        self.weight_col = weight_col
        self.capacity_col = capacity_col
        self.as_integer = as_integer
        self.as_delta = as_delta
        self.compute_delta = compute_delta
        self.segment_cols = segment_cols

        # handle segmentation columns
        if segment_cols is not None:
            self.ignore_cols = list(set(segment_cols) ^ set(amounts_df))
            keys = ['temp_year'] + segment_cols
        else:
            self.ignore_cols = None
            keys = ['temp_year']

        # check for duplicate amount records
        temp = amounts_df.copy()
        temp['temp_year'] = amounts_df.index.values
        groups = temp.groupby(keys)
        if groups.size().max() > 1:
            raise ValueError("Duplicate entries in amounts table")
        del temp

    def allocate(self, data, year):
        """
        Performs the allocation.

        Parameters:
        ----------
        data: pandas.DataFrame
            Target data frame to allocate to.
        year: int
            Simulation year.

        """

        # set up the output data frame
        data = data.copy()
        if data is None or len(data) == 0 or year not in self.amounts_df.index:
            return data
        if not self.as_delta:
            data[self.target_col] = 0

        # loop through the amounts for the current year
        for _, curr_row in self.amounts_df.loc[year:year].iterrows():

            # get the current amount
            amount = curr_row[self.amounts_col]

            # adjust the amount for deltas
            if self.as_delta and self.compute_delta:
                if year - 1 in self.amounts_df.index:
                    if self.segment_cols is None:
                        amount = amount - self.amounts_df.loc[year - 1][self.amounts_col]
                    else:
                        prev_q = ''
                        for seg_col in self.segment_cols:
                            prev_q += '{} == {} and '.format(seg_col, curr_row[seg_col])
                        prev_q = prev_q[:-4]
                        prev_rows = self.amounts_df.query(prev_q)
                        amount = amount - prev_rows[self.amounts_col][year - 1]

            # get the subset of rows for the current segment
            if self.segment_cols is None:
                subset = data
            else:
                subset = us_util.filter_table(data, curr_row, ignore=self.ignore_cols)

            # get the weight series
            if self.weight_col is not None:
                w = subset[self.weight_col]
            else:
                w = pd.Series(np.ones(len(subset)), index=subset.index)

            # get the capacity series
            if self.capacity_col is not None:
                c = subset[self.capacity_col]
                if c.sum() < amount:
                    raise ValueError('Amount exceeds the available capacity')
            else:
                c = pd.Series(np.ones(len(subset)) * amount, index=subset.index)

            # get the existing series
            if self.as_delta:
                e = subset[self.target_col]
            else:
                e = pd.Series(np.zeros(len(subset)), index=subset.index)

            # perform the initial allocation
            a = pd.Series(np.zeros(len(subset)), index=subset.index)
            while True:
                # allocate remaining amount to rows with capacity
                curr_amount = amount - a.sum()
                have_cap = a + e < c
                have_cap_cnt = len(a[have_cap])
                if have_cap_cnt > 0:
                    w_sum = w[have_cap].sum()
                    if w_sum != 0:
                        a[have_cap] = a + (curr_amount * (w / w_sum))
                    else:
                        a[have_cap] = a + (curr_amount / have_cap_cnt)

                # set allocation result to capacity for overages
                over = a + e > c
                if len(a[over]) > 0:
                    a[over] = c - e
                else:
                    break

            # integerize using stochastic rounding
            if self.as_integer:
                # make sure the amount is an integer
                if amount % 1 != 0:
                    raise ValueError('Cannot integerize and match non-integer amount')

                # only non-zero rows with capacity are eligible for rounding
                to_round_idx = a[(a < c) & (a != 0)].index.values
                to_round_vals = a[to_round_idx].values

                # get the fractional components
                fract = to_round_vals % 1
                fract_sum = fract.sum()
                if fract_sum > 0:
                    # round everything down
                    floor = np.floor(to_round_vals)
                    a[to_round_idx] = floor

                    # determine how many rows need to get rounded up
                    round_up_cnt = amount - a[a == c].sum() - floor.sum()

                    # randomly choose items to round up, weighted by their fractional part
                    fract_w = fract / fract_sum
                    idx_to_round_up = np.random.choice(to_round_idx, round_up_cnt, False, fract_w)
                    a[idx_to_round_up] += 1

            # update the data frame with the current allocation
            data[self.target_col].loc[subset.index.values] = a + e
        return data
