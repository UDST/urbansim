"""
The allocation classes are used for distributing
amounts or locating agents based on shares or
weights while optionally respecting capacities.
May also be thought of as pro-rating or benching
models.

"""
import logging

import numpy as np
import pandas as pd

from . import util as us_util
from ..utils.logutil import log_start_finish

logger = logging.getLogger(__name__)


class AllocationModel(object):
    """
    Used to allocate quantities or amounts to
    rows in a data frame.

    Parameters:
    ----------
    amounts_df: pandas.DataFrame
        Data frame containing amounts or quantities
        to allocate from. Assumed to be indexed
        by year.
    amounts_col: string
        Name of column in the amounts data frame
        to allocate from.
    target_col: string
        Name of column in the target data frame
        to allocate to.
    weight_col: string, optional
        Name of the column in the target data frame used to
        weight the allocation. If not provided all
        target rows will have an equal weight.
    capacity_col: string, optional
        Name of the column in the target data frame with
        maximum capacities that need to be respected. If not
        provided the allocation will be unconstrained.
    as_integer: bool, optional, default True
        If True, allocation values will be integerized to match
        the allocation amount exactly. If False then floating
        point values will be returned.
    as_delta: bool, optional, default False
        If True, the allocation will assume that the amount being
        distributed represents change. The resulting series
        will retain the original column values and will increment
        the column by the allocated change. If False, the allocation
        will be treated as a total and the target column will
        be entirely re-generated each year.
    compute_delta: bool optional, default False
        If True, the amounts are assumed
        to represent totals and changes will be computed
        by comparing the amount with the previous year, if
        available, or the most recent, previous allocation.
    segment_cols: list <string>
        List of field names that will be used for segmentation.
        Each segment will have its own allocation. The segment columns
        must exist, with the same names, on both the amounts and
        target data frames.
    floor_col: string optional, default None
        Name of column in the target frame defining the lowest
        allocation value that can be assigned to a given row
        (sort of like the inverse of capacity) when allocating changes.
        If not provided the floor_value argument will be used.
    floor_value: int optional, default 0
        Superseded by the floor_col argument. Defines the lowest
        possible value that can be assigned to any row in the allocation
        when allocating changes. This is important for declining amounts
        so that allocations will not fall below a given amount (usually 0).

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
                 segment_cols=None,
                 floor_col=None,
                 floor_value=0):
        self.amounts_df = amounts_df
        self.amounts_col = amounts_col
        self.target_col = target_col
        self.weight_col = weight_col
        self.capacity_col = capacity_col
        self.as_integer = as_integer
        self.as_delta = as_delta
        self.compute_delta = compute_delta
        self.segment_cols = segment_cols
        self.floor_col = floor_col
        self.floor_value = floor_value

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

        # check for floating amounts
        if self.as_integer:
            if (amounts_df[amounts_col] % 1 != 0).any():
                raise ValueError('Amounts table has floating values')

        # retain the previous amount for computing amount changes for missing years
        if segment_cols is None:
            self.prev_amount = 0
        else:
            pa = amounts_df.groupby(segment_cols).size() * 0
            self.prev_amount = pa.reset_index(name='amount')

    def allocate(self, data, year):
        """
        Performs the allocation for a given year.

        Parameters:
        ----------
        data: pandas.DataFrame
            Target data frame to allocate to.
        year: int
            Simulation year.

        Returns
        --------
        pandas.Series w/ allocated results.

        """
        logger.debug('start allocation for year {}'.format(year))
        if data is None or len(data) == 0:
            return None

        # set up the output series
        target_dtype = data[self.target_col].dtype
        if self.target_col in data and (self.as_delta or year not in self.amounts_df.index):
            results = data[self.target_col].copy()
        else:
            results = pd.Series(np.zeros(len(data), dtype=target_dtype), index=data.index)

        # maintain previous allocation if no amount provided for current year
        if year not in self.amounts_df.index:
            return results

        # loop through the amounts for the current year
        for _, curr_row in self.amounts_df.loc[[year]].iterrows():

            # get the current amount
            amount = curr_row[self.amounts_col]

            # adjust the amount for deltas
            if self.compute_delta:
                amount_total = amount

                if self.segment_cols is None:
                    if year - 1 in self.amounts_df.index:
                        # use the previous year if available
                        amount = amount - self.amounts_df.loc[year - 1][self.amounts_col]
                    else:
                        # otherwise use the most recent amount
                        amount = amount - self.prev_amount

                    # update the most recent amount
                    self.prev_amount = amount_total
                else:
                    # build a query for the current segmentation
                    prev_q = ""
                    for seg_col in self.segment_cols:
                        curr_val = curr_row[seg_col]
                        if isinstance(curr_val, basestring):
                            prev_q += "{} == '{}' and ".format(seg_col, curr_val)
                        else:
                            prev_q += "{} == {} and ".format(seg_col, curr_val)
                    prev_q = prev_q[:-4]

                    # get the index for the most recent amount w/ same segments
                    prev_row_idx = self.prev_amount.query(prev_q).index.values[0]

                    if year - 1 in self.amounts_df.index:
                        # use the previous year if available
                        prev_rows = self.amounts_df.query(prev_q)
                        amount = amount - prev_rows[self.amounts_col][year - 1]
                    else:
                        # otherwise use the most recent amount
                        amount = amount - self.prev_amount['amount'][prev_row_idx]

                    # update the most recent amount
                    self.prev_amount['amount'].loc[prev_row_idx] = amount_total

            if amount == 0:
                continue

            # get the subset of rows for the current segment
            if self.segment_cols is None:
                subset = data
                logger.debug('amount to allocate: {}'.format(amount))
            else:
                subset = us_util.filter_table(data, curr_row, ignore=self.ignore_cols)
                segment_str = ''
                for curr_seg in self.segment_cols:
                    segment_str += ' ' + str(curr_seg)
                logger.debug('on segment {}, amount to allocate: {}'.format(segment_str, amount))

            # get the existing series
            if self.as_delta:
                e = subset[self.target_col]
            else:
                e = pd.Series(np.zeros(len(subset)), index=subset.index)

            # get the weight series
            if self.weight_col is not None:
                w = subset[self.weight_col]
            else:
                w = pd.Series(np.ones(len(subset)), index=subset.index)

            # get the capacity series, capacity reflects max or min change
            if amount > 0:
                is_positive = True
                if self.capacity_col is not None:
                    c = subset[self.capacity_col] - e
                else:
                    c = pd.Series(np.ones(len(subset)) * amount, index=subset.index)
            else:
                is_positive = False
                amount *= -1
                if self.floor_col is not None:
                    c = subset[self.floor_col]
                else:
                    c = pd.Series(np.ones(len(subset)) * self.floor_value, index=subset.index)
                c = e - c

            if c.sum() < amount:
                # if not enough capacity, set all rows to their capacity
                a = c
                if is_positive:
                    logger.debug('Amount exceeds total available capacity ({})'.format(c.sum()))
                else:
                    logger.debug('Attempting to remove more than is available ({})'.format(c.sum()))
            else:
                # perform the initial allocation
                a = pd.Series(np.zeros(len(subset)), index=subset.index)
                while True:
                    # allocate remaining amount to rows with capacity
                    curr_amount = amount - a.sum()
                    have_cap = a < c
                    if have_cap.any() > 0:
                        w_sum = w[have_cap].sum()
                        if w_sum != 0:
                            a[have_cap] = a + (curr_amount * (w / w_sum))
                        else:
                            a[have_cap] = a + (curr_amount / len(a[have_cap]))

                    # set allocation result to capacity for overages
                    over = a > c
                    if over.any():
                        a[over] = c
                    else:
                        break

                # integerize using stochastic rounding
                if self.as_integer:
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
                        idx_to_round_up = np.random.choice(
                            to_round_idx, round_up_cnt, False, fract_w)
                        a[idx_to_round_up] += 1

            # update the results with the current allocation
            if is_positive:
                results[subset.index.values] = e + a
            else:
                results[subset.index.values] = e - a

        logger.debug('done with allocation')
        return results.astype(target_dtype)


class AgentAllocationModel(object):
    """
    Used to assign a set of agents to a set locations, based on
    locational weights and capacities. Similar in spirit  to the
    OPUS ScalingJobsModel and CapacityLocationModel models.

    Intended for location choice problems in which we (a) have
    limited data and therefore cannot estimate an MNL choice
    model, (b) the choice sets are fairly constrained or trivial
    or (c) there are specific, predefined controls or patterns
    that need to be realized by the simulation.

    Parameters:
    ----------
    location_col: string
        Name of the column on the agents data frame with foreign
        key reference to index on locations data frame.
    allocation_col: string
        Name of allocation column on locations data frame
        used to assign allocation results.
    weight_col: string, optional
        Name of the column in the locations data frame used to
        weight the allocation. If not provided all
        target rows will have an equal weight.
    capacity_col: string, optional
        Name of the column in the locations data frame with
        maximum capacities that need to be respected. If not
        provided the allocation will be unconstrained.
    segment_cols: list <string>, optional, default None
        List of field names that will be used for segmentation.
        Each segment will have its own allocation.

    """
    def __init__(self,
                 allocation_col,
                 weight_col=None,
                 capacity_col=None,
                 segment_cols=None):
        self.allocation_col = allocation_col
        self.weight_col = weight_col
        self.capacity_col = capacity_col
        self.segment_cols = segment_cols

    def locate_agents(self, locations, agents, year):
        """
        Assigns a location ID to a set of agents.

        Parameters:
        ----------

        locations: pandas.DataFrame
            Locations with index IDs to assign to agents.
        agents: pandas.DataFrame
            Agents that need to be assigned.
        year: int
            Simulation year. Not really used except to be
            consistent with other modules.

        Returns
        --------
        agent_loc_ids: pandas.Series
            Series containing the locations assigned to the agents.
        location_allocation: pandas.Series
            Series containing the updated number of agents assigned to each
            location. This is cumulative, the original values will be retained
            and additional agents located here will be added to these totals. The
            series must be applied to update the locations data frame in order for
            capacity constraints to be respected over multiple simulation years.

        """
        # init the agent locations with NaNs
        agent_loc_ids = pd.Series(np.ones(len(agents)) * np.nan, index=agents.index)

        # create the allocation controls from the agent distribution
        if self.segment_cols is None:
            amounts_df = pd.DataFrame(
                {'amount': [len(agents)]},
                index=[year]
                )
        else:
            agent_cnts = agents.groupby(self.segment_cols).size()
            amounts_df = agent_cnts.reset_index(name='amount')
            amounts_df.index = pd.Index(np.ones(len(amounts_df)) * year)

        # allocate agent quantities to locations (cumulative)
        a_mod = AllocationModel(amounts_df,
                                'amount',
                                self.allocation_col,
                                self.weight_col,
                                self.capacity_col,
                                as_delta=True,
                                segment_cols=self.segment_cols)
        location_allocation = a_mod.allocate(locations, year)
        curr_allocation = location_allocation - locations[self.allocation_col]

        # assign location IDs to agents for each segment
        for _, curr_row in amounts_df.loc[[year]].iterrows():
            # get agents and locations for current segment
            if self.segment_cols is not None:
                agent_subset = us_util.filter_table(agents, curr_row, ignore=a_mod.ignore_cols)
                loc_subset = us_util.filter_table(locations, curr_row, ignore=a_mod.ignore_cols)
            else:
                agent_subset = agents
                loc_subset = locations

            # assign the location IDs to the agents randomly
            loc_subset_idx = loc_subset.index.values
            loc_repeat = np.repeat(loc_subset_idx, curr_allocation[loc_subset_idx])

            agent_idx = agent_subset.index.values
            agent_idx = np.random.permutation(agent_idx)
            agent_overage = len(agent_subset) - curr_allocation[loc_subset.index.values].sum()
            if agent_overage > 0:
                # not enough capacity, keep some agents un-located
                agent_idx = agent_idx[agent_overage:]
            agent_loc_ids[agent_idx] = loc_repeat

        return agent_loc_ids, location_allocation
