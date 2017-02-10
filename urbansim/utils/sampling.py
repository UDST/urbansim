import math
import numpy as np
import pandas as pd


def get_probs(data, prob_column=None):
    """
    Checks for presence of a probability column and returns the result
    as a numpy array. If the probabilities are weights (i.e. they don't
    sum to 1), then this will be recalculated.

    Parameters
    ----------
    data: pandas.DataFrame
        Table to sample from.
    prob_column: string, optional, default None
        Name of the column in the data to provide probabilities or weights.

    Returns
    -------
    numpy.array

    """
    if prob_column is None:
        p = None
    else:
        p = data[prob_column].fillna(0).values
        if p.sum() == 0:
            p = np.ones(len(p))
        if abs(p.sum() - 1.0) > 1e-8:
            p = p / (1.0 * p.sum())
    return p


def accounting_sample_replace(total, data, accounting_column, prob_column=None, max_iterations=50):
    """
    Sample rows with accounting with replacement.

    Parameters
    ----------
    total : int
        The control total the sampled rows will attempt to match.
    data: pandas.DataFrame
        Table to sample from.
    accounting_column: string
        Name of column with accounting totals/quantities to apply towards the control.
    prob_column: string, optional, default None
        Name of the column in the data to provide probabilities or weights.
    max_iterations: int, optional, default 50
        When using an accounting attribute, the maximum number of sampling iterations
        that will be applied.

    Returns
    -------
    sample_rows : pandas.DataFrame
        Table containing the sample.
    matched: bool
        Indicates if the total was matched exactly.

    """
    # check for probabilities
    p = get_probs(data, prob_column)

    # determine avg number of accounting items per sample (e.g. persons per household)
    per_sample = data[accounting_column].sum() / (1.0 * len(data.index.values))

    curr_total = 0
    remaining = total
    sample_rows = pd.DataFrame()
    closest = None
    closest_remain = total
    matched = False

    for i in range(0, max_iterations):

        # stop if we've hit the control
        if remaining == 0:
            matched = True
            break

        # if sampling with probabilities, re-caclc the # of items per sample
        # after the initial sample, this way the sample size reflects the probabilities
        if p is not None and i == 1:
            per_sample = sample_rows[accounting_column].sum() / (1.0 * len(sample_rows))

        # update the sample
        num_samples = int(math.ceil(math.fabs(remaining) / per_sample))

        if remaining > 0:
            # we're short, add to the sample
            curr_ids = np.random.choice(data.index.values, num_samples, p=p)
            sample_rows = pd.concat([sample_rows, data.loc[curr_ids]])
        else:
            # we've overshot, remove from existing samples (FIFO)
            sample_rows = sample_rows.iloc[num_samples:].copy()

        # update the total and check for the closest result
        curr_total = sample_rows[accounting_column].sum()
        remaining = total - curr_total

        if abs(remaining) < closest_remain:
            closest_remain = abs(remaining)
            closest = sample_rows

    return closest, matched


def accounting_sample_no_replace(total, data, accounting_column, prob_column=None):
    """
    Samples rows with accounting without replacement.

    Parameters
    ----------
    total : int
        The control total the sampled rows will attempt to match.
    data: pandas.DataFrame
        Table to sample from.
    accounting_column: string
        Name of column with accounting totals/quantities to apply towards the control.
    prob_column: string, optional, default None
        Name of the column in the data to provide probabilities or weights.

    Returns
    -------
    sample_rows : pandas.DataFrame
        Table containing the sample.
    matched: bool
        Indicates if the total was matched exactly.

    """
    # make sure this is even feasible
    if total > data[accounting_column].sum():
        raise ValueError('Control total exceeds the available samples')

    # check for probabilities
    p = get_probs(data, prob_column)

    # shuffle the rows
    if p is None:
        # random shuffle
        shuff_idx = np.random.permutation(data.index.values)
    else:
        # weighted shuffle
        ran_p = pd.Series(np.power(np.random.rand(len(p)), 1.0 / p), index=data.index)
        ran_p.sort(ascending=False)
        shuff_idx = ran_p.index.values

    # get the initial sample
    shuffle = data.loc[shuff_idx]
    csum = np.cumsum(shuffle[accounting_column].values)
    pos = np.searchsorted(csum, total, 'right')
    sample = shuffle.iloc[:pos]

    # refine the sample
    sample_idx = sample.index.values
    sample_total = sample[accounting_column].sum()
    shortage = total - sample_total
    matched = False

    for idx, row in shuffle.iloc[pos:].iterrows():
        if shortage == 0:
            # we've matached
            matched = True
            break

        # add the current element if it doesnt exceed the total
        cnt = row[accounting_column]
        if cnt <= shortage:
            sample_idx = np.append(sample_idx, idx)
            shortage -= cnt

    return shuffle.loc[sample_idx].copy(), matched


def sample_rows(total, data, replace=True, accounting_column=None,
                max_iterations=50, prob_column=None, return_status=False):
    """
    Samples and returns rows from a data frame while matching a desired control total. The total may
    represent a simple row count or may attempt to match a sum/quantity from an accounting column.

    Parameters
    ----------
    total : int
        The control total the sampled rows will attempt to match.
    data: pandas.DataFrame
        Table to sample from.
    replace: bool, optional, default True
        Indicates if sampling with or without replacement.
    accounting_column: string, optional
        Name of column with accounting totals/quantities to apply towards the control.
        If not provided then row counts will be used for accounting.
    max_iterations: int, optional, default 50
        When using an accounting attribute, the maximum number of sampling iterations
        that will be applied. Only applicable when sampling with replacement.
    prob_column: string, optional, default None
        If provided, name of the column in the data frame to provide probabilities
        or weights. If not provided, the sampling is random.
    return_status: bool, optional, default True
        If True, will also return a bool indicating if the total was matched exactly.

    Returns
    -------
    sample_rows : pandas.DataFrame
        Table containing the sample.
    matched: bool
        If return_status is True, returns True if total is matched exactly.

    """
    if not data.index.is_unique:
        raise ValueError('Data must have a unique index')

    # simplest case, just return n random rows
    if accounting_column is None:
        if replace is False and total > len(data.index.values):
            raise ValueError('Control total exceeds the available samples')
        p = get_probs(prob_column)
        rows = data.loc[np.random.choice(
            data.index.values, int(total), replace=replace, p=p)].copy()
        matched = True

    # sample with accounting
    else:
        if replace:
            rows, matched = accounting_sample_replace(
                total, data, accounting_column, prob_column, max_iterations)
        else:
            rows, matched = accounting_sample_no_replace(
                total, data, accounting_column, prob_column)

    # return the results
    if return_status:
        return rows, matched
    else:
        return rows
