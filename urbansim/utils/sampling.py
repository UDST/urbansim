import math
import numpy as np
import pandas as pd


def sample_rows(total, data, replace=True, accounting_column=None, max_iterations=50):
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
        that will be applied.

    Returns
    -------
    sample_rows : pandas.DataFrame
        Table containing the sample.
    """

    # simplest case, just return n random rows
    if accounting_column is None:
        if replace is False and total > len(data.index.values):
            raise ValueError('Control total exceeds the available samples')
        return data.loc[np.random.choice(data.index.values, total, replace=replace)].copy()

    # make sure this is even feasible
    if replace is False and total > data[accounting_column].sum():
        raise ValueError('Control total exceeds the available samples')

    # determine avg number of accounting items per sample (e.g. persons per household)
    per_sample = data[accounting_column].sum() / (1.0 * len(data.index.values))

    # do the initial sample
    num_samples = int(math.ceil(total / per_sample))
    if replace:
        sample_idx = data.index.values
        sample_ids = np.random.choice(sample_idx, num_samples)
    else:
        sample_idx = np.random.permutation(data.index.values)
        sample_ids = sample_idx[0:num_samples]
        sample_pos = num_samples

    sample_rows = data.loc[sample_ids].copy()
    curr_total = sample_rows[accounting_column].sum()

    # iteratively refine the sample until we match the accounting total
    for i in range(0, max_iterations):

        # keep going if we haven't hit the control
        remaining = total - curr_total
        if remaining == 0:
            break
        num_samples = int(math.ceil(math.fabs(remaining) / per_sample))

        if remaining > 0:
            # we're short, keep sampling
            if replace:
                curr_ids = np.random.choice(sample_idx, num_samples)
            else:
                curr_ids = sample_idx[sample_pos:sample_pos + num_samples]
                sample_pos += num_samples

            curr_rows = data.loc[curr_ids].copy()
            sample_rows = pd.concat([sample_rows, curr_rows])
            curr_total += curr_rows[accounting_column].sum()
        else:
            # we've overshot, remove from existing samples (FIFO)
            curr_rows = sample_rows[:num_samples]
            sample_rows = sample_rows[num_samples:]
            curr_total -= curr_rows[accounting_column].sum()
            if not replace:
                np.append(sample_idx, curr_rows.index.values)

    return sample_rows.copy()
