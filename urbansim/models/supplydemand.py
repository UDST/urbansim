"""
Tools for modeling how supply and demand affect real estate prices.

"""
import numpy as np
import pandas as pd


def _calculate_adjustment_ratio(
        lcm, choosers, alternatives, alt_segmenter,
        clip_change_low, clip_change_high):
    """
    Calculate adjustments to prices to compensate for
    supply and demand effects.

    Parameters
    ----------
    lcm : LocationChoiceModel
        Used to calculate the probability of agents choosing among
        alternatives. Must be fully configured and fitted.
    choosers : pandas.DataFrame
    alternatives : pandas.DataFrame
    alt_segmenter : pandas.Series
        Will be used to segment alternatives and probabilities to do
        comparisons of supply and demand by submarket.
    clip_change_low : float
        The minimum amount by which to multiply prices each iteration.
    clip_change_high : float
        The maximum amount by which to multiply prices each iteration.

    Returns
    -------
    ratio : pandas.Series
        Same index as `alternatives`, values clipped to `clip_change_low`
        and `clip_change_high`.

    """
    # probabilities of agents choosing * number of agents = demand
    demand = pd.Series(lcm.summed_probabilities(choosers, alternatives))
    # group by submarket
    demand = demand.groupby(alt_segmenter.values).sum()

    # number of alternatives
    supply = alt_segmenter.value_counts()

    ratio = (demand / supply).clip(clip_change_low, clip_change_high)

    # broadcast ratio back to alternatives index
    ratio = ratio.loc[alt_segmenter]
    ratio.index = alt_segmenter.index

    return ratio


def supply_and_demand(
        lcm, choosers, alternatives, alt_segmenter, price_col,
        clip_change_low=0.75, clip_change_high=1.25, iterations=5):
    """
    Adjust real estate prices to compensate for supply and demand effects.

    Parameters
    ----------
    lcm : LocationChoiceModel
        Used to calculate the probability of agents choosing among
        alternatives. Must be fully configured and fitted.
    choosers : pandas.DataFrame
    alternatives : pandas.DataFrame
    alt_segmenter : str, array, or pandas.Series
        Will be used to segment alternatives and probabilities to do
        comparisons of supply and demand by submarket.
        If a string, it is expected to be the name of a column
        in `alternatives`.
    price_col : str
        The name of the column in `alternatives` that corresponds to price.
        This column is what is adjusted by this model.
    clip_change_low : float, optional
        The minimum amount by which to multiply prices each iteration.
    clip_change_high : float, optional
        The maximum amount by which to multiply prices each iteration.
    iterations : int, optional
        Number of times to update prices based on supply/demand comparisons.

    Returns
    -------
    new_prices : pandas.Series
        Equivalent of the `price_col` in `alternatives`.

    """
    # copy alternatives so we don't modify the user's original
    alternatives = alternatives.copy()

    # if alt_segmenter is a string, get the actual column for segmenting demand
    if isinstance(alt_segmenter, str):
        alt_segmenter = alternatives[alt_segmenter]
    elif isinstance(alt_segmenter, np.array):
        alt_segmenter = pd.Series(alt_segmenter)

    for _ in range(iterations):
        ratio = _calculate_adjustment_ratio(
            lcm, choosers, alternatives, alt_segmenter,
            clip_change_low, clip_change_high)
        alternatives[price_col] = alternatives[price_col] * ratio

    return alternatives[price_col]
