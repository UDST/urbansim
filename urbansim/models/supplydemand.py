"""
Tools for modeling how supply and demand affect real estate prices.

"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _calculate_adjustment(
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
    alts_muliplier : pandas.Series
        Same index as `alternatives`, values clipped to `clip_change_low`
        and `clip_change_high`.
    submarkets_multiplier : pandas.Series
        Index is unique values from `alt_segmenter`, values are the ratio
        of demand / supply for each segment in `alt_segmenter`.

    """
    logger.debug('start: calculate supply and demand price adjustment ratio')
    # probabilities of agents choosing * number of agents = demand
    demand = lcm.summed_probabilities(choosers, alternatives)
    # group by submarket
    demand = demand.groupby(alt_segmenter.loc[demand.index].values).sum()

    # number of alternatives
    supply = alt_segmenter.value_counts()

    multiplier = (demand / supply).clip(clip_change_low, clip_change_high)

    # broadcast multiplier back to alternatives index
    alts_muliplier = multiplier.loc[alt_segmenter]
    alts_muliplier.index = alt_segmenter.index

    logger.debug(
        ('finish: calculate supply and demand price adjustment multiplier '
         'with mean multiplier {}').format(multiplier.mean()))
    return alts_muliplier, multiplier


def supply_and_demand(
        lcm, choosers, alternatives, alt_segmenter, price_col,
        base_multiplier=None, clip_change_low=0.75, clip_change_high=1.25,
        iterations=5):
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
        in `alternatives`. If a Series it should have the same index
        as `alternatives`.
    price_col : str
        The name of the column in `alternatives` that corresponds to price.
        This column is what is adjusted by this model.
    base_multiplier : pandas.Series, optional
        A series describing a starting multiplier for submarket prices.
        Index should be submarket IDs.
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
    submarkets_ratios : pandas.Series
        Price adjustment ratio for each submarket. If `base_multiplier` is
        given this will be a cummulative multiplier including the
        `base_multiplier` and the multipliers calculated for this year.

    """
    logger.debug('start: calculating supply and demand price adjustment')
    # copy alternatives so we don't modify the user's original
    alternatives = alternatives.copy()

    # if alt_segmenter is a string, get the actual column for segmenting demand
    if isinstance(alt_segmenter, str):
        alt_segmenter = alternatives[alt_segmenter]
    elif isinstance(alt_segmenter, np.array):
        alt_segmenter = pd.Series(alt_segmenter, index=alternatives.index)

    choosers, alternatives = lcm.apply_predict_filters(choosers, alternatives)
    alt_segmenter = alt_segmenter.loc[alternatives.index]

    # check base ratio and apply it to prices if given
    if base_multiplier is not None:
        bm = base_multiplier.loc[alt_segmenter]
        bm.index = alt_segmenter.index
        alternatives[price_col] = alternatives[price_col] * bm
        base_multiplier = base_multiplier.copy()

    for _ in range(iterations):
        alts_muliplier, submarkets_multiplier = _calculate_adjustment(
            lcm, choosers, alternatives, alt_segmenter,
            clip_change_low, clip_change_high)
        alternatives[price_col] = alternatives[price_col] * alts_muliplier

        # might need to initialize this for holding cumulative multiplier
        if base_multiplier is None:
            base_multiplier = pd.Series(
                np.ones(len(submarkets_multiplier)),
                index=submarkets_multiplier.index)

        base_multiplier *= submarkets_multiplier

    logger.debug('finish: calculating supply and demand price adjustment')
    return alternatives[price_col], base_multiplier
