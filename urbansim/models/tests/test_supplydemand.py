from __future__ import division

import pandas as pd
import pytest
from pandas.util import testing as pdt

from .. import supplydemand as supdem


@pytest.fixture
def choosers():
    return pd.DataFrame(
        {'var1': range(5, 10),
         'thing_id': ['a', 'c', 'e', 'g', 'i']})


@pytest.fixture
def alternatives():
    return pd.DataFrame(
        {'var2': range(10, 20),
         'var3': range(20, 30),
         'price_col': [1] * 10,
         'zone_id': ['w', 'x', 'y', 'z', 'z', 'x', 'y', 'w', 'y', 'y']},
        index=pd.Index([x for x in 'abcdefghij'], name='thing_id'))


@pytest.fixture(scope='module')
def alt_segmenter():
    return 'zone_id'


class _TestLCM(object):
    def apply_predict_filters(self, choosers, alternatives):
        choosers = choosers.query('var1 != 7')
        alternatives = alternatives.query('var2 != 14')
        return choosers, alternatives

    def summed_probabilities(self, choosers, alternatives):
        return pd.Series(
            [1, 0.25, 1, 2, 0.75, 2, 1, 1.5, 0.5],
            index=['a', 'b', 'c', 'd', 'f', 'g', 'h', 'i', 'j'])


@pytest.fixture(scope='module')
def lcm():
    return _TestLCM()


@pytest.fixture
def filtered(lcm, choosers, alternatives):
    return lcm.apply_predict_filters(choosers, alternatives)


@pytest.fixture(scope='module')
def wxyz():
    w = 1
    x = 0.5
    y = 1.25
    z = 2
    return w, x, y, z


def test_calculate_adjustment_clips(lcm, filtered, alt_segmenter):
    clip = 1

    choosers, alternatives = filtered

    alts_multiplier, submarkets_multiplier, finished = \
        supdem._calculate_adjustment(
            lcm, choosers, alternatives, alternatives[alt_segmenter],
            clip, clip)

    pdt.assert_series_equal(
        alts_multiplier, pd.Series([1] * 9, index=alternatives.index),
        check_dtype=False)
    pdt.assert_series_equal(
        submarkets_multiplier, pd.Series([1] * 4, index=['w', 'x', 'y', 'z']),
        check_dtype=False)


def test_calculate_adjustment(lcm, filtered, alt_segmenter, wxyz):
    clip_low = 0
    clip_high = 2

    choosers, alternatives = filtered

    alts_multiplier, submarkets_multiplier, finished = \
        supdem._calculate_adjustment(
            lcm, choosers, alternatives, alternatives[alt_segmenter],
            clip_low, clip_high)

    w, x, y, z = wxyz

    pdt.assert_series_equal(
        alts_multiplier,
        pd.Series([w, x, y, z, x, y, w, y, y],
                  index=alternatives.index))
    pdt.assert_series_equal(
        submarkets_multiplier,
        pd.Series([w, x, y, z], index=['w', 'x', 'y', 'z']))


def test_supply_and_demand(
        lcm, choosers, alternatives, alt_segmenter, filtered, wxyz):
    clip_low = 0
    clip_high = 2
    price_col = 'price_col'

    w, x, y, z = wxyz

    filtered_choosers, filtered_alts = filtered

    new_price, submarkets_multiplier = supdem.supply_and_demand(
        lcm, choosers, alternatives, alt_segmenter, price_col,
        clip_change_low=clip_low, clip_change_high=clip_high)

    pdt.assert_series_equal(
        new_price,
        pd.Series(
            [w, x, y, z, x, y, w, y, y],
            index=filtered_alts.index, name='price_col') ** 5)
    pdt.assert_series_equal(
        submarkets_multiplier,
        pd.Series([w, x, y, z], index=['w', 'x', 'y', 'z']) ** 5)


def test_supply_and_demand_base_ratio(
        lcm, choosers, alternatives, alt_segmenter, filtered, wxyz):
    clip_low = 0
    clip_high = 2
    price_col = 'price_col'

    w, x, y, z = wxyz

    filtered_choosers, filtered_alts = filtered

    base_multiplier = pd.Series([w, x, y, z], index=['w', 'x', 'y', 'z'])

    new_price, submarkets_multiplier = supdem.supply_and_demand(
        lcm, choosers, alternatives, alt_segmenter, price_col,
        base_multiplier, clip_low, clip_high)

    pdt.assert_series_equal(
        new_price,
        pd.Series(
            [w, x, y, z, x, y, w, y, y],
            index=filtered_alts.index, name='price_col') ** 6)
    pdt.assert_series_equal(
        submarkets_multiplier,
        pd.Series([w, x, y, z], index=['w', 'x', 'y', 'z']) ** 6)
