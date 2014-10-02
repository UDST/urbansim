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


@pytest.fixture
def alt_segmenter():
    return 'zone_id'


class _TestLCM(object):
    def summed_probabilities(self, choosers, alternatives):
        return [
            1, 0.25, 1, 2, 1, 0.75, 2, 1, 1.5, 0.5]
        #   w, x,    y, z, z, x,    y, w, y,   y


@pytest.fixture(scope='module')
def lcm():
    return _TestLCM()


def test_calculate_adjustment_ratio_clips(
        lcm, choosers, alternatives, alt_segmenter):
    clip = 1

    alt_ratio, submarket_ratio = supdem._calculate_adjustment_ratio(
        lcm, choosers, alternatives, alternatives[alt_segmenter], clip, clip)

    pdt.assert_series_equal(
        alt_ratio, pd.Series([1] * 10, index=alternatives.index),
        check_dtype=False)
    pdt.assert_series_equal(
        submarket_ratio, pd.Series([1] * 4, index=['w', 'x', 'y', 'z']),
        check_dtype=False)


def test_calculate_adjustment_ratio(
        lcm, choosers, alternatives, alt_segmenter):
    clip_low = 0
    clip_high = 2

    alt_ratio, submarket_ratio = supdem._calculate_adjustment_ratio(
        lcm, choosers, alternatives, alternatives[alt_segmenter],
        clip_low, clip_high)

    w = 1
    x = 0.5
    y = 1.25
    z = 3 / 2

    pdt.assert_series_equal(
        alt_ratio,
        pd.Series([w, x, y, z, z, x, y, w, y, y],
                  index=alternatives.index))
    pdt.assert_series_equal(
        submarket_ratio,
        pd.Series([w, x, y, z], index=['w', 'x', 'y', 'z']))


def test_supply_and_demand(
        lcm, choosers, alternatives, alt_segmenter):
    clip_low = 0
    clip_high = 2
    price_col = 'price_col'

    w = 1
    x = 0.5
    y = 1.25
    z = 3 / 2

    new_price, submarket_ratio = supdem.supply_and_demand(
        lcm, choosers, alternatives, alt_segmenter, price_col,
        clip_low, clip_high)

    pdt.assert_series_equal(
        new_price,
        pd.Series(
            [w, x, y, z, z, x, y, w, y, y],
            index=alternatives.index) ** 5)
    pdt.assert_series_equal(
        submarket_ratio,
        pd.Series([w, x, y, z], index=['w', 'x', 'y', 'z']))
