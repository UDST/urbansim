import numpy as np
import pandas as pd
import pytest
from pandas.util import testing as pdt

from .. import util


@pytest.fixture
def test_df():
    return pd.DataFrame(
        {'col1': range(5),
         'col2': range(5, 10)},
        index=['a', 'b', 'c', 'd', 'e'])


@pytest.fixture
def choosers():
    return pd.DataFrame(
        {'var1': range(5),
         'var2': range(5, 10),
         'var3': ['q', 'w', 'e', 'r', 't'],
         'building_id': range(100, 105)},
        index=['a', 'b', 'c', 'd', 'e'])


@pytest.fixture
def rates():
    return pd.DataFrame(
        {'var1_min': [np.nan, np.nan, np.nan],
         'var1_max': [1, np.nan, np.nan],
         'var2_min': [np.nan, 7, np.nan],
         'var2_max': [np.nan, 8, np.nan],
         'var3': [np.nan, np.nan, 't'],
         'probability_of_relocating': [1, 1, 1]})


def test_apply_filter_query(test_df):
    filters = ['col1 < 3', 'col2 > 6']
    filtered = util.apply_filter_query(test_df, filters)
    expected = pd.DataFrame(
        {'col1': [2], 'col2': [7]},
        index=['c'])
    pdt.assert_frame_equal(filtered, expected)


def test_apply_filter_query_empty(test_df):
    filters = ['col1 < 1', 'col2 > 8']
    filtered = util.apply_filter_query(test_df, filters)
    expected = pd.DataFrame(
        {'col1': [], 'col2': []},
        index=[])
    pdt.assert_frame_equal(filtered, expected)


def test_apply_filter_query_or(test_df):
    filters = ['col1 < 1 or col2 > 8']
    filtered = util.apply_filter_query(test_df, filters)
    expected = pd.DataFrame(
        {'col1': [0, 4], 'col2': [5, 9]},
        index=['a', 'e'])
    pdt.assert_frame_equal(filtered, expected)


def test_apply_filter_query_no_filter(test_df):
    filters = []
    filtered = util.apply_filter_query(test_df, filters)
    expected = test_df
    pdt.assert_frame_equal(filtered, expected)


@pytest.mark.parametrize('name, val, filter_exp', [
    ('x', 1, 'x == 1'),
    ('x', 'a', "x == 'a'"),
    ('y_min', 2, 'y >= 2'),
    ('z_max', 3, 'z < 3')])
def test_filterize(name, val, filter_exp):
    assert util._filterize(name, val) == filter_exp


def test_filter_table(choosers, rates):
    filtered = util.filter_table(
        choosers, rates.iloc[1], ignore={'probability_of_relocating'})
    pdt.assert_frame_equal(filtered, choosers.iloc[[2]])
