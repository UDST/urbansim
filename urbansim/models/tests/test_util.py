import string

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
    pdt.assert_frame_equal(filtered, expected, check_dtype=False)


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


def test_apply_filter_query_str(test_df):
    filters = 'col1 < 3'
    filtered = util.apply_filter_query(test_df, filters)
    expected = pd.DataFrame(
        {'col1': [0, 1, 2],
         'col2': [5, 6, 7]},
        index=['a', 'b', 'c'])
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


def test_has_constant_expr():
    assert util.has_constant_expr('a + b') is False
    assert util.has_constant_expr('a +   1 + b') is True
    assert util.has_constant_expr('a - 1 + b') is True
    assert util.has_constant_expr('-1 + a + b') is True
    assert util.has_constant_expr('a + b +1') is True


class Test_str_model_expression(object):
    @classmethod
    def setup_class(cls):
        left_side = 'np.log1p(x)'
        cls.rs_expected = 'np.log1p(y) + I((x + y) < z) + 1'
        cls.full_expected = ' ~ '.join((left_side, cls.rs_expected))
        cls.rs_expected_no_const = 'np.log1p(y) + I((x + y) < z) - 1'
        cls.full_expected_no_const = ' ~ '.join(
            (left_side, cls.rs_expected_no_const))

    def test_string(self):
        assert util.str_model_expression(self.full_expected) == \
            self.full_expected

    def test_string_ignores_add_constant(self):
        assert util.str_model_expression(
            self.full_expected_no_const, add_constant=True
            ) == self.full_expected_no_const

    def test_list(self):
        expr_list = ['np.log1p(y)', 'I((x + y) < z)']
        assert util.str_model_expression(expr_list) == self.rs_expected
        assert util.str_model_expression(
            expr_list, add_constant=False) == self.rs_expected_no_const

    def test_dict_right_only(self):
        expr_dict = {'right_side': ['np.log1p(y)', 'I((x + y) < z)']}
        assert util.str_model_expression(expr_dict) == self.rs_expected
        assert util.str_model_expression(
            expr_dict, add_constant=False) == self.rs_expected_no_const

    def test_dict_full(self):
        expr_dict = {
            'left_side': 'np.log1p(x)',
            'right_side': ['np.log1p(y)', 'I((x + y) < z)']}
        assert util.str_model_expression(expr_dict) == self.full_expected
        assert util.str_model_expression(
            expr_dict, add_constant=False) == self.full_expected_no_const


def test_sorted_groupby():
    df = pd.DataFrame(
        {'alpha': np.random.choice(list(string.lowercase), 100),
         'num': np.random.randint(100)})
    sorted_df = df.sort('alpha')

    expected = {name: d.to_dict() for name, d in df.groupby('alpha')}
    test = {name: d.to_dict()
            for name, d in util.sorted_groupby(sorted_df, 'alpha')}

    assert test == expected


def test_columns_in_filters():
    assert util.columns_in_filters([]) == []

    filters = ['a > 5', '10 < b < 20', 'c == "x"', '(d > 20 or e < 50)']
    assert set(util.columns_in_filters(filters)) == {'a', 'b', 'c', 'd', 'e'}

    filters = 'a > 5 and 10 < b < 20 and c == "x" and (d > 20 or e < 50)'
    assert set(util.columns_in_filters(filters)) == {'a', 'b', 'c', 'd', 'e'}

    filters = 'a in [1, 2, 3] or b not in ["x", "y", "z"]'
    assert set(util.columns_in_filters(filters)) == {'a', 'b'}


def test_columns_in_formula():
    formula = 'a ~ b + c - d * e:f'
    assert set(util.columns_in_formula(formula)) == \
        {'a', 'b', 'c', 'd', 'e', 'f'}

    formula = ('np.log(x) + I(y > 9000) + C(z) + I(w - x < 0) + '
               'np.exp((u * np.pow(v, 2)))')
    assert set(util.columns_in_formula(formula)) == \
        {'u', 'v', 'w', 'x', 'y', 'z'}
