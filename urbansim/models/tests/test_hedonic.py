import pandas as pd
import pytest
from pandas.util import testing as pdt

from statsmodels.regression.linear_model import RegressionResultsWrapper

from .. import hedonic


@pytest.fixture
def test_df():
    return pd.DataFrame(
        {'col1': range(5),
         'col2': range(5, 10)},
        index=['a', 'b', 'c', 'd', 'e'])


def test_apply_filter_query(test_df):
    filters = ['col1 < 3', 'col2 > 6']
    filtered = hedonic.apply_filter_query(test_df, filters)
    expected = pd.DataFrame(
        {'col1': [2], 'col2': [7]},
        index=['c'])
    pdt.assert_frame_equal(filtered, expected)


def test_apply_filter_query_empty(test_df):
    filters = ['col1 < 1', 'col2 > 8']
    filtered = hedonic.apply_filter_query(test_df, filters)
    expected = pd.DataFrame(
        {'col1': [], 'col2': []},
        index=[])
    pdt.assert_frame_equal(filtered, expected)


def test_apply_filter_query_or(test_df):
    filters = ['col1 < 1 or col2 > 8']
    filtered = hedonic.apply_filter_query(test_df, filters)
    expected = pd.DataFrame(
        {'col1': [0, 4], 'col2': [5, 9]},
        index=['a', 'e'])
    pdt.assert_frame_equal(filtered, expected)


def test_apply_filter_query_no_filter(test_df):
    filters = []
    filtered = hedonic.apply_filter_query(test_df, filters)
    expected = test_df
    pdt.assert_frame_equal(filtered, expected)


def test_fit_model(test_df):
    filters = []
    model_exp = 'col1 ~ col2'
    fit = hedonic.fit_model(test_df, filters, model_exp)
    assert isinstance(fit, RegressionResultsWrapper)


def test_predict(test_df):
    filters = ['col1 in [0, 2, 4]']
    model_exp = 'col1 ~ col2'
    fit = hedonic.fit_model(test_df, filters, model_exp)
    predicted = hedonic.predict(
        test_df.query('col1 in [1, 3]'), None, fit)
    expected = pd.Series([1., 3.], index=['b', 'd'])
    pdt.assert_series_equal(predicted, expected)


def test_predict_ytransform(test_df):
    yt = lambda x: x / 2.
    filters = ['col1 in [0, 2, 4]']
    model_exp = 'col1 ~ col2'
    fit = hedonic.fit_model(test_df, filters, model_exp)
    predicted = hedonic.predict(
        test_df.query('col1 in [1, 3]'), None, fit, ytransform=yt)
    expected = pd.Series([0.5, 1.5], index=['b', 'd'])
    pdt.assert_series_equal(predicted, expected)
