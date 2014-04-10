import pandas as pd
import pytest
from pandas.util import testing as pdt

from statsmodels.regression.linear_model import RegressionResultsWrapper

from .. import hedonic
from ...exceptions import ModelEvaluationError


@pytest.fixture
def test_df():
    return pd.DataFrame(
        {'col1': range(5),
         'col2': range(5, 10)},
        index=['a', 'b', 'c', 'd', 'e'])


@pytest.fixture
def groupby_df(test_df):
    test_df['group'] = ['x', 'y', 'x', 'x', 'y']
    return test_df


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


def test_predict_with_nans():
    df = pd.DataFrame(
        {'col1': range(5),
         'col2': [5, 6, pd.np.nan, 8, 9]},
        index=['a', 'b', 'c', 'd', 'e'])
    fit = hedonic.fit_model(df.loc[['a', 'b', 'e']], None, 'col1 ~ col2')

    with pytest.raises(ModelEvaluationError):
        hedonic.predict(
            df.loc[['c', 'd']], None, fit)


def test_HedonicModel(test_df):
    fit_filters = ['col1 in [0, 2, 4]']
    predict_filters = ['col1 in [1, 3]']
    model_exp = 'col1 ~ col2'
    ytransform = lambda x: x / 2.
    name = 'test hedonic'

    model = hedonic.HedonicModel(
        fit_filters, predict_filters, model_exp, ytransform, name)
    assert model.fit_filters == fit_filters
    assert model.predict_filters == predict_filters
    assert model.model_expression == model_exp
    assert model.ytransform == ytransform
    assert model.name == name
    assert model.model_fit is None

    # verify there's an error if there isn't a model fit yet
    with pytest.raises(RuntimeError):
        model.predict(test_df)

    fit = model.fit_model(test_df)
    assert isinstance(fit, RegressionResultsWrapper)
    assert isinstance(model.model_fit, RegressionResultsWrapper)

    predicted = model.predict(test_df)
    expected = pd.Series([0.5, 1.5], index=['b', 'd'])
    pdt.assert_series_equal(predicted, expected)


def test_HedonicModelGroup(groupby_df):
    model_exp = 'col1 ~ col2'

    hmg = hedonic.HedonicModelGroup('group')

    xmodel = hedonic.HedonicModel(None, None, model_exp, name='x')
    hmg.add_model(xmodel)
    assert isinstance(hmg.models['x'], hedonic.HedonicModel)

    hmg.add_model_from_params('y', None, None, model_exp)
    assert isinstance(hmg.models['y'], hedonic.HedonicModel)
    assert hmg.models['y'].name == 'y'

    fits = hmg.fit_models(groupby_df)
    assert isinstance(fits['x'], RegressionResultsWrapper)
    assert isinstance(fits['y'], RegressionResultsWrapper)

    predicted = hmg.predict(groupby_df)
    assert isinstance(predicted, pd.Series)
    pdt.assert_series_equal(
        predicted.sort_index(), groupby_df.col1, check_dtype=False)
