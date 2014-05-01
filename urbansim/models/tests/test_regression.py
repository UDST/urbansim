import numpy.testing as npt
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from pandas.util import testing as pdt

from statsmodels.regression.linear_model import RegressionResultsWrapper

from .. import regression
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


def test_fit_model(test_df):
    filters = []
    model_exp = 'col1 ~ col2'
    fit = regression.fit_model(test_df, filters, model_exp)
    assert isinstance(fit, RegressionResultsWrapper)


def test_predict(test_df):
    filters = ['col1 in [0, 2, 4]']
    model_exp = 'col1 ~ col2'
    fit = regression.fit_model(test_df, filters, model_exp)
    predicted = regression.predict(
        test_df.query('col1 in [1, 3]'), None, fit)
    expected = pd.Series([1., 3.], index=['b', 'd'])
    pdt.assert_series_equal(predicted, expected)


def test_predict_ytransform(test_df):
    yt = lambda x: x / 2.
    filters = ['col1 in [0, 2, 4]']
    model_exp = 'col1 ~ col2'
    fit = regression.fit_model(test_df, filters, model_exp)
    predicted = regression.predict(
        test_df.query('col1 in [1, 3]'), None, fit, ytransform=yt)
    expected = pd.Series([0.5, 1.5], index=['b', 'd'])
    pdt.assert_series_equal(predicted, expected)


def test_predict_with_nans():
    df = pd.DataFrame(
        {'col1': range(5),
         'col2': [5, 6, pd.np.nan, 8, 9]},
        index=['a', 'b', 'c', 'd', 'e'])
    fit = regression.fit_model(df.loc[['a', 'b', 'e']], None, 'col1 ~ col2')

    with pytest.raises(ModelEvaluationError):
        regression.predict(
            df.loc[['c', 'd']], None, fit)


def test_rhs():
    assert regression._rhs('col1 + col2') == 'col1 + col2'
    assert regression._rhs('col3 ~ col1 + col2') == 'col1 + col2'


def test_FakeRegressionResults(test_df):
    model_exp = 'col1 ~ col2'
    model = smf.ols(formula=model_exp, data=test_df)
    fit = model.fit()

    wrapper = regression._FakeRegressionResults(
        model_exp, fit.params)

    test_predict = pd.DataFrame({'col2': [0.5, 10, 25.6]})

    npt.assert_array_equal(
        wrapper.predict(test_predict), fit.predict(test_predict))


def test_RegressionModel(test_df):
    fit_filters = ['col1 in [0, 2, 4]']
    predict_filters = ['col1 in [1, 3]']
    model_exp = 'col1 ~ col2'
    ytransform = lambda x: x / 2.
    name = 'test hedonic'

    model = regression.RegressionModel(
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

    fit = model.fit(test_df)
    assert isinstance(fit, RegressionResultsWrapper)
    assert isinstance(model.model_fit, RegressionResultsWrapper)

    predicted = model.predict(test_df)
    expected = pd.Series([0.5, 1.5], index=['b', 'd'])
    pdt.assert_series_equal(predicted, expected)


def test_RegressionModelGroup(groupby_df):
    model_exp = 'col1 ~ col2'

    hmg = regression.RegressionModelGroup('group')

    xmodel = regression.RegressionModel(None, None, model_exp, name='x')
    hmg.add_model(xmodel)
    assert isinstance(hmg.models['x'], regression.RegressionModel)

    hmg.add_model_from_params('y', None, None, model_exp)
    assert isinstance(hmg.models['y'], regression.RegressionModel)
    assert hmg.models['y'].name == 'y'

    fits = hmg.fit(groupby_df)
    assert isinstance(fits['x'], RegressionResultsWrapper)
    assert isinstance(fits['y'], RegressionResultsWrapper)

    predicted = hmg.predict(groupby_df)
    assert isinstance(predicted, pd.Series)
    pdt.assert_series_equal(
        predicted.sort_index(), groupby_df.col1, check_dtype=False)
