import os
import tempfile
from StringIO import StringIO

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
import statsmodels.formula.api as smf
import yaml
from pandas.util import testing as pdt

from statsmodels.regression.linear_model import RegressionResultsWrapper

from .. import regression
from ...exceptions import ModelEvaluationError
from ...utils import testing


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

    fit_parameters = regression._model_fit_to_table(fit)

    wrapper = regression._FakeRegressionResults(
        model_exp, fit_parameters, fit.rsquared, fit.rsquared_adj)

    test_predict = pd.DataFrame({'col2': [0.5, 10, 25.6]})

    npt.assert_array_equal(
        wrapper.predict(test_predict), fit.predict(test_predict))
    pdt.assert_series_equal(wrapper.params, fit.params)
    pdt.assert_series_equal(wrapper.bse, fit.bse)
    pdt.assert_series_equal(wrapper.pvalues, fit.pvalues)
    assert wrapper.rsquared == fit.rsquared
    assert wrapper.rsquared_adj == fit.rsquared_adj


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


def assert_dict_specs_equal(j1, j2):
    j1_params = j1.pop('fit_parameters')
    j2_params = j2.pop('fit_parameters')

    assert j1 == j2

    if j1_params and j2_params:
        testing.assert_frames_equal(
            pd.DataFrame(j1_params), pd.DataFrame(j2_params))
    else:
        assert j1_params is None
        assert j2_params is None


class TestRegressionModelYAMLNotFit(object):
    def setup_method(self, method):
        fit_filters = ['col1 in [0, 2, 4]']
        predict_filters = ['col1 in [1, 3]']
        model_exp = 'col1 ~ col2'
        ytransform = np.log1p
        name = 'test hedonic'

        self.model = regression.RegressionModel(
            fit_filters, predict_filters, model_exp, ytransform, name)

        self.expected_dict = {
            'model_type': 'regression',
            'name': name,
            'fit_filters': fit_filters,
            'predict_filters': predict_filters,
            'model_expression': model_exp,
            'ytransform': regression.YTRANSFORM_MAPPING[ytransform],
            'fitted': False,
            'fit_parameters': None,
            'fit_rsquared': None,
            'fit_rsquared_adj': None
        }

    def test_string(self):
        test_yaml = self.model.to_yaml()
        assert_dict_specs_equal(yaml.load(test_yaml), self.expected_dict)

        model = regression.RegressionModel.from_yaml(yaml_str=test_yaml)
        assert isinstance(model, regression.RegressionModel)

    def test_buffer(self):
        test_buffer = StringIO()
        self.model.to_yaml(str_or_buffer=test_buffer)
        assert_dict_specs_equal(
            yaml.load(test_buffer.getvalue()), self.expected_dict)

        test_buffer.seek(0)
        model = regression.RegressionModel.from_yaml(str_or_buffer=test_buffer)
        assert isinstance(model, regression.RegressionModel)

        test_buffer.close()

    def test_file(self):
        test_file = tempfile.NamedTemporaryFile(suffix='.yaml').name
        self.model.to_yaml(str_or_buffer=test_file)

        with open(test_file) as f:
            assert_dict_specs_equal(yaml.load(f), self.expected_dict)

        model = regression.RegressionModel.from_yaml(str_or_buffer=test_file)
        assert isinstance(model, regression.RegressionModel)

        os.remove(test_file)


class TestRegressionModelYAMLFit(TestRegressionModelYAMLNotFit):
    def setup_method(self, method):
        super(TestRegressionModelYAMLFit, self).setup_method(method)

        self.model.fit(test_df())

        self.expected_dict['fitted'] = True
        self.expected_dict['fit_rsquared'] = 1.0
        self.expected_dict['fit_rsquared_adj'] = 1.0
        self.expected_dict['fit_parameters'] = {
            'Coefficient': {
                'Intercept': -5.0,
                'col2': 1.0000000000000002},
            'T-Score': {
                'Intercept': 8.621678386539817e-16,
                'col2': 5.997311421859925e-16},
            'Std. Error': {
                'Intercept': 6.771450370191848e-15,
                'col2': 9.420554752102651e-16}}

    def test_fitted_load(self, test_df):
        model = regression.RegressionModel.from_yaml(
            yaml_str=self.model.to_yaml())
        assert isinstance(model.model_fit, regression._FakeRegressionResults)
        npt.assert_array_equal(
            model.predict(test_df), self.model.predict(test_df))
        testing.assert_frames_equal(
            model.fit_parameters, self.model.fit_parameters)
        assert model.fit_parameters.rsquared == \
            self.model.fit_parameters.rsquared
        assert model.fit_parameters.rsquared_adj == \
            self.model.fit_parameters.rsquared_adj


def test_model_fit_to_table(test_df):
    filters = []
    model_exp = 'col1 ~ col2'
    fit = regression.fit_model(test_df, filters, model_exp)
    params = regression._model_fit_to_table(fit)

    pdt.assert_series_equal(params['Coefficient'], fit.params)
    pdt.assert_series_equal(params['Std. Error'], fit.bse)
    pdt.assert_series_equal(params['T-Score'], fit.pvalues)

    assert params.rsquared == fit.rsquared
    assert params.rsquared_adj == fit.rsquared_adj
