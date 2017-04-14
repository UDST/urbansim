import os
import tempfile
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

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
    def yt(x):
        return x / 2.
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

    with pytest.raises(ModelEvaluationError):
        regression.fit_model(df, None, 'col1 ~ col2')

    fit = regression.fit_model(df.loc[['a', 'b', 'e']], None, 'col1 ~ col2')

    predict = regression.predict(
        df.loc[['c', 'd']], None, fit)

    assert np.isnan(predict.loc['c'])


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
    pdt.assert_series_equal(wrapper.params, fit.params, check_names=False)
    pdt.assert_series_equal(wrapper.bse, fit.bse, check_names=False)
    pdt.assert_series_equal(wrapper.tvalues, fit.tvalues, check_names=False)
    assert wrapper.rsquared == fit.rsquared
    assert wrapper.rsquared_adj == fit.rsquared_adj


def test_RegressionModel(test_df):
    fit_filters = ['col1 in [0, 2, 4]']
    predict_filters = ['col1 in [1, 3]']
    model_exp = 'col1 ~ col2'

    def ytransform(x):
        return x / 2.

    name = 'test hedonic'

    model = regression.RegressionModel(
        fit_filters, predict_filters, model_exp, ytransform, name)
    assert model.fit_filters == fit_filters
    assert model.predict_filters == predict_filters
    assert model.model_expression == model_exp
    assert model.ytransform == ytransform
    assert model.name == name
    assert model.model_fit is None
    assert set(model.columns_used()) == {'col1', 'col2'}

    # verify there's an error if there isn't a model fit yet
    with pytest.raises(RuntimeError):
        model.predict(test_df)

    fit = model.fit(test_df)
    assert isinstance(fit, RegressionResultsWrapper)
    assert isinstance(model.model_fit, RegressionResultsWrapper)

    predicted = model.predict(test_df)
    expected = pd.Series([0.5, 1.5], index=['b', 'd'])
    pdt.assert_series_equal(predicted, expected)

    # make sure this doesn't cause an error
    model.report_fit()


def test_RegressionModelGroup(groupby_df):
    model_exp = 'col1 ~ col2'

    hmg = regression.RegressionModelGroup('group')

    xmodel = regression.RegressionModel(None, None, model_exp, name='x')
    hmg.add_model(xmodel)
    assert isinstance(hmg.models['x'], regression.RegressionModel)

    hmg.add_model_from_params('y', None, None, model_exp)
    assert isinstance(hmg.models['y'], regression.RegressionModel)
    assert hmg.models['y'].name == 'y'

    assert set(hmg.columns_used()) == {'col1', 'col2'}

    assert hmg.fitted is False
    fits = hmg.fit(groupby_df)
    assert hmg.fitted is True
    assert isinstance(fits['x'], RegressionResultsWrapper)
    assert isinstance(fits['y'], RegressionResultsWrapper)

    predicted = hmg.predict(groupby_df)
    assert isinstance(predicted, pd.Series)
    pdt.assert_series_equal(
        predicted.sort_index(), groupby_df.col1,
        check_dtype=False, check_names=False)


def assert_dict_specs_equal(j1, j2):
    j1_params = j1.pop('fit_parameters')
    j2_params = j2.pop('fit_parameters')

    assert j1 == j2

    if j1_params and j2_params:
        pdt.assert_series_equal(
            pd.Series(j1_params['Coefficient']),
            pd.Series(j2_params['Coefficient']))
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
                'col2': 1.0},
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

    pdt.assert_series_equal(
        params['Coefficient'], fit.params, check_names=False)
    pdt.assert_series_equal(params['Std. Error'], fit.bse, check_names=False)
    pdt.assert_series_equal(params['T-Score'], fit.tvalues, check_names=False)

    assert params.rsquared == fit.rsquared
    assert params.rsquared_adj == fit.rsquared_adj


def test_SegmentedRegressionModel_raises(groupby_df):
    seg = regression.SegmentedRegressionModel('group')

    with pytest.raises(ValueError):
        seg.fit(groupby_df)


def test_SegmentedRegressionModel(groupby_df):
    seg = regression.SegmentedRegressionModel(
        'group', default_model_expr='col1 ~ col2')
    assert seg.fitted is False
    fits = seg.fit(groupby_df)
    assert seg.fitted is True

    assert 'x' in fits and 'y' in fits
    assert isinstance(fits['x'], RegressionResultsWrapper)

    test_data = pd.DataFrame({'group': ['x', 'y'], 'col2': [0.5, 10.5]})
    predicted = seg.predict(test_data)

    pdt.assert_series_equal(predicted.sort_index(), pd.Series([-4.5, 5.5]))


def test_SegmentedRegressionModel_explicit(groupby_df):
    seg = regression.SegmentedRegressionModel(
        'group', fit_filters=['col1 not in [2]'],
        predict_filters=['group != "z"'])
    seg.add_segment('x', 'col1 ~ col2')
    seg.add_segment('y', 'np.exp(col2) ~ np.exp(col1)', np.log)
    assert set(seg.columns_used()) == {'col1', 'col2', 'group'}

    fits = seg.fit(groupby_df)
    assert 'x' in fits and 'y' in fits
    assert isinstance(fits['x'], RegressionResultsWrapper)

    test_data = pd.DataFrame(
        {'group': ['x', 'z', 'y'],
         'col1': [-5, 42, 100],
         'col2': [0.5, 42, 10.5]})
    predicted = seg.predict(test_data)

    pdt.assert_series_equal(
        predicted.sort_index(), pd.Series([-4.5, 105], index=[0, 2]))


def test_SegmentedRegressionModel_yaml(groupby_df):
    seg = regression.SegmentedRegressionModel(
        'group', fit_filters=['col1 not in [2]'],
        predict_filters=['group != "z"'], default_model_expr='col1 ~ col2',
        min_segment_size=5000, name='test_seg')
    seg.add_segment('x')
    seg.add_segment('y', 'np.exp(col2) ~ np.exp(col1)', np.log)

    expected_dict = {
        'model_type': 'segmented_regression',
        'name': 'test_seg',
        'segmentation_col': 'group',
        'fit_filters': ['col1 not in [2]'],
        'predict_filters': ['group != "z"'],
        'min_segment_size': 5000,
        'default_config': {
            'model_expression': 'col1 ~ col2',
            'ytransform': None
        },
        'fitted': False,
        'models': {
            'x': {
                'name': 'x',
                'fitted': False,
                'fit_parameters': None,
                'fit_rsquared': None,
                'fit_rsquared_adj': None
            },
            'y': {
                'name': 'y',
                'model_expression': 'np.exp(col2) ~ np.exp(col1)',
                'ytransform': 'np.log',
                'fitted': False,
                'fit_parameters': None,
                'fit_rsquared': None,
                'fit_rsquared_adj': None
            }
        }
    }

    assert yaml.load(seg.to_yaml()) == expected_dict

    new_seg = regression.SegmentedRegressionModel.from_yaml(seg.to_yaml())
    assert yaml.load(new_seg.to_yaml()) == expected_dict

    seg.fit(groupby_df)

    expected_dict['fitted'] = True
    expected_dict['models']['x']['fitted'] = True
    expected_dict['models']['y']['fitted'] = True
    del expected_dict['models']['x']['fit_parameters']
    del expected_dict['models']['x']['fit_rsquared']
    del expected_dict['models']['x']['fit_rsquared_adj']
    del expected_dict['models']['y']['fit_parameters']
    del expected_dict['models']['y']['fit_rsquared']
    del expected_dict['models']['y']['fit_rsquared_adj']

    actual_dict = yaml.load(seg.to_yaml())
    assert isinstance(actual_dict['models']['x'].pop('fit_parameters'), dict)
    assert isinstance(actual_dict['models']['x'].pop('fit_rsquared'), float)
    assert isinstance(
        actual_dict['models']['x'].pop('fit_rsquared_adj'), float)
    assert isinstance(actual_dict['models']['y'].pop('fit_parameters'), dict)
    assert isinstance(actual_dict['models']['y'].pop('fit_rsquared'), float)
    assert isinstance(
        actual_dict['models']['y'].pop('fit_rsquared_adj'), float)

    assert actual_dict == expected_dict

    new_seg = regression.SegmentedRegressionModel.from_yaml(seg.to_yaml())
    assert new_seg.fitted is True


def test_SegmentedRegressionModel_removes_gone_segments(groupby_df):
    seg = regression.SegmentedRegressionModel(
        'group', default_model_expr='col1 ~ col2')
    seg.add_segment('a')
    seg.add_segment('b')
    seg.add_segment('c')

    seg.fit(groupby_df)

    assert sorted(seg._group.models.keys()) == ['x', 'y']


def test_fit_from_cfg(test_df):
    fit_filters = ['col1 in [0, 2, 4]']
    predict_filters = ['col1 in [1, 3]']
    model_exp = 'col1 ~ col2'
    ytransform = np.log
    name = 'test hedonic'

    model = regression.RegressionModel(
        fit_filters, predict_filters, model_exp, ytransform, name)

    cfgname = tempfile.NamedTemporaryFile(suffix='.yaml').name
    model.to_yaml(cfgname)
    regression.RegressionModel.fit_from_cfg(test_df, cfgname, debug=True)
    regression.RegressionModel.predict_from_cfg(test_df, cfgname)
    os.remove(cfgname)


def test_fit_from_cfg_segmented(groupby_df):
    seg = regression.SegmentedRegressionModel(
        'group', fit_filters=['col1 not in [2]'],
        predict_filters=['group != "z"'], default_model_expr='col1 ~ col2',
        min_segment_size=5000, name='test_seg')
    seg.add_segment('x')

    cfgname = tempfile.NamedTemporaryFile(suffix='.yaml').name
    seg.to_yaml(cfgname)
    regression.SegmentedRegressionModel.fit_from_cfg(groupby_df, cfgname, debug=True,
                                                     min_segment_size=5000)
    regression.SegmentedRegressionModel.predict_from_cfg(groupby_df, cfgname, min_segment_size=5000)
    os.remove(cfgname)
