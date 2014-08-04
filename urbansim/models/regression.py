"""
Use the ``RegressionModel`` class to fit a model using statsmodels'
OLS capability and then do subsequent prediction.

"""
from __future__ import print_function

import logging

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import toolz
from patsy import dmatrix
from prettytable import PrettyTable

from . import util
from ..exceptions import ModelEvaluationError
from ..utils import yamlio
from ..utils.logutil import log_start_finish

logger = logging.getLogger(__name__)


def fit_model(df, filters, model_expression):
    """
    Use statsmodels OLS to construct a model relation.

    Parameters
    ----------
    df : pandas.DataFrame
        Data to use for fit. Should contain all the columns
        referenced in the `model_expression`.
    filters : list of str
        Any filters to apply before doing the model fit.
    model_expression : str
        A patsy model expression that can be used with statsmodels.
        Should contain both the left- and right-hand sides.

    Returns
    -------
    fit : statsmodels.regression.linear_model.OLSResults

    """
    df = util.apply_filter_query(df, filters)
    model = smf.ols(formula=model_expression, data=df)

    if len(model.exog) != len(df):
        raise ModelEvaluationError(
            'Estimated data does not have the same length as input.  '
            'This suggests there are null values in one or more of '
            'the input columns.')

    with log_start_finish('statsmodels OLS fit', logger):
        return model.fit()


def predict(df, filters, model_fit, ytransform=None):
    """
    Apply model to new data to predict new dependent values.

    Parameters
    ----------
    df : pandas.DataFrame
    filters : list of str
        Any filters to apply before doing prediction.
    model_fit : statsmodels.regression.linear_model.OLSResults
        Result of model estimation.
    ytransform : callable, optional
        A function to call on the array of predicted output.
        For example, if the model relation is predicting the log
        of price, you might pass ``ytransform=np.exp`` so that
        the results reflect actual price.

        By default no transformation is applied.

    Returns
    -------
    result : pandas.Series
        Predicted values as a pandas Series. Will have the index of `df`
        after applying filters.

    """
    df = util.apply_filter_query(df, filters)

    with log_start_finish('statsmodels predict', logger):
        sim_data = model_fit.predict(df)

    if len(sim_data) != len(df):
        raise ModelEvaluationError(
            'Predicted data does not have the same length as input. '
            'This suggests there are null values in one or more of '
            'the input columns.')

    if ytransform:
        sim_data = ytransform(sim_data)
    return pd.Series(sim_data, index=df.index)


def _rhs(model_expression):
    """
    Get only the right-hand side of a patsy model expression.

    Parameters
    ----------
    model_expression : str

    Returns
    -------
    rhs : str

    """
    if '~' not in model_expression:
        return model_expression
    else:
        return model_expression.split('~')[1].strip()


class _FakeRegressionResults(object):
    """
    This can be used in place of a statsmodels RegressionResults
    for limited purposes when it comes to model prediction.

    Intended for use when loading a model from a YAML representation;
    we can do model evaluation using the stored coefficients, but can't
    recreate the original statsmodels fit result.

    Parameters
    ----------
    model_expression : str
        A patsy model expression that can be used with statsmodels.
        Should contain both the left- and right-hand sides.
    fit_parameters : pandas.DataFrame
        Stats results from fitting `model_expression` to data.
        Should include columns 'Coefficient', 'Std. Error', and 'T-Score'.
    rsquared : float
    rsquared_adj : float

    """
    def __init__(self, model_expression, fit_parameters, rsquared,
                 rsquared_adj):
        self.model_expression = model_expression
        self.params = fit_parameters['Coefficient']
        self.bse = fit_parameters['Std. Error']
        self.tvalues = fit_parameters['T-Score']
        self.rsquared = rsquared
        self.rsquared_adj = rsquared_adj

    @property
    def _rhs(self):
        """
        Get only the right-hand side of `model_expression`.

        """
        return _rhs(self.model_expression)

    def predict(self, data):
        """
        Predict new values by running data through the fit model.

        Parameters
        ----------
        data : pandas.DataFrame
            Table with columns corresponding to the RHS of `model_expression`.

        Returns
        -------
        predicted : ndarray
            Array of predicted values.

        """
        with log_start_finish('_FakeRegressionResults prediction', logger):
            model_design = dmatrix(
                self._rhs, data=data, return_type='dataframe')
            return model_design.dot(self.params).values


def _model_fit_to_table(fit):
    """
    Produce a pandas DataFrame of model fit results from a statsmodels
    fit result object.

    Parameters
    ----------
    fit : statsmodels.regression.linear_model.RegressionResults

    Returns
    -------
    fit_parameters : pandas.DataFrame
        Will have columns 'Coefficient', 'Std. Error', and 'T-Score'.
        Index will be model terms.

        This frame will also have non-standard attributes
        .rsquared and .rsquared_adj with the same meaning and value
        as on `fit`.

    """
    fit_parameters = pd.DataFrame(
        {'Coefficient': fit.params,
         'Std. Error': fit.bse,
         'T-Score': fit.tvalues})
    fit_parameters.rsquared = fit.rsquared
    fit_parameters.rsquared_adj = fit.rsquared_adj
    return fit_parameters


YTRANSFORM_MAPPING = {
    None: None,
    np.exp: 'np.exp',
    'np.exp': np.exp,
    np.log: 'np.log',
    'np.log': np.log,
    np.log1p: 'np.log1p',
    'np.log1p': np.log1p,
    np.expm1: 'np.expm1',
    'np.expm1': np.expm1
}


class RegressionModel(object):
    """
    A hedonic (regression) model with the ability to store an
    estimated model and predict new data based on the model.

    statsmodels' OLS implementation is used.

    Parameters
    ----------
    fit_filters : list of str
        Filters applied before fitting the model.
    predict_filters : list of str
        Filters applied before calculating new data points.
    model_expression : str or dict
        A patsy model expression that can be used with statsmodels.
        Should contain both the left- and right-hand sides.
    ytransform : callable, optional
        A function to call on the array of predicted output.
        For example, if the model relation is predicting the log
        of price, you might pass ``ytransform=np.exp`` so that
        the results reflect actual price.

        By default no transformation is applied.
    name : optional
        Optional descriptive name for this model that may be used
        in output.

    """
    def __init__(self, fit_filters, predict_filters, model_expression,
                 ytransform=None, name=None):
        self.fit_filters = fit_filters
        self.predict_filters = predict_filters
        self.model_expression = model_expression
        self.ytransform = ytransform
        self.name = name or 'RegressionModel'
        self.model_fit = None
        self.fit_parameters = None
        self.est_data = None

    @classmethod
    def from_yaml(cls, yaml_str=None, str_or_buffer=None):
        """
        Create a RegressionModel instance from a saved YAML configuration.
        Arguments are mutually exclusive.

        Parameters
        ----------
        yaml_str : str, optional
            A YAML string from which to load model.
        str_or_buffer : str or file like, optional
            File name or buffer from which to load YAML.

        Returns
        -------
        RegressionModel

        """
        cfg = yamlio.yaml_to_dict(yaml_str, str_or_buffer)

        model = cls(
            cfg['fit_filters'],
            cfg['predict_filters'],
            cfg['model_expression'],
            YTRANSFORM_MAPPING[cfg['ytransform']],
            cfg['name'])

        if 'fitted' in cfg and cfg['fitted']:
            fit_parameters = pd.DataFrame(cfg['fit_parameters'])
            fit_parameters.rsquared = cfg['fit_rsquared']
            fit_parameters.rsquared_adj = cfg['fit_rsquared_adj']

            model.model_fit = _FakeRegressionResults(
                model.str_model_expression,
                fit_parameters,
                cfg['fit_rsquared'], cfg['fit_rsquared_adj'])
            model.fit_parameters = fit_parameters

        logger.debug('loaded regression model {} from YAML'.format(model.name))
        return model

    @property
    def str_model_expression(self):
        """
        Model expression as a string suitable for use with patsy/statsmodels.

        """
        return util.str_model_expression(
            self.model_expression, add_constant=True)

    def fit(self, data, debug=False):
        """
        Fit the model to data and store/return the results.

        Parameters
        ----------
        data : pandas.DataFrame
            Data to use for fitting the model. Must contain all the
            columns referenced by the `model_expression`.
        debug : bool
            If debug is set to true, this sets the attribute "est_data"
            to a dataframe with the actual data used for estimation of
            this model.

        Returns
        -------
        fit : statsmodels.regression.linear_model.OLSResults
            This is returned for inspection, but also stored on the
            class instance for use during prediction.

        """
        with log_start_finish('fitting model {}'.format(self.name), logger):
            fit = fit_model(data, self.fit_filters, self.str_model_expression)

        self.model_fit = fit
        self.fit_parameters = _model_fit_to_table(fit)
        if debug:
            index = util.apply_filter_query(data, self.fit_filters).index
            assert len(fit.model.exog) == len(index), (
                "The estimate data is unequal in length to the original "
                "dataframe, usually caused by nans")
            df = pd.DataFrame(
                fit.model.exog, columns=fit.model.exog_names, index=index)
            df[fit.model.endog_names] = fit.model.endog
            df["fittedvalues"] = fit.fittedvalues
            df["residuals"] = fit.resid
            self.est_data = df
        return fit

    @property
    def fitted(self):
        """
        True if the model is ready for prediction.

        """
        return self.model_fit is not None

    def assert_fitted(self):
        """
        Raises a RuntimeError if the model is not ready for prediction.

        """
        if not self.fitted:
            raise RuntimeError('Model has not been fit.')

    def report_fit(self):
        """
        Print a report of the fit results.

        """
        if not self.fitted:
            print('Model not yet fit.')
            return

        print('R-Squared: {0:.3f}'.format(self.model_fit.rsquared))
        print('Adj. R-Squared: {0:.3f}'.format(self.model_fit.rsquared_adj))
        print('')

        tbl = PrettyTable(
            ['Component', ])
        tbl = PrettyTable()

        tbl.add_column('Component', self.fit_parameters.index.values)
        for col in ('Coefficient', 'Std. Error', 'T-Score'):
            tbl.add_column(col, self.fit_parameters[col].values)

        tbl.align['Component'] = 'l'
        tbl.float_format = '.3'

        print(tbl)

    def predict(self, data):
        """
        Predict a new data set based on an estimated model.

        Parameters
        ----------
        data : pandas.DataFrame
            Data to use for prediction. Must contain all the columns
            referenced by the right-hand side of the `model_expression`.

        Returns
        -------
        result : pandas.Series
            Predicted values as a pandas Series. Will have the index of `data`
            after applying filters.

        """
        self.assert_fitted()
        with log_start_finish('predicting model {}'.format(self.name), logger):
            return predict(
                data, self.predict_filters, self.model_fit, self.ytransform)

    def to_dict(self):
        """
        Returns a dictionary representation of a RegressionModel instance.

        """
        d = {
            'model_type': 'regression',
            'name': self.name,
            'fit_filters': self.fit_filters,
            'predict_filters': self.predict_filters,
            'model_expression': self.model_expression,
            'ytransform': YTRANSFORM_MAPPING[self.ytransform],
            'fitted': self.fitted,
            'fit_parameters': None,
            'fit_rsquared': None,
            'fit_rsquared_adj': None
        }

        if self.fitted:
            d['fit_parameters'] = yamlio.frame_to_yaml_safe(
                self.fit_parameters)
            d['fit_rsquared'] = float(self.model_fit.rsquared)
            d['fit_rsquared_adj'] = float(self.model_fit.rsquared_adj)

        return d

    def to_yaml(self, str_or_buffer=None):
        """
        Save a model respresentation to YAML.

        Parameters
        ----------
        str_or_buffer : str or file like, optional
            By default a YAML string is returned. If a string is
            given here the YAML will be written to that file.
            If an object with a ``.write`` method is given the
            YAML will be written to that object.

        Returns
        -------
        j : str
            YAML string if `str_or_buffer` is not given.

        """
        logger.debug(
            'serializing regression model {} to YAML'.format(self.name))
        return yamlio.convert_to_yaml(self.to_dict(), str_or_buffer)

    def columns_used(self):
        """
        Returns all the columns used in this model for filtering
        and in the model expression.

        """
        return list(toolz.unique(toolz.concatv(
            util.columns_in_filters(self.fit_filters),
            util.columns_in_filters(self.predict_filters),
            util.columns_in_formula(self.model_expression))))

    @classmethod
    def fit_from_cfg(cls, df, cfgname, debug=False):
        """
        Parameters
        ----------
        df : DataFrame
            The dataframe which contains the columns to use for the estimation.
        cfgname : string
            The name of the yaml config file which describes the hedonic model.
        debug : boolean, optional (default False)
            Whether to generate debug information on the model.

        Returns
        -------
        RegressionModel which was used to fit
        """
        logger.debug('start: fit from configuration {}'.format(cfgname))
        hm = cls.from_yaml(str_or_buffer=cfgname)
        ret = hm.fit(df, debug=debug)
        print(ret.summary())
        hm.to_yaml(str_or_buffer=cfgname)
        logger.debug('start: fit from configuration {}'.format(cfgname))
        return hm

    @classmethod
    def predict_from_cfg(cls, df, cfgname):
        """
        Parameters
        ----------
        df : DataFrame
            The dataframe which contains the columns to use for the estimation.
        cfgname : string
            The name of the yaml config file which describes the hedonic model.

        Returns
        -------
        predicted : pandas.Series
            Predicted data in a pandas Series. Will have the index of `data`
            after applying filters and minus any groups that do not have
            models.
        hm : RegressionModel which was used to predict
        """
        logger.debug('start: predict from configuration {}'.format(cfgname))
        hm = cls.from_yaml(str_or_buffer=cfgname)

        price_or_rent = hm.predict(df)
        print(price_or_rent.describe())

        logger.debug('start: predict from configuration {}'.format(cfgname))
        return price_or_rent, hm


class RegressionModelGroup(object):
    """
    Manages a group of regression models that refer to different segments
    within a single table.

    Model names must match the segment names after doing a Pandas groupby.

    Parameters
    ----------
    segmentation_col
        Name of the column on which to segment.
    name
        Optional name used to identify the model in places.

    """
    def __init__(self, segmentation_col, name=None):
        self.segmentation_col = segmentation_col
        self.name = name if name is not None else 'RegressionModelGroup'
        self.models = {}

    def add_model(self, model):
        """
        Add a `RegressionModel` instance.

        Parameters
        ----------
        model : `RegressionModel`
            Should have a ``.name`` attribute matching one of
            the groupby segments.

        """
        logger.debug(
            'adding model {} to group {}'.format(model.name, self.name))
        self.models[model.name] = model

    def add_model_from_params(self, name, fit_filters, predict_filters,
                              model_expression, ytransform=None):
        """
        Add a model by passing arguments through to `RegressionModel`.

        Parameters
        ----------
        name : any
            Must match a groupby segment name.
        fit_filters : list of str
            Filters applied before fitting the model.
        predict_filters : list of str
            Filters applied before calculating new data points.
        model_expression : str
            A patsy model expression that can be used with statsmodels.
            Should contain both the left- and right-hand sides.
        ytransform : callable, optional
            A function to call on the array of predicted output.
            For example, if the model relation is predicting the log
            of price, you might pass ``ytransform=np.exp`` so that
            the results reflect actual price.

            By default no transformation is applied.

        """
        logger.debug(
            'adding model {} to group {}'.format(name, self.name))
        model = RegressionModel(
            fit_filters, predict_filters, model_expression, ytransform, name)
        self.models[name] = model

    def _iter_groups(self, data):
        """
        Iterate over the groups in `data` after grouping by
        `segmentation_col`. Skips any groups for which there
        is no model stored.

        Yields tuples of (name, df) where name is the group key
        and df is the group DataFrame.

        Parameters
        ----------
        data : pandas.DataFrame
            Must have a column with the same name as `segmentation_col`.

        """
        groups = data.groupby(self.segmentation_col)

        for name in self.models:
            yield name, groups.get_group(name)

    def fit(self, data, debug=False):
        """
        Fit each of the models in the group.

        Parameters
        ----------
        data : pandas.DataFrame
            Must have a column with the same name as `segmentation_col`.
        debug : bool
            If set to true (default false) will pass the debug parameter
            to model estimation.

        Returns
        -------
        fits : dict of statsmodels.regression.linear_model.OLSResults
            Keys are the segment names.

        """
        with log_start_finish(
                'fitting models in group {}'.format(self.name), logger):
            return {name: self.models[name].fit(df, debug=debug)
                    for name, df in self._iter_groups(data)}

    @property
    def fitted(self):
        """
        Whether all models in the group have been fitted.

        """
        return (all(m.fitted for m in self.models.values())
                if self.models else False)

    def predict(self, data):
        """
        Predict new data for each group in the segmentation.

        Parameters
        ----------
        data : pandas.DataFrame
            Data to use for prediction. Must have a column with the
            same name as `segmentation_col`.

        Returns
        -------
        predicted : pandas.Series
            Predicted data in a pandas Series. Will have the index of `data`
            after applying filters and minus any groups that do not have
            models.

        """
        with log_start_finish(
                'predicting models in group {}'.format(self.name), logger):
            results = [self.models[name].predict(df)
                       for name, df in self._iter_groups(data)]
        return pd.concat(results)

    def columns_used(self):
        """
        Returns all the columns used across all models in the group
        for filtering and in the model expression.

        """
        return list(toolz.unique(toolz.concat(
            m.columns_used() for m in self.models.values())))


class SegmentedRegressionModel(object):
    """
    A regression model group that allows segments to have different
    model expressions and ytransforms but all have the same filters.

    Parameters
    ----------
    segmentation_col
        Name of column in the data table on which to segment. Will be used
        with a pandas groupby on the data table.
    fit_filters : list of str, optional
        Filters applied before fitting the model.
    predict_filters : list of str, optional
        Filters applied before calculating new data points.
    min_segment_size : int
        This model will add all segments that have at least this number of
        observations. A very small number of observations (e.g. 1) will
        cause an error with estimation.
    default_model_expr : str or dict, optional
        A patsy model expression that can be used with statsmodels.
        Should contain both the left- and right-hand sides.
    default_ytransform : callable, optional
        A function to call on the array of predicted output.
        For example, if the model relation is predicting the log
        of price, you might pass ``ytransform=np.exp`` so that
        the results reflect actual price.

        By default no transformation is applied.
    min_segment_size : int, optional
        Segments with less than this many members will be skipped.
    name : str, optional
        A name used in places to identify the model.

    """
    def __init__(
            self, segmentation_col, fit_filters=None, predict_filters=None,
            default_model_expr=None, default_ytransform=None,
            min_segment_size=0, name=None):
        self.segmentation_col = segmentation_col
        self._group = RegressionModelGroup(segmentation_col)
        self.fit_filters = fit_filters
        self.predict_filters = predict_filters
        self.default_model_expr = default_model_expr
        self.default_ytransform = default_ytransform
        self.min_segment_size = min_segment_size
        self.name = name if name is not None else 'SegmentedRegressionModel'

    @classmethod
    def from_yaml(cls, yaml_str=None, str_or_buffer=None):
        """
        Create a SegmentedRegressionModel instance from a saved YAML
        configuration. Arguments are mutally exclusive.

        Parameters
        ----------
        yaml_str : str, optional
            A YAML string from which to load model.
        str_or_buffer : str or file like, optional
            File name or buffer from which to load YAML.

        Returns
        -------
        SegmentedRegressionModel

        """
        cfg = yamlio.yaml_to_dict(yaml_str, str_or_buffer)

        default_model_expr = cfg['default_config']['model_expression']
        default_ytransform = cfg['default_config']['ytransform']

        seg = cls(
            cfg['segmentation_col'], cfg['fit_filters'],
            cfg['predict_filters'], default_model_expr,
            YTRANSFORM_MAPPING[default_ytransform], cfg['min_segment_size'],
            cfg['name'])

        if "models" not in cfg:
            cfg["models"] = {}

        for name, m in cfg['models'].items():
            m['model_expression'] = m.get(
                'model_expression', default_model_expr)
            m['ytransform'] = m.get('ytransform', default_ytransform)
            m['fit_filters'] = None
            m['predict_filters'] = None
            reg = RegressionModel.from_yaml(yamlio.convert_to_yaml(m, None))
            seg._group.add_model(reg)

        logger.debug(
            'loaded segmented regression model {} from yaml'.format(seg.name))
        return seg

    def add_segment(self, name, model_expression=None, ytransform='default'):
        """
        Add a new segment with its own model expression and ytransform.

        Parameters
        ----------
        name :
            Segment name. Must match a segment in the groupby of the data.
        model_expression : str or dict, optional
            A patsy model expression that can be used with statsmodels.
            Should contain both the left- and right-hand sides.
            If not given the default model will be used, which must not be
            None.
        ytransform : callable, optional
            A function to call on the array of predicted output.
            For example, if the model relation is predicting the log
            of price, you might pass ``ytransform=np.exp`` so that
            the results reflect actual price.

            If not given the default ytransform will be used.

        """
        if not model_expression:
            if self.default_model_expr is None:
                raise ValueError(
                    'No default model available, '
                    'you must supply a model experssion.')
            model_expression = self.default_model_expr

        if ytransform == 'default':
            ytransform = self.default_ytransform

        # no fit or predict filters, we'll take care of that this side.
        self._group.add_model_from_params(
            name, None, None, model_expression, ytransform)

        logger.debug('added segment {} to model {}'.format(name, self.name))

    def fit(self, data, debug=False):
        """
        Fit each segment. Segments that have not already been explicitly
        added will be automatically added with default model and ytransform.

        Parameters
        ----------
        data : pandas.DataFrame
            Must have a column with the same name as `segmentation_col`.
        debug : bool
            If set to true will pass debug to the fit method of each model.

        Returns
        -------
        fits : dict of statsmodels.regression.linear_model.OLSResults
            Keys are the segment names.

        """
        data = util.apply_filter_query(data, self.fit_filters)

        unique = data[self.segmentation_col].unique()
        value_counts = data[self.segmentation_col].value_counts()

        # Remove any existing segments that may no longer have counterparts
        # in the data. This can happen when loading a saved model and then
        # calling this method with data that no longer has segments that
        # were there the last time this was called.
        gone = set(self._group.models) - set(unique)
        for g in gone:
            del self._group.models[g]

        for x in unique:
            if x not in self._group.models and \
                    value_counts[x] > self.min_segment_size:
                self.add_segment(x)

        with log_start_finish(
                'fitting models in segmented model {}'.format(self.name),
                logger):
            return self._group.fit(data, debug=debug)

    @property
    def fitted(self):
        """
        Whether models for all segments have been fit.

        """
        return self._group.fitted

    def predict(self, data):
        """
        Predict new data for each group in the segmentation.

        Parameters
        ----------
        data : pandas.DataFrame
            Data to use for prediction. Must have a column with the
            same name as `segmentation_col`.

        Returns
        -------
        predicted : pandas.Series
            Predicted data in a pandas Series. Will have the index of `data`
            after applying filters.

        """
        with log_start_finish(
                'predicting models in segmented model {}'.format(self.name),
                logger):
            data = util.apply_filter_query(data, self.predict_filters)
            return self._group.predict(data)

    def _process_model_dict(self, d):
        """
        Remove redundant items from a model's configuration dict.

        Parameters
        ----------
        d : dict
            Modified in place.

        Returns
        -------
        dict
            Modified `d`.

        """
        del d['model_type']
        del d['fit_filters']
        del d['predict_filters']

        if d['model_expression'] == self.default_model_expr:
            del d['model_expression']

        if YTRANSFORM_MAPPING[d['ytransform']] == self.default_ytransform:
            del d['ytransform']

        d["name"] = yamlio.to_scalar_safe(d["name"])

        return d

    def to_dict(self):
        """
        Returns a dict representation of this instance suitable for
        conversion to YAML.

        """
        return {
            'model_type': 'segmented_regression',
            'name': self.name,
            'segmentation_col': self.segmentation_col,
            'fit_filters': self.fit_filters,
            'predict_filters': self.predict_filters,
            'min_segment_size': self.min_segment_size,
            'default_config': {
                'model_expression': self.default_model_expr,
                'ytransform': YTRANSFORM_MAPPING[self.default_ytransform]
            },
            'fitted': self.fitted,
            'models': {
                yamlio.to_scalar_safe(name):
                    self._process_model_dict(m.to_dict())
                for name, m in self._group.models.items()}
        }

    def to_yaml(self, str_or_buffer=None):
        """
        Save a model respresentation to YAML.

        Parameters
        ----------
        str_or_buffer : str or file like, optional
            By default a YAML string is returned. If a string is
            given here the YAML will be written to that file.
            If an object with a ``.write`` method is given the
            YAML will be written to that object.

        Returns
        -------
        j : str
            YAML string if `str_or_buffer` is not given.

        """
        logger.debug(
            'serializing segmented regression model {} to yaml'.format(
                self.name))
        return yamlio.convert_to_yaml(self.to_dict(), str_or_buffer)

    def columns_used(self):
        """
        Returns all the columns used across all models in the group
        for filtering and in the model expression.

        """
        return list(toolz.unique(toolz.concatv(
            util.columns_in_filters(self.fit_filters),
            util.columns_in_filters(self.predict_filters),
            util.columns_in_formula(self.default_model_expr),
            self._group.columns_used(),
            [self.segmentation_col])))

    @classmethod
    def fit_from_cfg(cls, df, cfgname, debug=False, min_segment_size=None):
        """
        Parameters
        ----------
        df : DataFrame
            The dataframe which contains the columns to use for the estimation.
        cfgname : string
            The name of the yaml config file which describes the hedonic model.
        debug : boolean, optional (default False)
            Whether to generate debug information on the model.
        min_segment_size : int, optional
            Set attribute on the model.

        Returns
        -------
        hm : SegmentedRegressionModel which was used to fit
        """
        logger.debug('start: fit from configuration {}'.format(cfgname))
        hm = cls.from_yaml(str_or_buffer=cfgname)
        if min_segment_size:
            hm.min_segment_size = min_segment_size

        for k, v in hm.fit(df, debug=debug).items():
            print("REGRESSION RESULTS FOR SEGMENT %s\n" % str(k))
            print(v.summary())
        hm.to_yaml(str_or_buffer=cfgname)
        logger.debug('finish: fit from configuration {}'.format(cfgname))
        return hm

    @classmethod
    def predict_from_cfg(cls, df, cfgname, min_segment_size=None):
        """
        Parameters
        ----------
        df : DataFrame
            The dataframe which contains the columns to use for the estimation.
        cfgname : string
            The name of the yaml config file which describes the hedonic model.
        min_segment_size : int, optional
            Set attribute on the model.

        Returns
        -------
        predicted : pandas.Series
            Predicted data in a pandas Series. Will have the index of `data`
            after applying filters and minus any groups that do not have
            models.
        hm : SegmentedRegressionModel which was used to predict
        """
        logger.debug('start: predict from configuration {}'.format(cfgname))
        hm = cls.from_yaml(str_or_buffer=cfgname)
        if min_segment_size:
            hm.min_segment_size = min_segment_size

        price_or_rent = hm.predict(df)
        print(price_or_rent.describe())
        logger.debug('finish: predict from configuration {}'.format(cfgname))

        return price_or_rent, hm
