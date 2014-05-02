import numpy as np
import pandas as pd
import simplejson as json
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrix

from . import util
from ..exceptions import ModelEvaluationError


def fit_model(df, filters, model_expression):
    """
    Use statsmodels to construct a model relation.

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

    Intended for use when loading a model from a JSON representation;
    we can do model evaluation using the stored coefficients, but can't
    recreate the original statsmodels fit result.

    Parameters
    ----------
    model_expression : str
        A patsy model expression that can be used with statsmodels.
        Should contain both the left- and right-hand sides.
    coefficients : pandas.Series
        Coefficients (params) from fitting `model_expression` to data.

    """
    def __init__(self, model_expression, coefficients):
        self.model_expression = model_expression
        self.coefficients = np.asanyarray(coefficients)

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
        model_design = dmatrix(self._rhs, data=data, return_type='dataframe')
        return model_design.dot(self.coefficients).values


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

    Parameters
    ----------
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

    @classmethod
    def from_json(cls, json_str=None, str_or_buffer=None):
        """
        Create a RegressionModel instance from a saved JSON configuration.
        Arguments are mutally exclusive.

        Parameters
        ----------
        json_str : str, optional
            A JSON string from which to load model.
        str_or_buffer : str or file like, optional
            File name or buffer from which to load JSON.

        Returns
        -------
        RegressionModel

        """
        if json_str:
            j = json.loads(json_str)
        elif isinstance(str_or_buffer, str):
            with open(str_or_buffer) as f:
                j = json.load(f)
        else:
            j = json.load(str_or_buffer)

        model = cls(
            j['fit_filters'],
            j['predict_filters'],
            j['model_expression'],
            YTRANSFORM_MAPPING[j['ytransform']],
            j['name'])

        if j['fitted']:
            model.model_fit = _FakeRegressionResults(
                j['model_expression'], pd.Series(j['coefficients']))

        return model

    def fit(self, data):
        """
        Fit the model to data and store/return the results.

        Parameters
        ----------
        data : pandas.DataFrame
            Data to use for fitting the model. Must contain all the
            columns referenced by the `model_expression`.

        Returns
        -------
        fit : statsmodels.regression.linear_model.OLSResults
            This is returned for inspection, but also stored on the
            class instance for use during prediction.

        """
        fit = fit_model(data, self.fit_filters, self.model_expression)
        self.model_fit = fit
        return fit

    @property
    def fitted(self):
        """
        True if the model is ready for prediction.

        """
        return bool(self.model_fit)

    def assert_fitted(self):
        """
        Raises a RuntimeError if the model is not ready for prediction.

        """
        if not self.fitted:
            raise RuntimeError('Model has not been fit.')

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
        return predict(
            data, self.predict_filters, self.model_fit, self.ytransform)

    def to_json(self, str_or_buffer=None):
        """
        Save a model respresentation to JSON.

        Parameters
        ----------
        str_or_buffer : str or file like, optional
            By default a JSON string is returned. If a string is
            given here the JSON will be written to that file.
            If an object with a ``.write`` method is given the
            JSON will be written to that object.

        Returns
        -------
        j : str
            JSON is string if `str_or_buffer` is not given.

        """
        indent = 2

        j = {
            'model_type': 'regression',
            'name': self.name,
            'fit_filters': self.fit_filters,
            'predict_filters': self.predict_filters,
            'model_expression': self.model_expression,
            'ytransform': YTRANSFORM_MAPPING[self.ytransform],
            'coefficients': (None if not self.fitted
                             else self.model_fit.params.to_dict()),
            'fitted': self.fitted
        }

        if not str_or_buffer:
            return json.dumps(j, indent=indent)
        elif isinstance(str_or_buffer, str):
            with open(str_or_buffer, 'w') as f:
                json.dump(j, f, indent=indent)
        else:
            json.dump(j, str_or_buffer, indent=indent)


class RegressionModelGroup(object):
    """
    Manages a group of regression models that refer to different segments
    within a single table.

    Model names must match the segment names after doing a Pandas groupby.

    Parameters
    ----------
    segmentation_col : str
        Name of the column on which to segment.

    """
    def __init__(self, segmentation_col):
        self.segmentation_col = segmentation_col
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

    def fit(self, data):
        """
        Fit each of the models in the group.

        Parameters
        ----------
        data : pandas.DataFrame
            Must have a column with the same name as `segmentation_col`.

        Returns
        -------
        fits : dict of statsmodels.regression.linear_model.OLSResults
            Keys are the segment names.

        """
        return {name: self.models[name].fit(df)
                for name, df in self._iter_groups(data)}

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
        results = [self.models[name].predict(df)
                   for name, df in self._iter_groups(data)]
        return pd.concat(results)
