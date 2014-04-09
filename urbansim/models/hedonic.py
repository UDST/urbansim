import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def apply_filter_query(df, filters):
    """
    Use the DataFrame.query method to filter a table down to the
    desired rows.

    Parameters
    ----------
    df : pandas.DataFrame
    filters : list of str
        List of filters to apply. Will be joined together with
        ' and ' and passed to DataFrame.query.

    Returns
    -------
    filtered_df : pandas.DataFrame

    """
    query = ' and '.join(filters)
    return df.query(query)


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
    df = apply_filter_query(df, filters)
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
    df = apply_filter_query(df, filters)
    sim_data = model_fit.predict(df)
    if ytransform:
        sim_data = ytransform(sim_data)
    return pd.Series(sim_data, index=df.index)


class HedonicModel(object):
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
    name : str, optional
        Optional descriptive name for this model that may be used
        in output.

    """
    def __init__(self, fit_filters, predict_filters, model_expression,
                 ytransform=None, name=None):
        self.fit_filters = fit_filters
        self.predict_filters = predict_filters
        self.model_expression = model_expression
        self.ytransform = ytransform
        self.name = name or 'HedonicModel'
        self.model_fit = None

    def fit_model(self, data):
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
        return predict(
            data, self.predict_filters, self.model_fit, self.ytransform)
