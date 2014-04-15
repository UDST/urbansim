import numpy as np
from patsy import dmatrix

from ..urbanchoice import interaction, mnl


class LocationChoiceModel(object):
    """
    A location choice model with the ability to store an estimated
    model and predict new data based on the model.

    Parameters
    ----------
    fit_filters : list of str
        Filters applied before fitting the model.
    predict_filters : list of str
        Filters applied before calculating new data points.
    model_expression : str
        A patsy model expression. Should contain only a right-hand side.
    sample_size : int
        Number of choices to sample for estimating the model.
    name : optional
        Optional descriptive name for this model that may be used
        in output.

    """
    def __init__(self, fit_filters, predict_filters, model_expression,
                 sample_size, name=None):
        self.fit_filters = fit_filters
        self.predict_filters = predict_filters
        # LCMs never have a constant
        self.model_expression = model_expression + ' - 1'
        self.sample_size = sample_size
        self.name = name or 'LocationChoiceModel'

    def fit(self, choosers, alternatives, current_choice):
        """
        Fit and save model parameters based on given data.

        Parameters
        ----------
        choosers : pandas.DataFrame
            Table describing the agents making choices, e.g. households.
        alternatives : pandas.DataFrame
            Table describing the things from which agents are choosing,
            e.g. buildings.
        current_choice : pandas.Series
            A Series describing the `alternatives` currently chosen
            by the `choosers`. Should have an index matching `choosers`
            and values matching the index of `alternatives`.

        Returns
        -------
        null_ll : float
            Null Log-liklihood
        conv_ll : float
            Log-liklihood at convergence
        ll_ratio : float
            Log-liklihood ratio

        """
        _, merged, chosen = interaction.mnl_interaction_dataset(
            choosers, alternatives, self.sample_size, current_choice)
        model_design = dmatrix(self.model_expression, data=merged)
        fit, results = mnl.mnl_estimate(
            model_design.as_matrix(), chosen, self.sample_size)
        self.fit_results = results
        return fit

    def predict(self, data):
        pass
