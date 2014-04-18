from __future__ import print_function

import numpy as np
from patsy import dmatrix
from prettytable import PrettyTable

from . import util
from ..urbanchoice import interaction, mnl


class LocationChoiceModel(object):
    """
    A location choice model with the ability to store an estimated
    model and predict new data based on the model.

    Parameters
    ----------
    alts_fit_filters : list of str
        Filters applied to the alternatives table before fitting the model.
    alts_predict_filters : list of str
        Filters applied to the alternatives table before calculating
        new data points.
    model_expression : str
        A patsy model expression. Should contain only a right-hand side.
    sample_size : int
        Number of choices to sample for estimating the model.
    name : optional
        Optional descriptive name for this model that may be used
        in output.

    """
    def __init__(self, alts_fit_filters, alts_predict_filters,
                 model_expression, sample_size, name=None):
        self.alts_fit_filters = alts_fit_filters
        self.alts_predict_filters = alts_predict_filters
        # LCMs never have a constant
        self.model_expression = model_expression + ' - 1'
        self.sample_size = sample_size
        self.name = name or 'LocationChoiceModel'

        self._log_lks = None
        self._model_columns = None
        self.fit_results = None

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
        alternatives = util.apply_filter_query(
            alternatives, self.alts_fit_filters)
        _, merged, chosen = interaction.mnl_interaction_dataset(
            choosers, alternatives, self.sample_size, current_choice)
        model_design = dmatrix(
            self.model_expression, data=merged, return_type='dataframe')
        self._model_columns = model_design.columns
        fit, results = mnl.mnl_estimate(
            model_design.as_matrix(), chosen, self.sample_size)
        self._log_lks = fit
        self.fit_results = results
        return fit

    def report_fit(self):
        """
        Print a report of the fit results.

        """
        if not self.fit_results:
            print('Model not yet fit.')
            return

        print('Null Log-liklihood: {}'.format(self._log_lks[0]))
        print('Log-liklihood at convergence: {}'.format(self._log_lks[1]))
        print('Log-liklihood Ratio: {}\n'.format(self._log_lks[2]))

        tbl = PrettyTable(
            ['Component', 'Coefficient', 'Std. Error', 'T-Score'])
        tbl.align['Component'] = 'l'
        for c, x in zip(self._model_columns, self.fit_results):
            tbl.add_row((c,) + x)

        print(tbl)
