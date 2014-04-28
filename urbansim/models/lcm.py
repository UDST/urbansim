from __future__ import print_function, division

import numpy as np
import pandas as pd
from patsy import dmatrix
from prettytable import PrettyTable

from . import util
from ..urbanchoice import interaction, mnl


def unit_choice(chooser_ids, alternative_ids, probabilities):
    """
    Have a set of choosers choose from among alternatives according
    to a probability distribution. Choice is binary: each
    alternative can only be chosen once.

    Parameters
    ----------
    chooser_ids : array_like
        Array of IDs of the agents that are making choices.
    alternative_ids : array_like
        Array of IDs of alternatives among which agents are making choices.
    probabilities : array_like
        The probability that an agent will choose an alternative.
        Must be the same shape as `alternative_ids`. Unavailable
        alternatives should have a probability of 0.

    Returns
    -------
    choices : pandas.Series
        Mapping of chooser ID to alternative ID. Some choosers
        will map to a nan value when there are not enough alternatives
        for all the choosers.

    """
    chooser_ids = np.asanyarray(chooser_ids)
    alternative_ids = np.asanyarray(alternative_ids)
    probabilities = np.asanyarray(probabilities)

    choices = pd.Series([np.nan] * len(chooser_ids), index=chooser_ids)

    if probabilities.sum() == 0:
        # return all nan if there are no available units
        return choices

    # probabilities need to sum to 1 for np.random.choice
    probabilities = probabilities / probabilities.sum()

    # need to see if there are as many available alternatives as choosers
    n_available = np.count_nonzero(probabilities)
    n_choosers = len(chooser_ids)
    n_to_choose = n_choosers if n_choosers < n_available else n_available

    chosen = np.random.choice(
        alternative_ids, size=n_to_choose, replace=False, p=probabilities)

    # if there are fewer available units than choosers we need to pick
    # which choosers get a unit
    if n_to_choose == n_available:
        chooser_ids = np.random.choice(
            chooser_ids, size=n_to_choose, replace=False)

    choices[chooser_ids] = chosen

    return choices


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
