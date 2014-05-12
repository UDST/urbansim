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
    chooser_ids : 1d array_like
        Array of IDs of the agents that are making choices.
    alternative_ids : 1d array_like
        Array of IDs of alternatives among which agents are making choices.
    probabilities : 1d array_like
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


class MNLLocationChoiceModel(object):
    """
    A location choice model with the ability to store an estimated
    model and predict new data based on the model.
    Based on multinomial logit.

    Parameters
    ----------
    model_expression : str
        A patsy model expression. Should contain only a right-hand side.
    sample_size : int
        Number of choices to sample for estimating the model.
    location_id_col : str, optional
        Name of a column in the choosers table that corresponds to the
        index of the location being chosen. If given, this is used to
        make sure that during prediction only choosers that have nan
        in this column choose new alternatives.
    choosers_fit_filters : list of str, optional
        Filters applied to choosers table before fitting the model.
    choosers_predict_filters : list of str, optional
        Filters applied to the choosers table before calculating
        new data points.
    alts_fit_filters : list of str, optional
        Filters applied to the alternatives table before fitting the model.
    alts_predict_filters : list of str, optional
        Filters applied to the alternatives table before calculating
        new data points.
    interaction_predict_filters : list of str, optional
        Filters applied to the merged choosers/alternatives table
        before predicting agent choices.
    choice_column : optional
        Name of the column in the `alternatives` table that choosers
        should choose. e.g. the 'building_id' column. If not provided
        the alternatives index is used.
    name : optional
        Optional descriptive name for this model that may be used
        in output.

    """
    def __init__(self, model_expression, sample_size, location_id_col=None,
                 choosers_fit_filters=None, choosers_predict_filters=None,
                 alts_fit_filters=None, alts_predict_filters=None,
                 interaction_predict_filters=None,
                 choice_column=None, name=None):
        # LCMs never have a constant
        self.model_expression = model_expression + ' - 1'
        self.sample_size = sample_size
        self.location_id_col = location_id_col
        self.choosers_fit_filters = choosers_fit_filters
        self.choosers_predict_filters = choosers_predict_filters
        self.alts_fit_filters = alts_fit_filters
        self.alts_predict_filters = alts_predict_filters
        self.interaction_predict_filters = interaction_predict_filters
        self.choice_column = choice_column
        self.name = name or 'MNLLocationChoiceModel'

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
        choosers = util.apply_filter_query(choosers, self.choosers_fit_filters)
        current_choice = current_choice.loc[choosers.index]
        alternatives = util.apply_filter_query(
            alternatives, self.alts_fit_filters)
        _, merged, chosen = interaction.mnl_interaction_dataset(
            choosers, alternatives, self.sample_size, current_choice)
        model_design = dmatrix(
            self.model_expression, data=merged, return_type='dataframe')
        self._model_columns = model_design.columns  # used for report
        fit, results = mnl.mnl_estimate(
            model_design.as_matrix(), chosen, self.sample_size)
        self._log_lks = fit
        self.fit_results = results
        return fit

    @property
    def fitted(self):
        """
        True if model is ready for prediction.

        """
        return bool(self.fit_results)

    def assert_fitted(self):
        """
        Raises `RuntimeError` if the model is not ready for prediction.

        """
        if not self.fitted:
            raise RuntimeError('Model has not been fit.')

    @property
    def coefficients(self):
        """
        Model coefficients as a list.

        """
        self.assert_fitted()
        return [x[0] for x in self.fit_results]

    def report_fit(self):
        """
        Print a report of the fit results.

        """
        if not self.fitted:
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

    def predict(self, choosers, alternatives):
        """
        Choose from among alternatives for a group of agents.

        Parameters
        ----------
        choosers : pandas.DataFrame
            Table describing the agents making choices, e.g. households.
            Only the first item in this table is used for determining
            agent probabilities of choosing alternatives.
        alternatives : pandas.DataFrame
            Table describing the things from which agents are choosing.

        Returns
        -------
        choices : pandas.Series
            Mapping of chooser ID to alternative ID. Some choosers
            will map to a nan value when there are not enough alternatives
            for all the choosers.

        """
        self.assert_fitted()

        if self.location_id_col:
            choosers = choosers[choosers[self.location_id_col].isnull()]
        choosers = util.apply_filter_query(
            choosers, self.choosers_predict_filters)
        alternatives = util.apply_filter_query(
            alternatives, self.alts_predict_filters)

        # TODO: only using 1st item in choosers for determining probabilities.
        # Need to expand options around this.
        _, merged, _ = interaction.mnl_interaction_dataset(
            choosers.head(1), alternatives, len(alternatives))
        merged = util.apply_filter_query(
            merged, self.interaction_predict_filters)
        model_design = dmatrix(
            self.model_expression, data=merged, return_type='dataframe')

        # probabilities are returned from mnl_simulate as a 2d array
        # and need to be flatted for use in unit_choice.
        probabilities = mnl.mnl_simulate(
            model_design.as_matrix(), self.coefficients,
            numalts=len(merged), returnprobs=True).flatten()

        # figure out exactly which things from which choices are drawn
        alt_choices = (
            merged[self.choice_column] if self.choice_column else merged.index)

        return unit_choice(
            choosers.index, alt_choices, probabilities)
