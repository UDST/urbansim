"""
Test data and results for this are generated
by the R script at data/mnl_tests.R.

"""
from __future__ import division

import os.path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from patsy import dmatrix

from .. import mnl


@pytest.fixture
def num_alts():
    return 4


@pytest.fixture(scope='module', params=[
    ('fish.csv',
        'fish_choosers.csv',
        'price + catch - 1',
        'mode',
        pd.Series([-0.02047652, 0.95309824], index=['price', 'catch']),
        pd.DataFrame([
            [0.2849598, 0.2742482, 0.1605457, 0.2802463],
            [0.1498991, 0.4542377, 0.2600969, 0.1357664]],
            columns=['beach', 'boat', 'charter', 'pier'])),
    ('fish.csv',
        'fish_choosers.csv',
        'price:income + catch:income + catch * price - 1',
        'mode',
        pd.Series([
            9.839876e-01, -2.659466e-02, 6.933946e-07, -1.324231e-04,
            7.646750e-03],
            index=[
            'catch', 'price', 'price:income', 'catch:income', 'catch:price']),
        pd.DataFrame([
            [0.2885868, 0.2799776, 0.1466286, 0.2848070],
            [0.1346205, 0.4855238, 0.2593983, 0.1204575]],
            columns=['beach', 'boat', 'charter', 'pier'])),
    ('travel_mode.csv',
        'travel_choosers.csv',
        'wait + travel + vcost - 1',
        'choice',
        pd.Series([
            -0.033976668, -0.002192951, 0.008890669],
            index=['wait', 'travel', 'vcost']),
        pd.DataFrame([
            [0.2776876, 0.1584818, 0.1049530, 0.4588777],
            [0.1154490, 0.1653297, 0.1372684, 0.5819528]],
            columns=['air', 'train', 'bus', 'car'])),
    ('travel_mode.csv',
        'travel_choosers.csv',
        'wait + travel + income:vcost + income:gcost - 1',
        'choice',
        pd.Series([
            -3.307586e-02, -2.518762e-03, 1.601746e-04, 3.745822e-05],
            index=['wait', 'travel', 'income:vcost', 'income:gcost']),
        pd.DataFrame([
            [0.2862046, 0.1439074, 0.1044490, 0.4654390],
            [0.1098313, 0.1597317, 0.1344395, 0.5959975]],
            columns=['air', 'train', 'bus', 'car']))])
def test_data(request):
    data, choosers, form, col, est_expected, sim_expected = request.param
    return {
        'data': data,
        'choosers': choosers,
        'formula': form,
        'column': col,
        'est_expected': est_expected,
        'sim_expected': sim_expected
    }


@pytest.fixture
def df(test_data):
    filen = os.path.join(os.path.dirname(__file__), 'data', test_data['data'])
    return pd.read_csv(filen)


@pytest.fixture
def choosers(test_data):
    filen = os.path.join(
        os.path.dirname(__file__), 'data', test_data['choosers'])
    return pd.read_csv(filen)


@pytest.fixture
def chosen(df, num_alts, test_data):
    return df[test_data['column']].values.astype('int').reshape(
        (len(df) / num_alts, num_alts))


@pytest.fixture
def dm(df, test_data):
    return dmatrix(test_data['formula'], data=df, return_type='dataframe')


@pytest.fixture
def choosers_dm(choosers, test_data):
    return dmatrix(
        test_data['formula'], data=choosers, return_type='dataframe')


@pytest.fixture
def fit_coeffs(dm, chosen, num_alts):
    log_like, fit = mnl.mnl_estimate(dm.as_matrix(), chosen, num_alts)
    return fit.Coefficient.values


def test_mnl_estimate(dm, chosen, num_alts, test_data):
    log_like, fit = mnl.mnl_estimate(dm.as_matrix(), chosen, num_alts)
    result = pd.Series(fit.Coefficient.values, index=dm.columns)
    result, expected = result.align(test_data['est_expected'])
    npt.assert_allclose(result.values, expected.values, rtol=1e-4)


def test_mnl_simulate(dm, fit_coeffs, num_alts, test_data, choosers_dm):
    # check that if all the alternatives have the same numbers
    # we get an even probability distribution
    data = np.array(
        [[10 ** (x + 1) for x in range(len(dm.columns))]] * num_alts)

    probs = mnl.mnl_simulate(
        data, fit_coeffs, num_alts, returnprobs=True)

    npt.assert_allclose(probs, [[1 / num_alts] * num_alts])

    # now test with real data
    probs = mnl.mnl_simulate(
        choosers_dm.as_matrix(), fit_coeffs, num_alts, returnprobs=True)
    results = pd.DataFrame(probs, columns=test_data['sim_expected'].columns)
    results, expected = results.align(test_data['sim_expected'])
    npt.assert_allclose(results.as_matrix(), expected.as_matrix(), rtol=1e-4)
