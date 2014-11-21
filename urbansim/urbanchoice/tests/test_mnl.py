import os.path

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
        'price + catch - 1',
        'mode',
        pd.Series([-0.02047652, 0.95309824], index=['price', 'catch'])),
    ('fish.csv',
        'price:income + catch:income + catch * price - 1',
        'mode',
        pd.Series([
            9.839876e-01, -2.659466e-02, 6.933946e-07, -1.324231e-04,
            7.646750e-03],
            index=[
            'catch', 'price', 'price:income', 'catch:income', 'catch:price'])),
    ('travel_mode.csv',
        'wait + travel + vcost - 1',
        'choice',
        pd.Series([
            -0.033976668, -0.002192951, 0.008890669],
            index=['wait', 'travel', 'vcost'])),
    ('travel_mode.csv',
        'wait + travel + income:vcost + income:gcost - 1',
        'choice',
        pd.Series([
            -3.307586e-02, -2.518762e-03, 1.601746e-04, 3.745822e-05],
            index=['wait', 'travel', 'income:vcost', 'income:gcost']))])
def test_data(request):
    file, form, col, expected = request.param
    return {
        'file': file,
        'formula': form,
        'column': col,
        'expected': expected
    }


@pytest.fixture
def df(test_data):
    filen = os.path.join(os.path.dirname(__file__), 'data', test_data['file'])
    return pd.read_csv(filen)


@pytest.fixture
def chosen(df, num_alts, test_data):
    return df[test_data['column']].values.astype('int').reshape(
        (len(df) / num_alts, num_alts))


@pytest.fixture
def dm(df, test_data):
    return dmatrix(test_data['formula'], data=df, return_type='dataframe')


@pytest.fixture
def fit_coeffs(dm, chosen, num_alts):
    log_like, fit = mnl.mnl_estimate(dm.as_matrix(), chosen, num_alts)
    return fit.Coefficient.values


def test_mnl_estimate(dm, chosen, num_alts, test_data):
    log_like, fit = mnl.mnl_estimate(dm.as_matrix(), chosen, num_alts)
    result = pd.Series(fit.Coefficient.values, index=dm.columns)
    result, expected = result.align(test_data['expected'])
    npt.assert_allclose(result.values, expected.values, rtol=1e-4)


def test_mnl_simulate(dm, fit_coeffs, num_alts):
    probs = mnl.mnl_simulate(
        dm.as_matrix(), fit_coeffs, num_alts, returnprobs=True)
    # pytest.set_trace()
