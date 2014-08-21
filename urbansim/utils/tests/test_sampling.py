import numpy as np
import pandas as pd
import pytest

from urbansim.utils.sampling import sample_rows


@pytest.fixture(scope='function')
def random_df(request):
    """
    Seed the numpy prng and return a data frame w/ predictable test inputs
    so that the tests will have consistent results across builds.
    """
    old_state = np.random.get_state()

    def fin():
        # tear down: reset the prng after the test to the pre-test state
        np.random.set_state(old_state)

    request.addfinalizer(fin)
    np.random.seed(1)
    return pd.DataFrame(
        {'some_count': np.random.randint(1, 8, 20)},
        index=range(0, 20))


def test_no_accounting_with_replacment(random_df):
    control = 3
    rows = sample_rows(control, random_df)
    assert control == len(rows)


def test_no_accounting_no_replacment(random_df):
    control = 3
    rows = sample_rows(control, random_df, replace=False)
    print random_df
    assert control == len(rows)


def test_no_accounting_no_replacment_raises(random_df):
    control = 21
    with pytest.raises(ValueError):
        sample_rows(control, random_df, replace=False)


def test_accounting_with_replacment(random_df):
    control = 10
    rows = sample_rows(control, random_df, accounting_column='some_count')
    assert control == rows['some_count'].sum()


def test_accounting_no_replacment(random_df):
    control = 10
    rows = sample_rows(control, random_df, accounting_column='some_count', replace=False)
    assert control == rows['some_count'].sum()


def test_accounting_no_replacment_raises(random_df):
    control = 200
    with pytest.raises(ValueError):
        sample_rows(control, random_df, accounting_column='some_count', replace=False)
