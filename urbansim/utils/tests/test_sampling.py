import numpy as np
import pandas as pd
import pytest

from urbansim.utils.sampling import sample_rows, get_probs


def test_get_probs():
    df = pd.DataFrame({
        'a': np.zeros(4),
        'b': [0.25, 0.25, 0.25, 0.25],
        'c': np.ones(4)
    })

    assert get_probs(df) is None
    expected = [0.25, 0.25, 0.25, 0.25]
    assert (get_probs(df, 'a') == expected).all()
    assert (get_probs(df, 'b') == expected).all()
    assert (get_probs(df, 'c') == expected).all()


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
    np.random.seed(123)
    return pd.DataFrame(
        {
            'some_count': np.random.randint(1, 8, 20),
            'p': np.arange(20)
        },
        index=range(0, 20)
    )


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

    rows, matched = sample_rows(
        control, random_df, accounting_column='some_count', return_status=True)
    assert control == rows['some_count'].sum()
    assert matched

    # test with probabilities
    rows, matched = sample_rows(
        control, random_df, accounting_column='some_count', prob_column='p', return_status=True)
    assert control == rows['some_count'].sum()
    assert matched


def test_accounting_no_replacment(random_df):
    control = 10
    rows, matched = sample_rows(
        control, random_df, accounting_column='some_count', replace=False, return_status=True)
    assert control == rows['some_count'].sum()
    assert matched

    # test with probabilities
    rows, matched = sample_rows(control, random_df, accounting_column='some_count',
                                replace=False, prob_column='p', return_status=True)
    assert control == rows['some_count'].sum()
    assert matched


def test_accounting_no_replacment_raises(random_df):
    control = 200
    with pytest.raises(ValueError):
        sample_rows(control, random_df, accounting_column='some_count', replace=False)
