import numpy as np
import pandas as pd
import pytest

from urbansim.utils.sampling import sample_rows

@pytest.fixture
def simple_df():
    return pd.DataFrame(
        {'some_count': [1,2,3,4,5]},
        index=range(0, 5))


def test_no_accounting_with_replacment(simple_df):
    control = 3
    rows = sample_rows(control, simple_df)
    assert control == len(rows)


def test_no_accounting_no_replacment(simple_df):
    control = 3
    rows = sample_rows(control, simple_df, replace=False)
    assert control == len(rows)


def test_accounting_with_replacment():
    old_state = np.random.get_state()
    np.random.seed(1)
    control = 10
    df = pd.DataFrame(
        {'some_count': np.random.randint(1,8,20)},
        index = range(0,20)
        )
    rows = sample_rows(control, df, accounting_column='some_count')
    np.random.set_state(old_state)
    assert control == rows['some_count'].sum()


def test_accounting_no_replacment():
    old_state = np.random.get_state()
    np.random.seed(1)
    control = 10
    df = pd.DataFrame(
        {'some_count': np.random.randint(1,8,10)},
        index = range(0,10)
        )
    rows = sample_rows(control, df, accounting_column='some_count', replace=False)
    np.random.set_state(old_state)
    assert control == rows['some_count'].sum()