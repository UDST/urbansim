import pandas as pd
import pytest

from .. import LocationChoiceModel


@pytest.fixture
def choosers():
    return pd.DataFrame(
        {'var1': range(5, 10),
         'thing_id': range(5)})


@pytest.fixture
def alternatives():
    return pd.DataFrame(
        {'var2': range(10, 20),
         'var3': range(20, 30)},
        index=range(10))


def test_lcm(choosers, alternatives):
    lcm = LocationChoiceModel(
        ['var3 != 15'], ['var2 != 14'], 'var2 + var1:var3', 5,
        name='Test LCM')
    loglik = lcm.fit(choosers, alternatives, choosers.thing_id)

    # hard to test things exactly because there's some randomness
    # involved, but can at least do a smoke test.
    assert len(loglik) == 3
    assert len(lcm.fit_results) == 2
