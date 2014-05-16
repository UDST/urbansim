import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from ...utils import testing

from .. import relocation as relo


@pytest.fixture
def choosers():
    return pd.DataFrame(
        {'var1': range(5),
         'var2': range(5, 10),
         'var3': ['q', 'w', 'e', 'r', 't'],
         'building_id': range(100, 105)},
        index=['a', 'b', 'c', 'd', 'e'])


@pytest.fixture
def rates():
    return pd.DataFrame(
        {'var1_min': [np.nan, np.nan, np.nan],
         'var1_max': [1, np.nan, np.nan],
         'var2_min': [np.nan, 7, np.nan],
         'var2_max': [np.nan, 8, np.nan],
         'var3': [np.nan, np.nan, 't'],
         'probability_of_relocating': [1, 1, 1]})


def test_find_movers(choosers, rates):
    movers = relo.find_movers(choosers, rates, 'probability_of_relocating')
    npt.assert_array_equal(movers, ['a', 'c', 'e'])


def test_relocation_model_find(choosers, rates):
    rm = relo.RelocationModel(rates)
    movers = rm.find_movers(choosers)
    npt.assert_array_equal(movers, ['a', 'c', 'e'])


def test_relocation_model_yaml(rates):
    rm = relo.RelocationModel(rates)
    new_rm = relo.RelocationModel.from_yaml(rm.to_yaml())

    testing.assert_frames_equal(new_rm.relocation_rates, rm.relocation_rates)
    assert new_rm.rate_column == rm.rate_column
