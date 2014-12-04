import numpy.testing as npt
import pandas as pd
import pandas.util.testing as pdt
import pytest

from .. import interaction as inter


@pytest.fixture
def choosers():
    return pd.DataFrame(
        {'var1': range(5, 10),
         'thing_id': ['a', 'c', 'e', 'g', 'i']})


@pytest.fixture
def alternatives():
    return pd.DataFrame(
        {'var2': range(10, 20),
         'var3': range(20, 30)},
        index=pd.Index([x for x in 'abcdefghij'], name='thing_id'))


def test_interaction_dataset_sim(choosers, alternatives):
    sample, merged, chosen = inter.mnl_interaction_dataset(
        choosers, alternatives, len(alternatives))

    # chosen should be len(choosers) rows * len(alternatives) cols
    assert chosen.shape == (len(choosers), len(alternatives))
    assert chosen[:, 0].sum() == len(choosers)
    assert chosen[:, 1:].sum() == 0

    npt.assert_array_equal(
        sample, list(alternatives.index.values) * len(choosers))

    assert len(merged) == len(choosers) * len(alternatives)
    npt.assert_array_equal(merged.index.values, sample)
    assert list(merged.columns) == [
        'var2', 'var3', 'join_index', 'thing_id', 'var1']
