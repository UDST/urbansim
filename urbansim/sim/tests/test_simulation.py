import pandas as pd
import pytest
from pandas.util import testing as pdt

from .. import simulation as sim


@pytest.fixture
def df():
    return pd.DataFrame(
        {'a': [1, 2, 3],
         'b': [4, 5, 6]})


def test_tables(df):
    sim.add_table('test_frame', df)

    @sim.table('test_func')
    def test_func(test_frame):
        return test_frame.to_frame() / 2

    assert set(sim.list_tables()) == {'test_frame', 'test_func'}

    table = sim.get_table('test_func')

    pdt.assert_frame_equal(table.to_frame(), df / 2)
    pdt.assert_frame_equal(table.to_frame(columns=['a']), df[['a']] / 2)
