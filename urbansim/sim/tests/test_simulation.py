import pandas as pd
import pytest
from pandas.util import testing as pdt

from .. import simulation as sim


@pytest.fixture
def clear_sim(request):
    sim.clear_sim()

    def fin():
        sim.clear_sim()
    request.addfinalizer(fin)


@pytest.fixture
def df():
    return pd.DataFrame(
        {'a': [1, 2, 3],
         'b': [4, 5, 6]},
        index=['x', 'y', 'z'])


def test_tables(df, clear_sim):
    sim.add_table('test_frame', df)

    @sim.table('test_func')
    def test_func(test_frame):
        return test_frame.to_frame() / 2

    assert set(sim.list_tables()) == {'test_frame', 'test_func'}

    table = sim.get_table('test_func')

    pdt.assert_frame_equal(table.to_frame(), df / 2)
    pdt.assert_frame_equal(table.to_frame(columns=['a']), df[['a']] / 2)


def test_columns_for_table(clear_sim):
    sim.add_column(
        'table1', 'col10', pd.Series([1, 2, 3], index=['a', 'b', 'c']))
    sim.add_column(
        'table2', 'col20', pd.Series([10, 11, 12], index=['x', 'y', 'z']))

    @sim.column('table1', 'col11')
    def col11():
        return pd.Series([4, 5, 6], index=['a', 'b', 'c'])

    @sim.column('table2', 'col21')
    def col21():
        return pd.Series([13, 14, 15], index=['x', 'y', 'z'])

    t1_col_names = sim._list_columns_for_table('table1')
    assert set(t1_col_names) == {'col10', 'col11'}

    t2_col_names = sim._list_columns_for_table('table2')
    assert set(t2_col_names) == {'col20', 'col21'}

    t1_cols = sim._columns_for_table('table1')
    assert 'col10' in t1_cols and 'col11' in t1_cols

    t2_cols = sim._columns_for_table('table2')
    assert 'col20' in t2_cols and 'col21' in t2_cols


def test_columns_and_tables(df, clear_sim):
    sim.add_table('test_frame', df)

    @sim.table('test_func')
    def test_func(test_frame):
        return test_frame.to_frame() / 2

    sim.add_column('test_frame', 'c', pd.Series([7, 8, 9], index=df.index))

    @sim.column('test_func', 'd')
    def col_d(test_func):
        return test_func.to_frame(columns=['b'])['b'] * 2

    test_frame = sim.get_table('test_frame')
    pdt.assert_frame_equal(
        test_frame.to_frame(),
        pd.DataFrame(
            {'a': [1, 2, 3],
             'b': [4, 5, 6],
             'c': [7, 8, 9]},
            index=['x', 'y', 'z']))
    pdt.assert_frame_equal(
        test_frame.to_frame(columns=['a', 'c']),
        pd.DataFrame(
            {'a': [1, 2, 3],
             'c': [7, 8, 9]},
            index=['x', 'y', 'z']))

    test_func_df = sim.get_table('test_func')
    pdt.assert_frame_equal(
        test_func_df.to_frame(),
        pd.DataFrame(
            {'a': [0.5, 1, 1.5],
             'b': [2, 2.5, 3],
             'c': [3.5, 4, 4.5],
             'd': [4., 5., 6.]},
            index=['x', 'y', 'z']))
    pdt.assert_frame_equal(
        test_func_df.to_frame(columns=['b', 'd']),
        pd.DataFrame(
            {'b': [2, 2.5, 3],
             'd': [4., 5., 6.]},
            index=['x', 'y', 'z']))
