import pandas as pd
import pytest
from pandas.util import testing as pdt

from .. import simulation as sim
from ...utils.testing import assert_frames_equal


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

    table = sim.get_table('test_frame')
    assert table.columns == ['a', 'b']
    assert table.local_columns == ['a', 'b']
    assert len(table) == 3
    pdt.assert_index_equal(table.index, df.index)
    pdt.assert_series_equal(table.get_column('a'), df.a)
    pdt.assert_series_equal(table.a, df.a)
    pdt.assert_series_equal(table['b'], df['b'])

    table = sim.get_table('test_func')
    assert table.index is None
    assert table.columns == []
    assert len(table) is 0
    pdt.assert_frame_equal(table.to_frame(), df / 2)
    pdt.assert_frame_equal(table.to_frame(columns=['a']), df[['a']] / 2)
    pdt.assert_index_equal(table.index, df.index)
    pdt.assert_series_equal(table.get_column('a'), df.a / 2)
    pdt.assert_series_equal(table.a, df.a / 2)
    pdt.assert_series_equal(table['b'], df['b'] / 2)
    assert len(table) == 3
    assert table.columns == ['a', 'b']


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
    assert test_frame.columns == ['a', 'b', 'c']
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
    assert test_func_df.columns == ['d']
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
    assert test_func_df.columns == ['a', 'b', 'c', 'd']


def test_models(df, clear_sim):
    sim.add_table('test_table', df)

    @sim.model('test_model')
    def test_model(test_table):
        tt = test_table.to_frame()
        test_table['a'] = tt['a'] + tt['b']

    model = sim.get_model('test_model')
    model()

    table = sim.get_table('test_table')
    pdt.assert_frame_equal(
        table.to_frame(),
        pd.DataFrame(
            {'a': [5, 7, 9],
             'b': [4, 5, 6]},
            index=['x', 'y', 'z']))


def test_model_run(df, clear_sim):
    sim.add_table('test_table', df)

    @sim.table('table_func')
    def table_func(test_table):
        tt = test_table.to_frame()
        tt['c'] = [7, 8, 9]
        return tt

    @sim.column('table_func', 'new_col')
    def new_col(test_table, table_func):
        tt = test_table.to_frame()
        tf = table_func.to_frame(columns=['c'])
        return tt['a'] + tt['b'] + tf['c']

    @sim.model('test_model1')
    def test_model1(year, test_table, table_func):
        tf = table_func.to_frame(columns=['new_col'])
        test_table[year] = tf['new_col'] + year

    @sim.model('test_model2')
    def test_model2(test_table):
        tt = test_table.to_frame()
        test_table['a'] = tt['a'] ** 2

    sim.run(models=['test_model1', 'test_model2'], years=[2000, 3000])

    test_table = sim.get_table('test_table')
    assert_frames_equal(
        test_table.to_frame(),
        pd.DataFrame(
            {'a': [1, 16, 81],
             'b': [4, 5, 6],
             2000: [2012, 2015, 2018],
             3000: [3012, 3017, 3024]},
            index=['x', 'y', 'z']))


def test_get_broadcasts(clear_sim):
    sim.broadcast('a', 'b')
    sim.broadcast('b', 'c')
    sim.broadcast('z', 'b')
    sim.broadcast('f', 'g')

    with pytest.raises(ValueError):
        sim._get_broadcasts(['a', 'b', 'g'])

    assert set(sim._get_broadcasts(['a', 'b', 'c', 'z']).keys()) == \
        {('a', 'b'), ('b', 'c'), ('z', 'b')}
    assert set(sim._get_broadcasts(['a', 'b', 'z']).keys()) == \
        {('a', 'b'), ('z', 'b')}
    assert set(sim._get_broadcasts(['a', 'b', 'c']).keys()) == \
        {('a', 'b'), ('b', 'c')}


def test_collect_injectables(clear_sim, df):
    sim.add_table('df', df)

    @sim.table('df_func')
    def test_df():
        return df

    @sim.column('df', 'zzz')
    def zzz():
        return df['a'] / 2

    sim.add_injectable('answer', 42)

    @sim.injectable('injected')
    def injected():
        return 'injected'

    with pytest.raises(KeyError):
        sim._collect_injectables(['asdf'])

    names = ['df', 'df_func', 'answer', 'injected']
    things = sim._collect_injectables(names)

    assert set(things.keys()) == set(names)


def test_injectables(clear_sim):
    sim.add_injectable('answer', 42)

    @sim.injectable('func1')
    def inj_func1(answer):
        return answer * 2

    @sim.injectable('func2', autocall=False)
    def inj_func2(x):
        return x / 2

    @sim.injectable('func3')
    def inj_func3(func2):
        return func2(4)

    @sim.injectable('func4')
    def inj_func4(func1):
        return func1 / 2

    assert sim.get_injectable('answer') == 42
    assert sim.get_injectable('func1')() == 42 * 2
    assert sim.get_injectable('func2')(4) == 2
    assert sim.get_injectable('func3')() == 2
    assert sim.get_injectable('func4')() == 42


def test_injectables_combined(clear_sim, df):
    @sim.injectable('column')
    def column():
        return pd.Series(['a', 'b', 'c'], index=df.index)

    @sim.table('table')
    def table():
        return df

    @sim.model('model')
    def model(table, column):
        df = table.to_frame()
        df['new'] = column
        sim.add_table('table', df)

    sim.run(models=['model'])

    table = sim.get_table('table').to_frame()

    pdt.assert_frame_equal(table[['a', 'b']], df)
    pdt.assert_series_equal(table['new'], column())


def test_table_source(clear_sim, df):
    @sim.table_source('source')
    def source():
        return df

    table = sim.get_table('source')
    assert isinstance(table, sim._TableSourceWrapper)

    test_df = table.to_frame()
    pdt.assert_frame_equal(test_df, df)
    assert table.columns == list(df.columns)
    assert len(table) == len(df)
    pdt.assert_index_equal(table.index, df.index)

    table = sim.get_table('source')
    assert isinstance(table, sim._DataFrameWrapper)

    test_df = table.to_frame()
    pdt.assert_frame_equal(test_df, df)


def test_table_source_convert(clear_sim, df):
    @sim.table_source('source')
    def source():
        return df

    table = sim.get_table('source')
    assert isinstance(table, sim._TableSourceWrapper)

    table = table.convert()
    assert isinstance(table, sim._DataFrameWrapper)
    pdt.assert_frame_equal(table.to_frame(), df)

    table2 = sim.get_table('source')
    assert table2 is table


def test_table_func_local_cols(clear_sim, df):
    @sim.table('table')
    def table():
        return df
    sim.add_column('table', 'new', pd.Series(['a', 'b', 'c'], index=df.index))

    assert sim.get_table('table').local_columns == ['a', 'b']


def test_table_source_local_cols(clear_sim, df):
    @sim.table_source('source')
    def source():
        return df
    sim.add_column('source', 'new', pd.Series(['a', 'b', 'c'], index=df.index))

    assert sim.get_table('source').local_columns == ['a', 'b']
