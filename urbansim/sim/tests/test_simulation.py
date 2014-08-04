import os
import tempfile

import pandas as pd
import pytest
from pandas.util import testing as pdt

from .. import simulation as sim
from ...utils.testing import assert_frames_equal


def setup_function(func):
    sim.clear_sim()
    sim.enable_cache()


def teardown_function(func):
    sim.clear_sim()
    sim.enable_cache()


@pytest.fixture
def df():
    return pd.DataFrame(
        {'a': [1, 2, 3],
         'b': [4, 5, 6]},
        index=['x', 'y', 'z'])


def test_tables(df):
    wrapped_df = sim.add_table('test_frame', df)

    @sim.table('test_func')
    def test_func(test_frame):
        return test_frame.to_frame() / 2

    assert set(sim.list_tables()) == {'test_frame', 'test_func'}

    table = sim.get_table('test_frame')
    assert table is wrapped_df
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


def test_table_func_cache(df):
    sim.add_injectable('x', 2)

    @sim.table('table', cache=True)
    def table(x):
        return df * x

    pdt.assert_frame_equal(sim.get_table('table').to_frame(), df * 2)
    sim.add_injectable('x', 3)
    pdt.assert_frame_equal(sim.get_table('table').to_frame(), df * 2)
    sim.get_table('table').clear_cached()
    pdt.assert_frame_equal(sim.get_table('table').to_frame(), df * 3)
    sim.add_injectable('x', 4)
    pdt.assert_frame_equal(sim.get_table('table').to_frame(), df * 3)
    sim.clear_cache()
    pdt.assert_frame_equal(sim.get_table('table').to_frame(), df * 4)
    sim.add_injectable('x', 5)
    pdt.assert_frame_equal(sim.get_table('table').to_frame(), df * 4)
    sim.add_table('table', table)
    pdt.assert_frame_equal(sim.get_table('table').to_frame(), df * 5)


def test_table_func_cache_disabled(df):
    sim.add_injectable('x', 2)

    @sim.table('table', cache=True)
    def table(x):
        return df * x

    sim.disable_cache()

    pdt.assert_frame_equal(sim.get_table('table').to_frame(), df * 2)
    sim.add_injectable('x', 3)
    pdt.assert_frame_equal(sim.get_table('table').to_frame(), df * 3)

    sim.enable_cache()

    sim.add_injectable('x', 4)
    pdt.assert_frame_equal(sim.get_table('table').to_frame(), df * 3)


def test_columns_for_table():
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


def test_columns_and_tables(df):
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

    assert set(sim.list_columns()) == {('test_frame', 'c'), ('test_func', 'd')}


def test_column_cache(df):
    sim.add_injectable('x', 2)
    series = pd.Series([1, 2, 3], index=['x', 'y', 'z'])
    key = ('table', 'col')

    @sim.table('table')
    def table():
        return df

    @sim.column(*key, cache=True)
    def col(x):
        return series * x

    c = lambda: sim._COLUMNS[key]

    pdt.assert_series_equal(c()(), series * 2)
    sim.add_injectable('x', 3)
    pdt.assert_series_equal(c()(), series * 2)
    c().clear_cached()
    pdt.assert_series_equal(c()(), series * 3)
    sim.add_injectable('x', 4)
    pdt.assert_series_equal(c()(), series * 3)
    sim.clear_cache()
    pdt.assert_series_equal(c()(), series * 4)
    sim.add_injectable('x', 5)
    pdt.assert_series_equal(c()(), series * 4)
    sim.get_table('table').clear_cached()
    pdt.assert_series_equal(c()(), series * 5)
    sim.add_injectable('x', 6)
    pdt.assert_series_equal(c()(), series * 5)
    sim.add_column(*key, column=col, cache=True)
    pdt.assert_series_equal(c()(), series * 6)


def test_column_cache_disabled(df):
    sim.add_injectable('x', 2)
    series = pd.Series([1, 2, 3], index=['x', 'y', 'z'])
    key = ('table', 'col')

    @sim.table('table')
    def table():
        return df

    @sim.column(*key, cache=True)
    def col(x):
        return series * x

    c = lambda: sim._COLUMNS[key]

    sim.disable_cache()

    pdt.assert_series_equal(c()(), series * 2)
    sim.add_injectable('x', 3)
    pdt.assert_series_equal(c()(), series * 3)

    sim.enable_cache()

    sim.add_injectable('x', 4)
    pdt.assert_series_equal(c()(), series * 3)


def test_update_col(df):
    wrapped = sim.add_table('table', df)

    wrapped.update_col('b', pd.Series([7, 8, 9], index=df.index))
    pdt.assert_series_equal(wrapped['b'], pd.Series([7, 8, 9], index=df.index))

    wrapped.update_col_from_series('a', pd.Series([]))
    pdt.assert_series_equal(wrapped['a'], df['a'])

    wrapped.update_col_from_series('a', pd.Series([99], index=['y']))
    pdt.assert_series_equal(
        wrapped['a'], pd.Series([1, 99, 3], index=df.index))


def test_models(df):
    sim.add_table('test_table', df)

    @sim.model('test_model')
    def test_model(test_table):
        tt = test_table.to_frame()
        test_table['a'] = tt['a'] + tt['b']

    model = sim.get_model('test_model')
    assert model._tables_used() == ['test_table']
    model()

    table = sim.get_table('test_table')
    pdt.assert_frame_equal(
        table.to_frame(),
        pd.DataFrame(
            {'a': [5, 7, 9],
             'b': [4, 5, 6]},
            index=['x', 'y', 'z']))

    assert sim.list_models() == ['test_model']


def test_model_run(df):
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

    m = sim.get_model('test_model1')
    assert set(m._tables_used()) == {'test_table', 'table_func'}


def test_get_broadcasts():
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

    assert set(sim.list_broadcasts()) == \
        {('a', 'b'), ('b', 'c'), ('z', 'b'), ('f', 'g')}


def test_collect_injectables(df):
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

    @sim.table_source('source')
    def source():
        return df

    with pytest.raises(KeyError):
        sim._collect_injectables(['asdf'])

    names = ['df', 'df_func', 'answer', 'injected', 'source']
    things = sim._collect_injectables(names)

    assert set(things.keys()) == set(names)
    assert isinstance(things['source'], sim.DataFrameWrapper)
    pdt.assert_frame_equal(things['source']._frame, df)


def test_injectables():
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

    assert set(sim.list_injectables()) == \
        {'answer', 'func1', 'func2', 'func3', 'func4'}


def test_injectables_combined(df):
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

    table_wr = sim.get_table('table').to_frame()

    pdt.assert_frame_equal(table_wr[['a', 'b']], df)
    pdt.assert_series_equal(table_wr['new'], column())


def test_injectables_cache():
    x = 2

    @sim.injectable('inj', autocall=True, cache=True)
    def inj():
        return x * x

    i = lambda: sim.get_injectable('inj')

    assert i()() == 4
    x = 3
    assert i()() == 4
    i().clear_cached()
    assert i()() == 9
    x = 4
    assert i()() == 9
    sim.clear_cache()
    assert i()() == 16
    x = 5
    assert i()() == 16
    sim.add_injectable('inj', inj, autocall=True, cache=True)
    assert i()() == 25


def test_injectables_cache_disabled():
    x = 2

    @sim.injectable('inj', autocall=True, cache=True)
    def inj():
        return x * x

    i = lambda: sim.get_injectable('inj')

    sim.disable_cache()

    assert i()() == 4
    x = 3
    assert i()() == 9

    sim.enable_cache()

    assert i()() == 9
    x = 4
    assert i()() == 9

    sim.disable_cache()
    assert i()() == 16


def test_table_source(df):
    @sim.table_source('source')
    def source():
        return df

    table = sim.get_table('source')
    assert isinstance(table, sim.TableSourceWrapper)

    test_df = table.to_frame()
    pdt.assert_frame_equal(test_df, df)
    assert table.columns == list(df.columns)
    assert len(table) == len(df)
    pdt.assert_index_equal(table.index, df.index)

    table = sim.get_table('source')
    assert isinstance(table, sim.DataFrameWrapper)

    test_df = table.to_frame()
    pdt.assert_frame_equal(test_df, df)


def test_table_source_convert(df):
    @sim.table_source('source')
    def source():
        return df

    table = sim.get_table('source')
    assert isinstance(table, sim.TableSourceWrapper)

    table = table.convert()
    assert isinstance(table, sim.DataFrameWrapper)
    pdt.assert_frame_equal(table.to_frame(), df)

    table2 = sim.get_table('source')
    assert table2 is table


def test_table_func_local_cols(df):
    @sim.table('table')
    def table():
        return df
    sim.add_column('table', 'new', pd.Series(['a', 'b', 'c'], index=df.index))

    assert sim.get_table('table').local_columns == ['a', 'b']


def test_table_source_local_cols(df):
    @sim.table_source('source')
    def source():
        return df
    sim.add_column('source', 'new', pd.Series(['a', 'b', 'c'], index=df.index))

    assert sim.get_table('source').local_columns == ['a', 'b']


def test_is_table(df):
    sim.add_table('table', df)
    assert sim._is_table('table') is True
    assert sim._is_table('asdf') is False


@pytest.fixture
def store_name(request):
    fname = tempfile.NamedTemporaryFile(suffix='.h5').name

    def fin():
        if os.path.isfile(fname):
            os.remove(fname)
    request.addfinalizer(fin)

    return fname


def test_write_tables(df, store_name):
    sim.add_table('table', df)

    @sim.model('model')
    def model(table):
        pass

    sim.write_tables(store_name, ['model'], None)

    with pd.get_store(store_name, mode='r') as store:
        assert 'table' in store
        pdt.assert_frame_equal(store['table'], df)

    sim.write_tables(store_name, ['model'], 1969)

    with pd.get_store(store_name, mode='r') as store:
        assert '1969/table' in store
        pdt.assert_frame_equal(store['1969/table'], df)


def test_run_and_write_tables(df, store_name):
    sim.add_table('table', df)

    year_key = lambda y: '{}'.format(y)
    series_year = lambda y: pd.Series([y] * 3, index=df.index)

    @sim.model('model')
    def model(year, table):
        table[year_key(year)] = series_year(year)

    sim.run(['model'], years=range(11), data_out=store_name, out_interval=3)

    with pd.get_store(store_name, mode='r') as store:
        for year in range(0, 11, 3) + [10]:
            key = '{}/table'.format(year)
            assert key in store

            for x in range(year):
                pdt.assert_series_equal(
                    store[key][year_key(x)], series_year(x))
