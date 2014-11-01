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

    @sim.table()
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

    @sim.table(cache=True)
    def table(variable='x'):
        return df * variable

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
    def asdf(x):
        return df * x

    sim.disable_cache()

    pdt.assert_frame_equal(sim.get_table('table').to_frame(), df * 2)
    sim.add_injectable('x', 3)
    pdt.assert_frame_equal(sim.get_table('table').to_frame(), df * 3)

    sim.enable_cache()

    sim.add_injectable('x', 4)
    pdt.assert_frame_equal(sim.get_table('table').to_frame(), df * 3)


def test_table_copy(df):
    sim.add_table('test_frame_copied', df, copy=True)
    sim.add_table('test_frame_uncopied', df, copy=False)
    sim.add_table('test_func_copied', lambda: df, copy=True)
    sim.add_table('test_func_uncopied', lambda: df, copy=False)

    @sim.table(copy=True)
    def test_funcd_copied():
        return df

    @sim.table(copy=False)
    def test_funcd_uncopied():
        return df

    @sim.table(copy=False)
    def test_funcd_copied2(test_frame_copied):
        return test_frame_copied.to_frame()

    @sim.table(copy=True)
    def test_funcd_copied3(test_frame_copied):
        return test_frame_copied.to_frame()

    @sim.table(copy=True)
    def test_funcd_copied4(test_frame_uncopied):
        return test_frame_uncopied.to_frame()

    @sim.table(copy=False)
    def test_funcd_uncopied2(test_frame_uncopied):
        return test_frame_uncopied.to_frame()

    sim.add_table_source('test_source_copied', lambda: df, copy=True)
    sim.add_table_source('test_source_uncopied', lambda: df, copy=False)

    @sim.table_source(copy=True)
    def test_sourced_copied():
        return df

    @sim.table_source(copy=False)
    def test_sourced_uncopied():
        return df

    sim.add_table('test_copied_columns', pd.DataFrame(index=df.index),
                  copy=True)
    sim.add_table('test_uncopied_columns', pd.DataFrame(index=df.index),
                  copy=False)

    @sim.column('test_copied_columns', 'a')
    def copied_column(col='test_frame_uncopied.a'):
        return col

    @sim.column('test_uncopied_columns', 'a')
    def uncopied_column(col='test_frame_uncopied.a'):
        return col

    for name in ['test_frame_copied', 'test_func_copied',
                 'test_funcd_copied', 'test_funcd_copied2',
                 'test_funcd_copied3', 'test_funcd_copied4',
                 'test_source_copied', 'test_sourced_copied',
                 'test_copied_columns']:
        table = sim.get_table(name)
        if 'columns' not in name:
            # Run these tests for tables without computed columns.
            pdt.assert_frame_equal(table.to_frame(), df)
            assert table.to_frame() is not df
            pdt.assert_frame_equal(table.to_frame(), table.to_frame())
            assert table.to_frame() is not table.to_frame()
            pdt.assert_series_equal(table.to_frame()['a'], df['a'])
            assert table.to_frame()['a'] is not df['a']
            pdt.assert_series_equal(table.to_frame()['a'],
                                    table.to_frame()['a'])
            assert table.to_frame()['a'] is not table.to_frame()['a']
        pdt.assert_series_equal(table['a'], df['a'])
        assert table['a'] is not df['a']
        pdt.assert_series_equal(table['a'], table['a'])
        assert table['a'] is not table['a']

    for name in ['test_frame_uncopied', 'test_func_uncopied',
                 'test_funcd_uncopied', 'test_funcd_uncopied2',
                 'test_source_uncopied', 'test_sourced_uncopied',
                 'test_uncopied_columns']:
        table = sim.get_table(name)
        if 'columns' not in name:
            # Run these tests for tables without computed columns.
            pdt.assert_frame_equal(table.to_frame(), df)
            assert table.to_frame() is df
            pdt.assert_frame_equal(table.to_frame(), table.to_frame())
            assert table.to_frame() is table.to_frame()
            pdt.assert_series_equal(table.to_frame()['a'], df['a'])
            assert table.to_frame()['a'] is df['a']
            pdt.assert_series_equal(table.to_frame()['a'],
                                    table.to_frame()['a'])
            assert table.to_frame()['a'] is table.to_frame()['a']
        pdt.assert_series_equal(table['a'], df['a'])
        assert table['a'] is df['a']
        pdt.assert_series_equal(table['a'], table['a'])
        assert table['a'] is table['a']


def test_columns_for_table():
    sim.add_column(
        'table1', 'col10', pd.Series([1, 2, 3], index=['a', 'b', 'c']))
    sim.add_column(
        'table2', 'col20', pd.Series([10, 11, 12], index=['x', 'y', 'z']))

    @sim.column('table1')
    def col11():
        return pd.Series([4, 5, 6], index=['a', 'b', 'c'])

    @sim.column('table2', 'col21')
    def asdf():
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

    @sim.table()
    def test_func(test_frame):
        return test_frame.to_frame() / 2

    sim.add_column('test_frame', 'c', pd.Series([7, 8, 9], index=df.index))

    @sim.column('test_func', 'd')
    def asdf(test_func):
        return test_func.to_frame(columns=['b'])['b'] * 2

    @sim.column('test_func')
    def e(column='test_func.d'):
        return column + 1

    test_frame = sim.get_table('test_frame')
    assert set(test_frame.columns) == set(['a', 'b', 'c'])
    assert_frames_equal(
        test_frame.to_frame(),
        pd.DataFrame(
            {'a': [1, 2, 3],
             'b': [4, 5, 6],
             'c': [7, 8, 9]},
            index=['x', 'y', 'z']))
    assert_frames_equal(
        test_frame.to_frame(columns=['a', 'c']),
        pd.DataFrame(
            {'a': [1, 2, 3],
             'c': [7, 8, 9]},
            index=['x', 'y', 'z']))

    test_func_df = sim.get_table('test_func')
    assert set(test_func_df.columns) == set(['d', 'e'])
    assert_frames_equal(
        test_func_df.to_frame(),
        pd.DataFrame(
            {'a': [0.5, 1, 1.5],
             'b': [2, 2.5, 3],
             'c': [3.5, 4, 4.5],
             'd': [4., 5., 6.],
             'e': [5., 6., 7.]},
            index=['x', 'y', 'z']))
    assert_frames_equal(
        test_func_df.to_frame(columns=['b', 'd']),
        pd.DataFrame(
            {'b': [2, 2.5, 3],
             'd': [4., 5., 6.]},
            index=['x', 'y', 'z']))
    assert set(test_func_df.columns) == set(['a', 'b', 'c', 'd', 'e'])

    assert set(sim.list_columns()) == {('test_frame', 'c'), ('test_func', 'd'),
                                       ('test_func', 'e')}


def test_column_cache(df):
    sim.add_injectable('x', 2)
    series = pd.Series([1, 2, 3], index=['x', 'y', 'z'])
    key = ('table', 'col')

    @sim.table()
    def table():
        return df

    @sim.column(*key, cache=True)
    def column(variable='x'):
        return series * variable

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
    sim.add_column(*key, column=column, cache=True)
    pdt.assert_series_equal(c()(), series * 6)


def test_column_cache_disabled(df):
    sim.add_injectable('x', 2)
    series = pd.Series([1, 2, 3], index=['x', 'y', 'z'])
    key = ('table', 'col')

    @sim.table()
    def table():
        return df

    @sim.column(*key, cache=True)
    def column(x):
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

    df2 = df / 2
    sim.add_table('test_table2', df2)

    @sim.model()
    def test_model(test_table, test_column='test_table2.b'):
        tt = test_table.to_frame()
        test_table['a'] = tt['a'] + tt['b']
        pdt.assert_series_equal(test_column, df2['b'])

    with pytest.raises(KeyError):
        sim.get_model('asdf')

    model = sim.get_model('test_model')
    assert model._tables_used() == set(['test_table', 'test_table2'])
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

    @sim.table()
    def table_func(test_table):
        tt = test_table.to_frame()
        tt['c'] = [7, 8, 9]
        return tt

    @sim.column('table_func')
    def new_col(test_table, table_func):
        tt = test_table.to_frame()
        tf = table_func.to_frame(columns=['c'])
        return tt['a'] + tt['b'] + tf['c']

    @sim.model()
    def test_model1(year, test_table, table_func):
        tf = table_func.to_frame(columns=['new_col'])
        test_table[year] = tf['new_col'] + year

    @sim.model('test_model2')
    def asdf(table='test_table'):
        tt = table.to_frame()
        table['a'] = tt['a'] ** 2

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


def test_collect_variables(df):
    sim.add_table('df', df)

    @sim.table()
    def df_func():
        return df

    @sim.column('df')
    def zzz():
        return df['a'] / 2

    sim.add_injectable('answer', 42)

    @sim.injectable()
    def injected():
        return 'injected'

    @sim.table_source('source table')
    def source():
        return df

    with pytest.raises(KeyError):
        sim._collect_variables(['asdf'])

    with pytest.raises(KeyError):
        sim._collect_variables(names=['df'], expressions=['asdf'])

    names = ['df', 'df_func', 'answer', 'injected', 'source_label', 'df_a']
    expressions = ['source table', 'df.a']
    things = sim._collect_variables(names, expressions)

    assert set(things.keys()) == set(names)
    assert isinstance(things['source_label'], sim.DataFrameWrapper)
    pdt.assert_frame_equal(things['source_label']._frame, df)
    assert isinstance(things['df_a'], pd.Series)
    pdt.assert_series_equal(things['df_a'], df['a'])


def test_collect_variables_expression_only(df):
    @sim.table()
    def table():
        return df

    vars = sim._collect_variables(['a'], ['table.a'])
    pdt.assert_series_equal(vars['a'], df.a)


def test_injectables():
    sim.add_injectable('answer', 42)

    @sim.injectable()
    def func1(answer):
        return answer * 2

    @sim.injectable('func2', autocall=False)
    def asdf(variable='x'):
        return variable / 2

    @sim.injectable()
    def func3(func2):
        return func2(4)

    @sim.injectable()
    def func4(func='func1'):
        return func / 2

    assert sim._INJECTABLES['answer'] == 42
    assert sim._INJECTABLES['func1']() == 42 * 2
    assert sim._INJECTABLES['func2'](4) == 2
    assert sim._INJECTABLES['func3']() == 2
    assert sim._INJECTABLES['func4']() == 42

    assert sim.get_injectable('answer') == 42
    assert sim.get_injectable('func1') == 42 * 2
    assert sim.get_injectable('func2')(4) == 2
    assert sim.get_injectable('func3') == 2
    assert sim.get_injectable('func4') == 42

    with pytest.raises(KeyError):
        sim.get_injectable('asdf')

    assert set(sim.list_injectables()) == \
        {'answer', 'func1', 'func2', 'func3', 'func4'}


def test_injectables_combined(df):
    @sim.injectable()
    def column():
        return pd.Series(['a', 'b', 'c'], index=df.index)

    @sim.table()
    def table():
        return df

    @sim.model()
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

    @sim.injectable(autocall=True, cache=True)
    def inj():
        return x * x

    i = lambda: sim._INJECTABLES['inj']

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

    @sim.injectable(autocall=True, cache=True)
    def inj():
        return x * x

    i = lambda: sim._INJECTABLES['inj']

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
    @sim.table_source()
    def source():
        return df

    _source = lambda: sim._TABLES['source']

    table = _source()
    assert isinstance(table, sim._TableSourceWrapper)

    test_df = table.to_frame()
    pdt.assert_frame_equal(test_df, df)
    assert table.columns == list(df.columns)
    assert len(table) == len(df)
    pdt.assert_index_equal(table.index, df.index)

    table = _source()
    assert isinstance(table, sim.DataFrameWrapper)

    test_df = table.to_frame()
    pdt.assert_frame_equal(test_df, df)


def test_table_source_convert(df):
    @sim.table_source()
    def source():
        return df

    _source = lambda: sim._TABLES['source']

    table = _source()
    assert isinstance(table, sim._TableSourceWrapper)

    table = table.convert()
    assert isinstance(table, sim.DataFrameWrapper)
    pdt.assert_frame_equal(table.to_frame(), df)

    table2 = _source()
    assert table2 is table


def test_table_func_local_cols(df):
    @sim.table()
    def table():
        return df
    sim.add_column('table', 'new', pd.Series(['a', 'b', 'c'], index=df.index))

    assert sim.get_table('table').local_columns == ['a', 'b']


def test_table_source_local_cols(df):
    @sim.table_source()
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

    @sim.model()
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

    @sim.model()
    def model(year, table):
        table[year_key(year)] = series_year(year)

    sim.run(['model'], years=range(11), data_out=store_name, out_interval=3)

    with pd.get_store(store_name, mode='r') as store:
        for year in range(3, 11, 3):
            key = '{}/table'.format(year)
            assert key in store

            for x in range(year):
                pdt.assert_series_equal(
                    store[key][year_key(x)], series_year(x))

        assert 'base/table' in store

        for x in range(11):
            pdt.assert_series_equal(
                store['final/table'][year_key(x)], series_year(x))


def test_get_table(df):
    sim.add_table('frame', df)

    @sim.table()
    def table():
        return df

    @sim.table_source()
    def source():
        return df

    fr = sim.get_table('frame')
    ta = sim.get_table('table')
    so = sim.get_table('source')

    with pytest.raises(KeyError):
        sim.get_table('asdf')

    assert isinstance(fr, sim.DataFrameWrapper)
    assert isinstance(ta, sim.TableFuncWrapper)
    assert isinstance(so, sim.DataFrameWrapper)

    pdt.assert_frame_equal(fr.to_frame(), df)
    pdt.assert_frame_equal(ta.to_frame(), df)
    pdt.assert_frame_equal(so.to_frame(), df)


def test_cache_disabled_cm():
    x = 3

    @sim.injectable(cache=True)
    def xi():
        return x

    assert sim.get_injectable('xi') == 3
    x = 5
    assert sim.get_injectable('xi') == 3

    with sim.cache_disabled():
        assert sim.get_injectable('xi') == 5

    # cache still gets updated even when cacheing is off
    assert sim.get_injectable('xi') == 5


def test_injectables_cm():
    sim.add_injectable('a', 'a')
    sim.add_injectable('b', 'b')
    sim.add_injectable('c', 'c')

    with sim.injectables():
        assert sim._INJECTABLES == {
            'a': 'a', 'b': 'b', 'c': 'c'
        }

    with sim.injectables(c='d', x='x', y='y', z='z'):
        assert sim._INJECTABLES == {
            'a': 'a', 'b': 'b', 'c': 'd',
            'x': 'x', 'y': 'y', 'z': 'z'
        }

    assert sim._INJECTABLES == {
        'a': 'a', 'b': 'b', 'c': 'c'
    }


def test_is_expression():
    assert sim.is_expression('name') is False
    assert sim.is_expression('table.column') is True


def test_eval_variable(df):
    sim.add_injectable('x', 3)
    assert sim.eval_variable('x') == 3

    @sim.injectable()
    def func(x):
        return 'xyz' * x
    assert sim.eval_variable('func') == 'xyzxyzxyz'
    assert sim.eval_variable('func', x=2) == 'xyzxyz'

    @sim.table()
    def table(x):
        return df * x
    pdt.assert_series_equal(sim.eval_variable('table.a'), df.a * 3)


def test_eval_model(df):
    sim.add_injectable('x', 3)

    @sim.model()
    def model(x):
        return df * x

    pdt.assert_frame_equal(sim.eval_model('model'), df * 3)
    pdt.assert_frame_equal(sim.eval_model('model', x=5), df * 5)
