import pandas as pd
import pytest

from .. import simulation as sim
from ...utils.testing import assert_frames_equal


def setup_function(func):
    sim.clear_sim()


def teardown_function(func):
    sim.clear_sim()


@pytest.fixture
def dfa():
    return sim.DataFrameWrapper('a', pd.DataFrame(
        {'a1': [1, 2, 3],
         'a2': [4, 5, 6],
         'a3': [7, 8, 9]},
        index=['aa', 'ab', 'ac']))


@pytest.fixture
def dfz():
    return sim.DataFrameWrapper('z', pd.DataFrame(
        {'z1': [90, 91],
         'z2': [92, 93],
         'z3': [94, 95],
         'z4': [96, 97],
         'z5': [98, 99]},
        index=['za', 'zb']))


@pytest.fixture
def dfb():
    return sim.DataFrameWrapper('b', pd.DataFrame(
        {'b1': range(10, 15),
         'b2': range(15, 20),
         'a_id': ['ac', 'ac', 'ab', 'aa', 'ab'],
         'z_id': ['zb', 'zb', 'za', 'za', 'zb']},
        index=['ba', 'bb', 'bc', 'bd', 'be']))


@pytest.fixture
def dfc():
    return sim.DataFrameWrapper('c', pd.DataFrame(
        {'c1': range(20, 30),
         'c2': range(30, 40),
         'b_id': ['ba', 'bd', 'bb', 'bc', 'bb', 'ba', 'bb', 'bc', 'bd', 'bb']},
        index=['ca', 'cb', 'cc', 'cd', 'ce', 'cf', 'cg', 'ch', 'ci', 'cj']))


@pytest.fixture
def dfg():
    return sim.DataFrameWrapper('g', pd.DataFrame(
        {'g1': [1, 2, 3]},
        index=['ga', 'gb', 'gc']))


@pytest.fixture
def dfh():
    return sim.DataFrameWrapper('h', pd.DataFrame(
        {'h1': range(10, 15),
         'g_id': ['ga', 'gb', 'gc', 'ga', 'gb']},
        index=['ha', 'hb', 'hc', 'hd', 'he']))


def all_broadcasts():
    sim.broadcast('a', 'b', cast_index=True, onto_on='a_id')
    sim.broadcast('z', 'b', cast_index=True, onto_on='z_id')
    sim.broadcast('b', 'c', cast_index=True, onto_on='b_id')
    sim.broadcast('g', 'h', cast_index=True, onto_on='g_id')


def test_recursive_getitem():
    assert sim._recursive_getitem({'a': {}}, 'a') == {'a': {}}
    assert sim._recursive_getitem(
        {'a': {'b': {'c': {'d': {}, 'e': {}}}}}, 'e') == {'d': {}, 'e': {}}

    with pytest.raises(KeyError):
        sim._recursive_getitem({'a': {'b': {'c': {'d': {}, 'e': {}}}}}, 'f')


def test_dict_value_to_pairs():
    assert sorted(sim._dict_value_to_pairs({'c': {'a': 1, 'b': 2}}),
                  key=lambda d: next(iter(d))) == \
        [{'a': 1}, {'b': 2}]


def test_is_leaf_node():
    assert sim._is_leaf_node({'b': {'a': {}}}) is False
    assert sim._is_leaf_node({'a': {}}) is True


def test_next_merge():
    assert sim._next_merge({'d': {'c': {}, 'b': {'a': {}}}}) == \
        {'b': {'a': {}}}
    assert sim._next_merge({'b': {'a': {}, 'z': {}}}) == \
        {'b': {'a': {}, 'z': {}}}


def test_merge_tables_raises(dfa, dfz, dfb, dfg, dfh):
    all_broadcasts()

    with pytest.raises(RuntimeError):
        sim.merge_tables('b', [dfa, dfb, dfz, dfg, dfh])


def test_merge_tables1(dfa, dfz, dfb):
    all_broadcasts()

    merged = sim.merge_tables('b', [dfa, dfz, dfb])

    expected = pd.merge(
        dfa.to_frame(), dfb.to_frame(), left_index=True, right_on='a_id')
    expected = pd.merge(
        expected, dfz.to_frame(), left_on='z_id', right_index=True)

    assert_frames_equal(merged, expected)


def test_merge_tables2(dfa, dfz, dfb, dfc):
    all_broadcasts()

    merged = sim.merge_tables(dfc, [dfa, dfz, dfb, dfc])

    expected = pd.merge(
        dfa.to_frame(), dfb.to_frame(), left_index=True, right_on='a_id')
    expected = pd.merge(
        expected, dfz.to_frame(), left_on='z_id', right_index=True)
    expected = pd.merge(
        expected, dfc.to_frame(), left_index=True, right_on='b_id')

    assert_frames_equal(merged, expected)


def test_merge_tables_cols(dfa, dfz, dfb, dfc):
    all_broadcasts()

    merged = sim.merge_tables(
        'c', [dfa, dfz, dfb, dfc], columns=['a1', 'b1', 'z1', 'c1'])

    expected = pd.DataFrame(
        {'c1': range(20, 30),
         'b1': [10, 13, 11, 12, 11, 10, 11, 12, 13, 11],
         'a1': [3, 1, 3, 2, 3, 3, 3, 2, 1, 3],
         'z1': [91, 90, 91, 90, 91, 91, 91, 90, 90, 91]},
        index=['ca', 'cb', 'cc', 'cd', 'ce', 'cf', 'cg', 'ch', 'ci', 'cj'])

    assert_frames_equal(merged, expected)


def test_merge_tables3():
    df_a = pd.DataFrame(
        {'a': [0, 1]},
        index=['a0', 'a1'])
    df_b = pd.DataFrame(
        {'b': [2, 3, 4, 5, 6],
         'a_id': ['a0', 'a1', 'a1', 'a0', 'a1']},
        index=['b0', 'b1', 'b2', 'b3', 'b4'])
    df_c = pd.DataFrame(
        {'c': [7, 8, 9]},
        index=['c0', 'c1', 'c2'])
    df_d = pd.DataFrame(
        {'d': [10, 11, 12, 13, 15, 16, 16, 17, 18, 19],
         'b_id': ['b2', 'b0', 'b3', 'b3', 'b1', 'b4', 'b1', 'b4', 'b3', 'b3'],
         'c_id': ['c0', 'c1', 'c1', 'c0', 'c0', 'c2', 'c1', 'c2', 'c1', 'c2']},
        index=['d0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9'])

    sim.add_table('a', df_a)
    sim.add_table('b', df_b)
    sim.add_table('c', df_c)
    sim.add_table('d', df_d)

    sim.broadcast(cast='a', onto='b', cast_index=True, onto_on='a_id')
    sim.broadcast(cast='b', onto='d', cast_index=True, onto_on='b_id')
    sim.broadcast(cast='c', onto='d', cast_index=True, onto_on='c_id')

    df = sim.merge_tables(target='d', tables=['a', 'b', 'c', 'd'])

    expected = pd.merge(df_a, df_b, left_index=True, right_on='a_id')
    expected = pd.merge(expected, df_d, left_index=True, right_on='b_id')
    expected = pd.merge(df_c, expected, left_index=True, right_on='c_id')

    assert_frames_equal(df, expected)
