import pandas as pd
import pytest
from pandas.util import testing as pdt

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

    merged = sim.merge_tables('c', [dfa, dfz, dfb, dfc])

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
