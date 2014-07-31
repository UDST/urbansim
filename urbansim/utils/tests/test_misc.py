import os
import shutil

import numpy as np
import pandas as pd
import pytest

from .. import misc


class _FakeTable(object):
    def __init__(self, name, columns):
        self.name = name
        self.columns = columns


@pytest.fixture
def fta():
    return _FakeTable('a', ['aa', 'ab', 'ac'])


@pytest.fixture
def ftb():
    return _FakeTable('b', ['bx', 'by', 'bz'])


@pytest.fixture
def clean_fake_data_home(request):
    def fin():
        if os.path.isdir('fake_data_home'):
            shutil.rmtree('fake_data_home')
    request.addfinalizer(fin)


def test_column_map_raises(fta, ftb):
    with pytest.raises(RuntimeError):
        misc.column_map([fta, ftb], ['aa', 'by', 'bz', 'cw'])


def test_column_map_none(fta, ftb):
    assert misc.column_map([fta, ftb], None) == {'a': None, 'b': None}


def test_column_map(fta, ftb):
    assert misc.column_map([fta, ftb], ['aa', 'by', 'bz']) == \
        {'a': ['aa'], 'b': ['by', 'bz']}
    assert misc.column_map([fta, ftb], ['by', 'bz']) == \
        {'a': [], 'b': ['by', 'bz']}


def test_dirs(clean_fake_data_home):
    misc._mkifnotexists("fake_data_home")
    os.environ["DATA_HOME"] = "fake_data_home"
    misc.get_run_number()
    misc.get_run_number()
    misc.data_dir()
    misc.configs_dir()
    misc.models_dir()
    misc.charts_dir()
    misc.maps_dir()
    misc.simulations_dir()
    misc.reports_dir()
    misc.runs_dir()
    misc.config("test")


@pytest.fixture
def range_df():
    df = pd.DataFrame({'to_zone_id': [2, 3, 4],
                       'from_zone_id': [1, 1, 1],
                       'distance': [.1, .2, .9]})
    df = df.set_index(['from_zone_id', 'to_zone_id'])
    return df


@pytest.fixture
def range_series():
    return pd.Series([10, 150, 75, 275], index=[1, 2, 3, 4])


def test_compute_range(range_df, range_series):
    assert misc.compute_range(range_df, range_series, "distance", .5).loc[1] == 225


def test_reindex():
    s = pd.Series([.5, 1.0, 1.5], index=[2, 1, 3])
    s2 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
    assert list(misc.reindex(s, s2).values) == [1.0, .5, 1.5]


def test_naics():
    assert misc.naicsname(54) == "Professional"


def test_signif():
    assert misc.signif(4.0) == '***'
    assert misc.signif(3.0) == '**'
    assert misc.signif(2.0) == '*'
    assert misc.signif(1.5) == '.'
    assert misc.signif(1.0) == ''


@pytest.fixture
def simple_dev_inputs():
    return pd.DataFrame(
        {'residential': [40, 40, 40],
         'office': [15, 18, 15],
         'retail': [12, 10, 10],
         'industrial': [12, 12, 12],
         'land_cost': [1000000, 2000000, 3000000],
         'parcel_size': [10000, 20000, 30000],
         'max_far': [2.0, 3.0, 4.0],
         'names': ['a', 'b', 'c'],
         'max_height': [40, 60, 80]},
        index=['a', 'b', 'c'])


def test_misc_dffunctions(simple_dev_inputs):
    misc.df64bitto32bit(simple_dev_inputs)
    misc.pandasdfsummarytojson(simple_dev_inputs[['land_cost', 'parcel_size']])
    misc.numpymat2df(np.array([[1, 2], [3, 4]]))


def test_column_list(fta, ftb):
    assert misc.column_list([fta, ftb], ['aa', 'by', 'bz', 'c']) == \
        ['aa', 'by', 'bz']
