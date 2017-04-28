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

    result = misc.column_map([fta, ftb], ['aa', 'by', 'bz'])
    # misc.column_map() does not guarantee order, so sort for testing
    result_sorted = {k: sorted(v) for k, v in result.items()}

    assert result_sorted == {'a': ['aa'], 'b': ['by', 'bz']}

    result = misc.column_map([fta, ftb], ['by', 'bz'])
    result_sorted = {k: sorted(v) for k, v in result.items()}

    assert result_sorted == {'a': [], 'b': ['by', 'bz']}


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
    result = misc.column_list([fta, ftb], ['aa', 'by', 'bz', 'c'])
    assert sorted(result) == ['aa', 'by', 'bz']


######################
# FK REINDEX TESTS
######################


@pytest.fixture()
def left_df():
    return pd.DataFrame({
        'some_val': [10, 9, 8, 7, 6],
        'fk': ['z', 'g', 'g', 'b', 't'],
        'grp': ['r', 'g', 'r', 'g', 'r']
    })


@pytest.fixture()
def right_df():
    return pd.DataFrame(
        {
            'col1': [100, 200, 50],
            'col2': [1, 2, 3]
        },
        index=pd.Index(['g', 'b', 'z'])
    )


@pytest.fixture()
def right_df2(right_df):
    df = pd.concat([right_df, right_df * -1])
    df['fk'] = df.index
    df['grp'] = ['r', 'r', 'r', 'g', 'g', 'g']
    df.set_index(['fk', 'grp'], inplace=True)
    return df


def test_fidx_right_not_unique(right_df, left_df):
    with pytest.raises(ValueError):
        s = right_df.col1
        misc.fidx(s.append(s), left_df.fk)


def test_series_fidx(right_df, left_df):
    b = misc.fidx(right_df.col1, left_df.fk).fillna(-1)
    assert (b.values == [50, 100, 100, 200, -1]).all()


def assert_df_fidx(b):
    assert (b.col1.values == [50, 100, 100, 200, -1]).all()
    assert (b.col2.values == [3, 1, 1, 2, -1]).all()


def test_df_fidx(right_df, left_df):
    b = misc.fidx(right_df, left_df.fk).fillna(-1)
    assert_df_fidx(b)


def test_fk_reindex_with_fk_col(right_df, left_df):
    b = misc.fidx(right_df, left_df, 'fk').fillna(-1)
    assert_df_fidx(b)


def test_series_multi_col_fidx(right_df2, left_df):
    b = misc.fidx(right_df2.col1, left_df, ['fk', 'grp']).fillna(-9999)
    assert (b.values == [50, -100, 100, -200, -9999]).all()


def test_df_multi_col_fidx(right_df2, left_df):
    b = misc.fidx(right_df2, left_df, ['fk', 'grp']).fillna(-9999)
    assert (b.col1.values == [50, -100, 100, -200, -9999]).all()
    assert (b.col2.values == [3, -1, 1, -2, -9999]).all()
