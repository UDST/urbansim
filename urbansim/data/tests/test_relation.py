import pandas as pd
import pandas.util.testing as pdt
import pytest

from ..relation import Relation


@pytest.fixture
def small_df():
    return pd.DataFrame(
        {'a': ['abc', 'def', 'ghi'],
         'other_id': [200, 201, 200]},
        index=[100, 101, 102])


def test_basic(small_df):
    df2 = pd.DataFrame(
        {'x': ['123', '456']},
        index=[200, 201])

    rel = Relation(small_df, df2, left_on='other_id')
    pdt.assert_series_equal(rel._merged_index, small_df.other_id)
    pdt.assert_series_equal(rel['a'], small_df.a)
    pdt.assert_series_equal(rel.a, small_df.a)
    pdt.assert_series_equal(
        rel['x'], pd.Series(['123', '456', '123'], index=small_df.index))
    pdt.assert_series_equal(
        rel.x, pd.Series(['123', '456', '123'], index=small_df.index))

    with pytest.raises(KeyError):
        rel['no-such-column']


def test_right_not_all_used(small_df):
    df2 = pd.DataFrame(
        {'x': ['123', '456', '789']},
        index=[200, 201, 202])

    rel = Relation(small_df, df2, left_on='other_id')
    pdt.assert_series_equal(rel._merged_index, small_df.other_id)
    pdt.assert_series_equal(rel['a'], small_df.a)
    pdt.assert_series_equal(rel.a, small_df.a)
    pdt.assert_series_equal(
        rel['x'], pd.Series(['123', '456', '123'], index=small_df.index))
    pdt.assert_series_equal(
        rel.x, pd.Series(['123', '456', '123'], index=small_df.index))
