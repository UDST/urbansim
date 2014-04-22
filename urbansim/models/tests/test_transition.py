import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from pandas.util import testing as pdt

from .. import transition


@pytest.fixture
def basic_df():
    return pd.DataFrame(
        {'x': range(5),
         'y': range(5, 10)},
        index=range(100, 105))


@pytest.fixture
def year():
    return 2112


@pytest.fixture
def grow_targets(year):
    return pd.Series([7], index=[year])


def assert_for_add_no_fill(new):
    assert len(new) == 7
    assert np.isnan(new.index.values[-2:].astype(np.float)).all()


def assert_for_add_and_fill(new):
    assert len(new) == 7
    assert not np.isnan(new.index.values[-2:].astype(np.float)).any()


def assert_for_remove(new):
    assert len(new) == 3


def test_add_rows(basic_df):
    nrows = 2
    new = transition.add_rows(basic_df, nrows)
    assert_for_add_no_fill(new)


def test_add_rows_zero(basic_df):
    nrows = 0
    new = transition.add_rows(basic_df, nrows)
    pdt.assert_frame_equal(new, basic_df)


def test_remove_rows(basic_df):
    nrows = 2
    new = transition.remove_rows(basic_df, nrows)
    assert_for_remove(new)


def test_remove_rows_zero(basic_df):
    nrows = 0
    new = transition.remove_rows(basic_df, nrows)
    pdt.assert_frame_equal(new, basic_df)


def test_remove_rows_raises(basic_df):
    # should raise ValueError if asked to remove more rows than
    # are in the table
    nrows = 25

    with pytest.raises(ValueError):
        transition.remove_rows(basic_df, nrows)


def test_fill_nan_ids(basic_df):
    nrows = 2
    new = transition.add_rows(basic_df, nrows)
    new = transition.fill_nan_ids(new)

    npt.assert_array_equal(new.index.values, range(100, 100 + 7))


def test_add_or_remove_rows_add_fill(basic_df):
    nrows = 2
    populate = True
    new = transition._add_or_remove_rows(basic_df, nrows, populate)
    assert_for_add_and_fill(new)


def test_add_or_remove_rows_add_nofill(basic_df):
    nrows = 2
    populate = False
    new = transition._add_or_remove_rows(basic_df, nrows, populate)
    assert_for_add_no_fill(new)


def test_add_or_remove_rows_remove(basic_df):
    nrows = -2
    new = transition._add_or_remove_rows(basic_df, nrows)
    assert_for_remove(new)


def test_grtransition_add_fills(basic_df):
    growth_rate = 0.4
    populate = True
    grt = transition.GRTransitionModel(growth_rate, populate)
    new = grt.transition(basic_df)
    assert_for_add_and_fill(new)


def test_grtransition_add_nofill(basic_df):
    growth_rate = 0.4
    populate = False
    grt = transition.GRTransitionModel(growth_rate, populate)
    new = grt.transition(basic_df)
    assert_for_add_no_fill(new)


def test_grtransition_remove(basic_df):
    growth_rate = -0.4
    grt = transition.GRTransitionModel(growth_rate)
    new = grt.transition(basic_df)
    assert_for_remove(new)


def test_tabular_transition_add_fill(basic_df, grow_targets, year):
    populate = True
    tran = transition.TabularTransitionModel(grow_targets, populate)
    new = tran.transition(basic_df, year=year)
    assert_for_add_and_fill(new)


def test_tabular_transition_add_nofill(basic_df, grow_targets, year):
    populate = False
    tran = transition.TabularTransitionModel(grow_targets, populate)
    new = tran.transition(basic_df, year=year)
    assert_for_add_no_fill(new)


def test_tabular_transition_remove(basic_df, year):
    grow_targets = pd.Series([3], index=[year])
    tran = transition.TabularTransitionModel(grow_targets)
    new = tran.transition(basic_df, year=year)
    assert_for_remove(new)


def test_tabular_transition_raises_on_bad_year(basic_df, grow_targets, year):
    tran = transition.TabularTransitionModel(grow_targets)

    with pytest.raises(ValueError):
        tran.transition(basic_df, year=year + 100)
