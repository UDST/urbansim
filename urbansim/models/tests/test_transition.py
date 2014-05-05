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
def totals_col():
    return 'total'


@pytest.fixture
def grow_targets(year, totals_col):
    return pd.DataFrame({totals_col: [7]}, index=[year])


@pytest.fixture
def grow_targets_filters(year, totals_col):
    return pd.DataFrame({'x_min': [0, 2, np.nan],
                         'y_max': [7, 9, np.nan],
                         'x': [np.nan, np.nan, 4],
                         totals_col: [1, 4, 10]},
                        index=[year, year, year])


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


def test_add_or_remove_rows_add(basic_df):
    nrows = 2
    new = transition._add_or_remove_rows(basic_df, nrows)
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


def test_tabular_transition_add_fill(basic_df, grow_targets, totals_col, year):
    populate = True
    tran = transition.TabularTransitionModel(
        grow_targets, totals_col, populate)
    new = tran.transition(basic_df, year=year)
    assert_for_add_and_fill(new)


def test_tabular_transition_add_nofill(
        basic_df, grow_targets, totals_col, year):
    populate = False
    tran = transition.TabularTransitionModel(
        grow_targets, totals_col, populate)
    new = tran.transition(basic_df, year=year)
    assert_for_add_no_fill(new)


def test_tabular_transition_remove(basic_df, totals_col, year):
    grow_targets = pd.DataFrame({totals_col: [3]}, index=[year])
    tran = transition.TabularTransitionModel(grow_targets, totals_col)
    new = tran.transition(basic_df, year=year)
    assert_for_remove(new)


def test_tabular_transition_raises_on_bad_year(
        basic_df, grow_targets, totals_col, year):
    tran = transition.TabularTransitionModel(grow_targets, totals_col)

    with pytest.raises(ValueError):
        tran.transition(basic_df, year=year + 100)


def test_tabular_transition_add_filters(
        basic_df, grow_targets_filters, totals_col, year):
    populate = True
    tran = transition.TabularTransitionModel(
        grow_targets_filters, totals_col, populate)
    new = tran.transition(basic_df, year=year)

    assert len(new) == grow_targets_filters[totals_col].sum()
