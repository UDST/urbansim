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


def test_add_rows(basic_df):
    nrows = 2
    new = transition.add_rows(basic_df, nrows)

    assert len(new) == 7
    assert np.isnan(new.index.values[-2:].astype(np.float)).all()


def test_add_rows_zero(basic_df):
    nrows = 0
    new = transition.add_rows(basic_df, nrows)
    pdt.assert_frame_equal(new, basic_df)


def test_remove_rows(basic_df):
    nrows = 2
    new = transition.remove_rows(basic_df, nrows)
    assert len(new) == 3


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


def test_grtransition_add_fills(basic_df):
    growth_rate = 0.4
    populate = True
    grt = transition.GRTransitionModel(growth_rate, populate)
    new = grt.transition(basic_df)

    assert len(new) == 7
    assert not np.isnan(new.index.values[-2:].astype(np.float)).any()


def test_grtransition_add_nofill(basic_df):
    growth_rate = 0.4
    populate = False
    grt = transition.GRTransitionModel(growth_rate, populate)
    new = grt.transition(basic_df)

    assert len(new) == 7
    assert np.isnan(new.index.values[-2:].astype(np.float)).all()


def test_grtransition_remove(basic_df):
    growth_rate = -0.4
    grt = transition.GRTransitionModel(growth_rate)
    new = grt.transition(basic_df)

    assert len(new) == 3
