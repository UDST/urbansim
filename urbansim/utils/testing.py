"""
Utilities used in testing of UrbanSim.

"""
import numpy as np
import numpy.testing as npt
import pandas as pd


def assert_frames_equal(actual, expected, use_close=False):
    """
    Compare DataFrame items by index and column and
    raise AssertionError if any item is not equal.

    Ordering is unimportant, items are compared only by label.
    NaN and infinite values are supported.

    Parameters
    ----------
    actual : pandas.DataFrame
    expected : pandas.DataFrame
    use_close : bool, optional
        If True, use numpy.testing.assert_allclose instead of
        numpy.testing.assert_equal.

    """
    if use_close:
        comp = npt.assert_allclose
    else:
        comp = npt.assert_equal

    assert (isinstance(actual, pd.DataFrame) and
            isinstance(expected, pd.DataFrame)), \
        'Inputs must both be pandas DataFrames.'

    for i, exp_row in expected.iterrows():
        assert i in actual.index, 'Expected row {!r} not found.'.format(i)

        act_row = actual.loc[i]

        for j, exp_item in exp_row.iteritems():
            assert j in act_row.index, \
                'Expected column {!r} not found.'.format(j)

            act_item = act_row[j]

            try:
                comp(act_item, exp_item)
            except AssertionError as e:
                raise AssertionError(
                    e.message + '\n\nColumn: {!r}\nRow: {!r}'.format(j, i))
