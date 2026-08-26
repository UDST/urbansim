"""
Utilities used in testing of UrbanSim.

"""
import pandas as pd
import pandas.testing as pdt


def assert_frames_equal(actual, expected, use_close=False):
    """
    Compare DataFrame items by index and column and
    raise AssertionError if any item is not equal.

    Ordering is unimportant, items are compared only by label
    (``check_like=True``). NaN and infinite values are supported.

    Parameters
    ----------
    actual : pandas.DataFrame
    expected : pandas.DataFrame
    use_close : bool, optional
        If True, compare with ``assert_frame_equal(check_exact=False)``
        (numerical tolerance); otherwise compare exactly
        (``check_exact=True``).

    """
    assert (isinstance(actual, pd.DataFrame) and
            isinstance(expected, pd.DataFrame)), \
        'Inputs must both be pandas DataFrames.'

    pdt.assert_frame_equal(
        actual, expected, check_exact=not use_close, check_dtype=False,
        check_like=True)


def assert_index_equal(left, right):
    """
    Order-agnostic index equality: the indexes are equal if neither has
    keys the other lacks, regardless of ordering.

    Parameters
    ----------
    left: pandas.Index
    right: pandas.Index
    """
    assert isinstance(left, pd.Index)
    assert isinstance(right, pd.Index)
    left_diff = left.difference(right)
    right_diff = right.difference(left)
    if len(left_diff) > 0 or len(right_diff) > 0:
        raise AssertionError("keys not in left [{0}], keys not in right [{1}]".format(
            left_diff, right_diff))
