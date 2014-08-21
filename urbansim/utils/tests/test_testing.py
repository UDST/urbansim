
import pandas as pd
import pytest

from .. import testing


def test_frames_equal_not_frames():
    frame = pd.DataFrame({'a': [1]})
    with pytest.raises(AssertionError) as info:
        testing.assert_frames_equal(frame, 1)

    assert info.value.message == 'Inputs must both be pandas DataFrames.'


def test_frames_equal_mismatched_columns():
    expected = pd.DataFrame({'a': [1]})
    actual = pd.DataFrame({'b': [2]})

    with pytest.raises(AssertionError) as info:
        testing.assert_frames_equal(actual, expected)

    assert info.value.message == "Expected column 'a' not found."


def test_frames_equal_mismatched_rows():
    expected = pd.DataFrame({'a': [1]}, index=[0])
    actual = pd.DataFrame({'a': [1]}, index=[1])

    with pytest.raises(AssertionError) as info:
        testing.assert_frames_equal(actual, expected)

    assert info.value.message == "Expected row 0 not found."


def test_frames_equal_mismatched_items():
    expected = pd.DataFrame({'a': [1]})
    actual = pd.DataFrame({'a': [2]})

    with pytest.raises(AssertionError) as info:
        testing.assert_frames_equal(actual, expected)

    assert info.value.message == """
Items are not equal:
 ACTUAL: 2
 DESIRED: 1

Column: 'a'
Row: 0"""


def test_frames_equal():
    frame = pd.DataFrame({'a': [1]})
    testing.assert_frames_equal(frame, frame)


def test_frames_equal_close():
    frame1 = pd.DataFrame({'a': [1]})
    frame2 = pd.DataFrame({'a': [1.00000000000002]})

    with pytest.raises(AssertionError):
        testing.assert_frames_equal(frame1, frame2)

    testing.assert_frames_equal(frame1, frame2, use_close=True)


def test_index_equal_order_agnostic():
    left = pd.Index([1, 2, 3])
    right = pd.Index([3, 2, 1])
    testing.assert_index_equal(left, right)


def test_index_equal_order_agnostic_raises_left():
    left = pd.Index([1, 2, 3, 4])
    right = pd.Index([3, 2, 1])
    with pytest.raises(AssertionError):
        testing.assert_index_equal(left, right)


def test_index_equal_order_agnostic_raises_right():
    left = pd.Index([1, 2, 3])
    right = pd.Index([3, 2, 1, 4])
    with pytest.raises(AssertionError):
        testing.assert_index_equal(left, right)
