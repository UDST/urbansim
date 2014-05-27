import numpy as np
import pandas as pd
import pytest
from patsy import PatsyError
from patsy.categorical import C
from patsy.missing import NAAction

from ..patsypatch import categorical_to_int, patch_patsy, sniff


def test_patch_patsy():
    patch_patsy()

    import patsy

    assert patsy.categorical.categorical_to_int.__code__ is \
        categorical_to_int.__code__
    assert patsy.categorical.CategoricalSniffer.sniff.__code__ is \
        sniff.__code__


def test_categorical_to_int():
    s = pd.Series(["a", "b", "c"], index=[10, 20, 30])
    c_pandas = categorical_to_int(s, ("a", "b", "c"), NAAction())
    assert np.all(c_pandas == [0, 1, 2])
    assert np.all(c_pandas.index == [10, 20, 30])
    # Input must be 1-dimensional
    pytest.raises(PatsyError,
                  categorical_to_int,
                  pd.DataFrame({10: s}), ("a", "b", "c"), NAAction())

    cat = pd.Categorical([1, 0, -1], ("a", "b"))
    conv = categorical_to_int(cat, ("a", "b"), NAAction())
    assert np.all(conv == [1, 0, -1])
    # Trust pandas NA marking
    cat2 = pd.Categorical([1, 0, -1], ("a", "None"))
    conv2 = categorical_to_int(cat, ("a", "b"), NAAction(NA_types=["None"]))
    assert np.all(conv2 == [1, 0, -1])
    # But levels must match
    pytest.raises(PatsyError,
                  categorical_to_int,
                  pd.Categorical([1, 0], ("a", "b")),
                  ("a", "c"),
                  NAAction())
    pytest.raises(PatsyError,
                  categorical_to_int,
                  pd.Categorical([1, 0], ("a", "b")),
                  ("b", "a"),
                  NAAction())

    def t(data, levels, expected, NA_action=NAAction()):
        got = categorical_to_int(data, levels, NA_action)
        assert np.array_equal(got, expected)

    t(["a", "b", "a"], ("a", "b"), [0, 1, 0])
    t(np.asarray(["a", "b", "a"]), ("a", "b"), [0, 1, 0])
    t(np.asarray(["a", "b", "a"], dtype=object), ("a", "b"), [0, 1, 0])
    t([0, 1, 2], (1, 2, 0), [2, 0, 1])
    t(np.asarray([0, 1, 2]), (1, 2, 0), [2, 0, 1])
    t(np.asarray([0, 1, 2], dtype=float), (1, 2, 0), [2, 0, 1])
    t(np.asarray([0, 1, 2], dtype=object), (1, 2, 0), [2, 0, 1])
    t(["a", "b", "a"], ("a", "d", "z", "b"), [0, 3, 0])
    t([("a", 1), ("b", 0), ("a", 1)], (("a", 1), ("b", 0)), [0, 1, 0])

    pytest.raises(PatsyError, categorical_to_int,
                  ["a", "b", "a"], ("a", "c"), NAAction())

    t(C(["a", "b", "a"]), ("a", "b"), [0, 1, 0])
    t(C(["a", "b", "a"]), ("b", "a"), [1, 0, 1])
    t(C(["a", "b", "a"], levels=["b", "a"]), ("b", "a"), [1, 0, 1])
    # Mismatch between C() levels and expected levels
    pytest.raises(PatsyError, categorical_to_int,
                  C(["a", "b", "a"], levels=["a", "b"]),
                  ("b", "a"), NAAction())

    # ndim == 2 is disallowed
    pytest.raises(PatsyError, categorical_to_int,
                  np.asarray([["a", "b"], ["b", "a"]]),
                  ("a", "b"), NAAction())
    # ndim == 0 is disallowed likewise
    pytest.raises(PatsyError, categorical_to_int,
                  "a",
                  ("a", "b"), NAAction())

    # levels must be hashable
    pytest.raises(PatsyError, categorical_to_int,
                  ["a", "b"], ("a", "b", {}), NAAction())
    pytest.raises(PatsyError, categorical_to_int,
                  ["a", "b", {}], ("a", "b"), NAAction())

    t(["b", None, np.nan, "a"], ("a", "b"), [1, -1, -1, 0],
      NAAction(NA_types=["None", "NaN"]))
    t(["b", None, np.nan, "a"], ("a", "b", None), [1, -1, -1, 0],
      NAAction(NA_types=["None", "NaN"]))
    t(["b", None, np.nan, "a"], ("a", "b", None), [1, 2, -1, 0],
      NAAction(NA_types=["NaN"]))

    # Smoke test for the branch that formats the ellipsized list of levels in
    # the error message:
    pytest.raises(PatsyError, categorical_to_int,
                  ["a", "b", "q"],
                  ("a", "b", "c", "d", "e", "f", "g", "h"),
                  NAAction())


def test_CategoricalSniffer():
    patch_patsy()

    from patsy.categorical import CategoricalSniffer

    def t(NA_types, datas, exp_finish_fast, exp_levels, exp_contrast=None):
        sniffer = CategoricalSniffer(NAAction(NA_types=NA_types))
        for data in datas:
            done = sniffer.sniff(data)
            if done:
                assert exp_finish_fast
                break
            else:
                assert not exp_finish_fast
        assert sniffer.levels_contrast() == (exp_levels, exp_contrast)

    t([], [pd.Categorical.from_array([1, 2, None])],
      True, (1, 2))
    # check order preservation
    t([], [pd.Categorical([1, 0], ["a", "b"])],
      True, ("a", "b"))
    t([], [pd.Categorical([1, 0], ["b", "a"])],
      True, ("b", "a"))
    # check that if someone sticks a .contrast field onto a Categorical
    # object, we pick it up:
    c = pd.Categorical.from_array(["a", "b"])
    c.contrast = "CONTRAST"
    t([], [c], True, ("a", "b"), "CONTRAST")

    t([], [C([1, 2]), C([3, 2])], False, (1, 2, 3))
    # check order preservation
    t([], [C([1, 2], levels=[1, 2, 3]), C([4, 2])], True, (1, 2, 3))
    t([], [C([1, 2], levels=[3, 2, 1]), C([4, 2])], True, (3, 2, 1))

    # do some actual sniffing with NAs in
    t(["None", "NaN"], [C([1, np.nan]), C([10, None])],
      False, (1, 10))
    # But 'None' can be a type if we don't make it represent NA:
    sniffer = CategoricalSniffer(NAAction(NA_types=["NaN"]))
    sniffer.sniff(C([1, np.nan, None]))
    # The level order here is different on py2 and py3 :-( Because there's no
    # consistent way to sort mixed-type values on both py2 and py3. Honestly
    # people probably shouldn't use this, but I don't know how to give a
    # sensible error.
    levels, _ = sniffer.levels_contrast()
    assert set(levels) == set([None, 1])

    # bool special case
    t(["None", "NaN"], [C([True, np.nan, None])],
      True, (False, True))
    t([], [C([10, 20]), C([False]), C([30, 40])],
      False, (False, True, 10, 20, 30, 40))

    # check tuples too
    t(["None", "NaN"], [C([("b", 2), None, ("a", 1), np.nan, ("c", None)])],
      False, (("a", 1), ("b", 2), ("c", None)))

    # contrasts
    t([], [C([10, 20], contrast="FOO")], False, (10, 20), "FOO")

    # unhashable level error:
    sniffer = CategoricalSniffer(NAAction())
    pytest.raises(PatsyError, sniffer.sniff, [{}])
