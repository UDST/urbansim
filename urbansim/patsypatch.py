"""
This module exists to help us with some performance headaches in patsy.
Specifically with categorical data. A lot of urbansim models have
boolean columns via expressions like I(year_built < 1940). patsy's
categorical code checks every single value in those columns to see
whether it is nan, but if we know the data are pandas.Series instances
we should be able to check the dtype to see if it is boolean and,
if so, take appropriate specific action that avoids looking at every
value.

The function ``patch_patsy`` monkeypatches a function and a method
in patsy with replacements that check the input data's dtype for
boolean and act appropriately.

"""
import numpy as np
import pandas as pd
from patsy import PatsyError
from patsy.categorical import _CategoricalBox, C
from patsy.util import (SortAnythingKey,
                        safe_scalar_isnan,
                        iterable)


# replaces function patsy.categorical.categorical_to_int
def categorical_to_int(data, levels, NA_action, origin=None):
    assert isinstance(levels, tuple)
    # In this function, missing values are always mapped to -1
    if isinstance(data, pd.Categorical):
        data_levels_tuple = tuple(data.levels)
        if not data_levels_tuple == levels:
            raise PatsyError("mismatching levels: expected %r, got %r"
                             % (levels, data_levels_tuple), origin)
        # pd.Categorical also uses -1 to indicate NA, and we don't try to
        # second-guess its NA detection, so we can just pass it back.
        return data.labels
    elif hasattr(data, 'dtype') and hasattr(data, 'astype') and \
            np.issubdtype(data.dtype, np.bool_):
        return data.astype('int')
    if isinstance(data, _CategoricalBox):
        if data.levels is not None and tuple(data.levels) != levels:
            raise PatsyError("mismatching levels: expected %r, got %r"
                             % (levels, tuple(data.levels)), origin)
        data = data.data
    if hasattr(data, "shape") and len(data.shape) > 1:
        raise PatsyError("categorical data must be 1-dimensional",
                         origin)
    if not iterable(data) or isinstance(data, basestring):
        raise PatsyError("categorical data must be an iterable container")
    try:
        level_to_int = dict(zip(levels, xrange(len(levels))))
    except TypeError:
        raise PatsyError("Error interpreting categorical data: "
                         "all items must be hashable", origin)
    out = np.empty(len(data), dtype=int)
    for i, value in enumerate(data):
        if NA_action.is_categorical_NA(value):
            out[i] = -1
        else:
            try:
                out[i] = level_to_int[value]
            except KeyError:
                SHOW_LEVELS = 4
                level_strs = []
                if len(levels) <= SHOW_LEVELS:
                    level_strs += [repr(level) for level in levels]
                else:
                    level_strs += [repr(level)
                                   for level in levels[:SHOW_LEVELS//2]]
                    level_strs.append("...")
                    level_strs += [repr(level)
                                   for level in levels[-SHOW_LEVELS//2:]]
                level_str = "[%s]" % (", ".join(level_strs))
                raise PatsyError("Error converting data to categorical: "
                                 "observation with value %r does not match "
                                 "any of the expected levels (expected: %s)"
                                 % (value, level_str), origin)
            except TypeError:
                raise PatsyError("Error converting data to categorical: "
                                 "encountered unhashable value %r"
                                 % (value,), origin)
    if isinstance(data, pd.Series):
        out = pd.Series(out, index=data.index)
    return out


# replaces method patsy.categorical.CategoricalSniffer.sniff
def sniff(self, data):
    if hasattr(data, "contrast"):
        self._contrast = data.contrast
    # returns a bool: are we confident that we found all the levels?
    if isinstance(data, pd.Categorical):
        # pandas.Categorical has its own NA detection, so don't try to
        # second-guess it.
        self._levels = tuple(data.levels)
        return True
    elif hasattr(data, 'dtype') and np.issubdtype(data.dtype, np.bool_):
        self._level_set = set([True, False])
        return True
    if isinstance(data, _CategoricalBox):
        if data.levels is not None:
            self._levels = tuple(data.levels)
            return True
        else:
            # unbox and fall through
            data = data.data
    for value in data:
        if self._NA_action.is_categorical_NA(value):
            continue
        if value is True or value is False:
            self._level_set.update([True, False])
        else:
            try:
                self._level_set.add(value)
            except TypeError:
                raise PatsyError("Error interpreting categorical data: "
                                 "all items must be hashable",
                                 self._origin)
    # If everything we've seen is boolean, assume that everything else
    # would be too. Otherwise we need to keep looking.
    return self._level_set == set([True, False])


def patch_patsy():
    """
    Replace the function patsy.categorical.categorical_to_int
    (in multiple places) and the method
    patsy.categorical.CategoricalSniffer.sniff with replacements
    of our own that have special handling for arrays with
    boolean dtypes.

    """
    import patsy
    patsy.categorical.categorical_to_int = categorical_to_int
    patsy.build.categorical_to_int = categorical_to_int
    patsy.categorical.CategoricalSniffer.sniff = sniff
