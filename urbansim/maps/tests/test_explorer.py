import pytest
import pandas as pd

from .. import dframe_explorer


@pytest.fixture
def simple_map_input():
    return pd.DataFrame(
        {'test_var': [40, 50, 60],
         'zone_id': [10, 10, 20]},
        index=['a', 'b', 'c'])


def test_explorer(simple_map_input):
    dframe_explorer.enable_cors()
    dframe_explorer.ans_options()
    dframe_explorer.data_static("test")

    d = {"dfname": simple_map_input}
    dframe_explorer.start(d, testing=True)

    dframe_explorer.map_query("dfname", "empty", "zone_id", "test_var", "mean()")

    dframe_explorer.map_query("dfname", "empty", "zone_id", "test_var > 1", "mean()")

    dframe_explorer.index()

    with pytest.raises(Exception):
        dframe_explorer.start(d, host="failure", testing=False)

    d = {"dfname": simple_map_input[["test_var"]]}

    with pytest.raises(Exception):
        dframe_explorer.start(d, testing=True)
