import pandas as pd
import pytest

from .. import sqftproforma as sqpf
from .. import developer


@pytest.fixture
def simple_dev_inputs():
    return pd.DataFrame(
        {'residential': [40, 40, 40],
         'office': [15, 18, 15],
         'retail': [12, 10, 10],
         'industrial': [12, 12, 12],
         'land_cost': [1000000, 2000000, 3000000],
         'parcel_size': [10000, 20000, 30000],
         'max_far': [2.0, 3.0, 4.0],
         'max_height': [40, 60, 80]},
        index=['a', 'b', 'c'])


def test_developer(simple_dev_inputs):
    pf = sqpf.SqFtProForma()

    out = pf.lookup("residential", simple_dev_inputs)
    dev = developer.Developer({"residential": out})
    target_units = 10
    parcel_size = pd.Series([1000, 1000, 1000], index=['a', 'b', 'c'])
    ave_unit_size = pd.Series([650, 650, 650], index=['a', 'b', 'c'])
    current_units = pd.Series([0, 0, 0], index=['a', 'b', 'c'])
    bldgs = dev.pick("residential", target_units, parcel_size, ave_unit_size,
                     current_units)
    assert len(bldgs) == 1

    # bldgs = dev.pick(["residential", "office"], target_units,
    #                 parcel_size, ave_unit_size, current_units)
    # assert len(bldgs) == 1

    target_units = 1000
    bldgs = dev.pick("residential", target_units, parcel_size, ave_unit_size,
                     current_units)
    assert len(bldgs) == 2

    target_units = 2
    bldgs = dev.pick("residential", target_units, parcel_size, ave_unit_size,
                     current_units, residential=False)
    assert bldgs is None

    target_units = 2
    bldgs = dev.pick("residential", target_units, parcel_size, ave_unit_size,
                     current_units, residential=False)
    assert bldgs is None


def test_developer_compute_units_to_build(simple_dev_inputs):
    pf = sqpf.SqFtProForma()
    out = pf.lookup("residential", simple_dev_inputs)
    dev = developer.Developer({"residential": out})
    to_build = dev.compute_units_to_build(30, 30, .1)
    assert int(to_build) == 3


def test_developer_compute_forms_max_profit(simple_dev_inputs):
    pf = sqpf.SqFtProForma()
    out = pf.lookup("residential", simple_dev_inputs)
    dev = developer.Developer({"residential": out})
    dev.keep_form_with_max_profit()


def test_developer_merge():
    df1 = pd.DataFrame({'test': [1]}, index=[1])
    df2 = pd.DataFrame({'test': [1]}, index=[1])
    dev = developer.Developer.merge(df1, df2)
    # make sure index is unique
    assert dev.index.values[1] == 2
