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


@pytest.fixture
def two_form_feasibility(simple_dev_inputs):
    # office out-earns residential on parcels a and c, and is not feasible
    # at all on parcel b, so the two forms split the parcels between them
    inputs = simple_dev_inputs.copy()
    inputs['office'] = [50, 15, 50]
    pf = sqpf.SqFtProForma()
    return {"residential": pf.lookup("residential", inputs),
            "office": pf.lookup("office", inputs)}


@pytest.fixture
def pick_args():
    index = ['a', 'b', 'c']
    return dict(target_units=1000,
                parcel_size=pd.Series([1000, 1000, 1000], index=index),
                ave_unit_size=pd.Series([650, 650, 650], index=index),
                current_units=pd.Series([0, 0, 0], index=index))


@pytest.mark.parametrize('form', [None, ["residential", "office"]])
def test_developer_pick_competing_forms(two_form_feasibility, pick_args, form):
    # residential wins only on parcel b, so that's the only residential
    # building; office wins on a and c
    dev = developer.Developer(two_form_feasibility)
    bldgs = dev.pick(form, **pick_args)
    assert bldgs.parcel_id.tolist() == ['b']
    assert bldgs.form.tolist() == ['residential']

    dev = developer.Developer(two_form_feasibility)
    bldgs = dev.pick(form, residential=False, **pick_args)
    assert sorted(bldgs.parcel_id) == ['a', 'c']
    assert bldgs.form.tolist() == ['office', 'office']


def test_developer_pick_flat_feasibility(simple_dev_inputs, pick_args):
    # a flat table of attributes for a single form can be passed directly,
    # and form=None then uses it as is
    pf = sqpf.SqFtProForma()
    out = pf.lookup("residential", simple_dev_inputs)
    assert not isinstance(out.columns, pd.MultiIndex)

    dev = developer.Developer({"residential": out.copy()})
    expected = dev.pick("residential", **pick_args)

    dev = developer.Developer(out)
    bldgs = dev.pick(None, **pick_args)
    assert bldgs.parcel_id.tolist() == expected.parcel_id.tolist() == ['a', 'b', 'c']


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
