import pandas as pd
import pytest

from .. import sqftproforma as sqpf


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


@pytest.fixture
def simple_dev_inputs_high_cost():
    sdi = simple_dev_inputs()
    sdi.land_cost *= 20
    return sdi


@pytest.fixture
def simple_dev_inputs_low_cost():
    sdi = simple_dev_inputs()
    sdi.land_cost /= 5
    return sdi


def test_sqftproforma_config_defaults():
    sqpf.SqFtProFormaConfig()


def test_sqftproforma_defaults(simple_dev_inputs):
    pf = sqpf.SqFtProForma()

    for form in pf.config.forms:
        out = pf.lookup(form, simple_dev_inputs)
        if form == "industrial":
            assert len(out) == 0
        if form == "residential":
            assert len(out) == 3
        if form == "office":
            assert len(out) == 2


def test_sqftproforma_low_cost(simple_dev_inputs_low_cost):
    pf = sqpf.SqFtProForma()

    for form in pf.config.forms:
        out = pf.lookup(form, simple_dev_inputs_low_cost)
        if form == "industrial":
            assert len(out) == 3
        if form == "residential":
            assert len(out) == 3
        if form == "office":
            assert len(out) == 3


def test_sqftproforma_high_cost(simple_dev_inputs_high_cost):
    pf = sqpf.SqFtProForma()

    for form in pf.config.forms:
        out = pf.lookup(form, simple_dev_inputs_high_cost)
        if form == "industrial":
            assert len(out) == 0
        if form == "residential":
            assert len(out) == 0
        if form == "office":
            assert len(out) == 0


def test_sqftproforma_debug():
    pf = sqpf.SqFtProForma()
    pf._debug_output()
