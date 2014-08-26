import numpy as np
import pandas as pd
import pytest

from urbansim.models.allocation import AllocationModel


@pytest.fixture
def amounts_df():
    return pd.DataFrame(
        {'amount': [150, 250, 300]},
        index=[2010, 2011, 2012])


@pytest.fixture
def rows_df():
    return pd.DataFrame(
        {
            'weight': [50, 0, 25],
            'capacity': [60, 70, 80],
            'existing': [10, 11, 12]
        })


@pytest.fixture
def amounts_col():
    return 'amount'


@pytest.fixture
def target_col():
    return 'existing'


@pytest.fixture
def weights_col():
    return 'weight'


@pytest.fixture
def capacity_col():
    return 'capacity'


def test_noWeights_noCapacity(amounts_df,
                              amounts_col,
                              rows_df,
                              target_col):
    year = 2010
    am = AllocationModel(
        amounts_df,
        amounts_col,
        target_col)
    results_df = am.allocate(rows_df, year)

    # check allocation amounts
    amount = amounts_df[amounts_col][year]
    assert amount == results_df[target_col].sum()


def test_noWeights_noCapacity__delta(amounts_df,
                                     amounts_col,
                                     rows_df,
                                     target_col):
    year = 2011
    am = AllocationModel(
        amounts_df,
        amounts_col,
        target_col,
        as_delta=True,
        compute_delta=True)
    results_df = am.allocate(rows_df, year)

    # result total should match amount
    amount = amounts_df[amounts_col][year]
    prev_amount = amounts_df[amounts_col][year - 1]
    expected_change = amount - prev_amount
    observed_change = results_df[target_col].sum() - rows_df[target_col].sum()
    assert expected_change == observed_change


def test_hasWeights_noCapacity(amounts_df,
                               amounts_col,
                               rows_df,
                               target_col,
                               weights_col):
    year = 2010
    am = AllocationModel(
        amounts_df,
        amounts_col,
        target_col,
        weights_col)
    results_df = am.allocate(rows_df, year)

    # check allocation amounts
    amount = amounts_df[amounts_col][year]
    assert amount == results_df[target_col].sum()


def test_hasWeights_noCapacity__delta(amounts_df,
                                      amounts_col,
                                      rows_df,
                                      target_col,
                                      weights_col):
    year = 2011
    am = AllocationModel(
        amounts_df,
        amounts_col,
        target_col,
        weights_col,
        as_delta=True,
        compute_delta=True)
    results_df = am.allocate(rows_df, year)

    # result total should match amount
    amount = amounts_df[amounts_col][year]
    prev_amount = amounts_df[amounts_col][year - 1]
    expected_change = amount - prev_amount
    observed_change = results_df[target_col].sum() - rows_df[target_col].sum()
    assert expected_change == observed_change


def test_hasWeights_hasCapacity(amounts_df,
                                amounts_col,
                                rows_df,
                                target_col,
                                weights_col,
                                capacity_col):
    year = 2010
    am = AllocationModel(
        amounts_df,
        amounts_col,
        target_col,
        weights_col,
        capacity_col)
    results_df = am.allocate(rows_df, year)

    # check allocation amounts
    amount = amounts_df[amounts_col][year]
    assert amount == results_df[target_col].sum()

    # check capacities
    c = rows_df[capacity_col]
    a = results_df[target_col]
    assert len(c[a > c]) == 0


def test_hasWeights_hasCapacity__delta(amounts_df,
                                       amounts_col,
                                       rows_df,
                                       target_col,
                                       weights_col,
                                       capacity_col):
    year = 2011
    am = AllocationModel(
        amounts_df,
        amounts_col,
        target_col,
        weights_col,
        capacity_col,
        as_delta=True,
        compute_delta=True)
    results_df = am.allocate(rows_df, year)

    # result total should match amount
    amount = amounts_df[amounts_col][year]
    prev_amount = amounts_df[amounts_col][year - 1]
    expected_change = amount - prev_amount
    observed_change = results_df[target_col].sum() - rows_df[target_col].sum()
    assert expected_change == observed_change

    # check capacities
    c = rows_df[capacity_col]
    a = results_df[target_col]
    assert len(c[a > c]) == 0
