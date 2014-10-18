import numpy as np
import pandas as pd
import pytest

from urbansim.models.allocation import AllocationModel, AgentAllocationModel


@pytest.fixture(scope='function')
def amounts_df():
    return pd.DataFrame(
        {'amount': [150, 250, 300]},
        index=[2010, 2011, 2012])


@pytest.fixture(scope='function')
def amounts_df_decline():
    return pd.DataFrame(
        {'amount': [33, 15, 40]},
        index=[2010, 2011, 2012])


@pytest.fixture(scope='function')
def rows_df():
    return pd.DataFrame(
        {
            'weight': [50, 0, 25],
            'capacity': [60, 70, 80],
            'existing': [10, 11, 12],
            'floor': [-1, 0, 1]
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


@pytest.fixture
def segment_cols():
    return ['taz']


@pytest.fixture
def seg_amounts_df():
    return pd.DataFrame(
        {
            'taz': [1, 2, 1, 2],
            'amount': [100, 200, 150, 250],
            'another_amount': [10, 20, 15, 25]
        },
        index=pd.Index([2010, 2010, 2011, 2011]))


@pytest.fixture
def seg_rows_df():
    return pd.DataFrame(
        {
            'taz': [1, 1, 1, 2, 2, 2],
            'weight': [50, 0, 25, 10, 20, 30],
            'capacity': [60, 70, 80, 100, 110, 120],
            'existing': [10, 11, 12, 20, 21, 22]
        })


def assert_totals_match(amounts_df,
                        amounts_col,
                        results,
                        target_col,
                        year):
    curr_amount = amounts_df[amounts_col][year]
    assert curr_amount == results.sum()


def assert_totals_match_delta(amounts_df,
                              amounts_col,
                              orig_df,
                              results,
                              target_col,
                              year):
    amount = amounts_df[amounts_col][year]
    prev_amount = amounts_df[amounts_col][year - 1]
    expected_change = amount - prev_amount
    observed_change = results.sum() - orig_df[target_col].sum()
    assert expected_change == observed_change


def assert_capacities(results, orig_df, capacity_col, floor_col=None):
    c = orig_df[capacity_col]
    over = results > c
    if floor_col is not None:
        under = results < orig_df[floor_col]
    else:
        under = results < 0
    assert not (over.any() or under.any())


def test_noWeights_noCapacity(amounts_df,
                              amounts_col,
                              rows_df,
                              target_col):
    year = 2010
    am = AllocationModel(amounts_df, amounts_col, target_col)
    results = am.allocate(rows_df, year)
    assert_totals_match(amounts_df, amounts_col, results, target_col, year)


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
    results = am.allocate(rows_df, year)
    assert_totals_match_delta(amounts_df, amounts_col, rows_df, results, target_col, year)


def test_hasWeights_noCapacity(amounts_df,
                               amounts_col,
                               rows_df,
                               target_col,
                               weights_col):
    year = 2010
    am = AllocationModel(amounts_df, amounts_col, target_col, weights_col)
    results = am.allocate(rows_df, year)
    assert_totals_match(amounts_df, amounts_col, results, target_col, year)


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
    results = am.allocate(rows_df, year)
    assert_totals_match_delta(amounts_df, amounts_col, rows_df, results, target_col, year)


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
    results = am.allocate(rows_df, year)
    assert_totals_match(amounts_df, amounts_col, results, target_col, year)
    assert_capacities(results, rows_df, capacity_col)


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
    results = am.allocate(rows_df, year)
    assert_totals_match_delta(amounts_df, amounts_col, rows_df, results, target_col, year)
    assert_capacities(results, rows_df, capacity_col)


def test_segmentation(seg_amounts_df,
                      amounts_col,
                      seg_rows_df,
                      target_col,
                      weights_col,
                      capacity_col,
                      segment_cols):
    year = 2010
    amod = AllocationModel(
        seg_amounts_df,
        amounts_col,
        target_col,
        weights_col,
        capacity_col,
        segment_cols=segment_cols)
    results = amod.allocate(seg_rows_df, year)

    # check resulting sums by segments
    results_df = seg_rows_df.copy()
    results_df[target_col] = results
    result_sums = results_df.groupby(segment_cols).sum()
    join = pd.merge(
        left=result_sums,
        right=seg_amounts_df.loc[year],
        left_index=True,
        right_on=segment_cols,
        how='outer')
    mismatches = len(join[join[target_col] != join[amounts_col]])
    assert mismatches == 0


def test_segmentation__delta(seg_amounts_df,
                             amounts_col,
                             seg_rows_df,
                             target_col,
                             weights_col,
                             capacity_col,
                             segment_cols):
    year = 2011

    # set up the model
    amod = AllocationModel(
        seg_amounts_df,
        amounts_col,
        target_col,
        weights_col,
        capacity_col,
        as_delta=True,
        compute_delta=True,
        segment_cols=segment_cols)

    # get the allocated change
    orig_sums = seg_rows_df.groupby(segment_cols).sum()
    results = amod.allocate(seg_rows_df, year)
    results_df = seg_rows_df.copy()
    results_df[target_col] = results
    result_sums = results_df.groupby(segment_cols).sum()
    change_sums = result_sums - orig_sums

    # check resulting sums by segments
    join1 = pd.merge(
        left=seg_amounts_df.loc[year],
        right=seg_amounts_df.loc[year - 1],
        on=segment_cols,
        how='outer')

    join2 = pd.merge(
        left=change_sums,
        right=join1,
        left_index=True,
        right_on=segment_cols,
        how='outer')
    amount_change = join2[amounts_col + '_x'] - join2[amounts_col + '_y']
    mismatches = len(join2[join2[target_col] != amount_change])
    assert mismatches == 0


def test_missing_year(amounts_df, amounts_col, rows_df, target_col):
    year = 2020
    am = AllocationModel(amounts_df, amounts_col, target_col)
    orig_cnt = len(rows_df)
    orig_sum = rows_df[target_col].sum()
    results = am.allocate(rows_df, year)
    assert orig_cnt == len(results)
    assert orig_sum == results.sum()


def test_declining_amount_delta(amounts_df_decline,
                                amounts_col,
                                rows_df,
                                target_col,
                                weights_col,
                                capacity_col):
    year = 2011
    am = AllocationModel(
        amounts_df_decline,
        amounts_col,
        target_col,
        weights_col,
        capacity_col,
        as_delta=True,
        compute_delta=True)
    results = am.allocate(rows_df, year)
    assert_totals_match_delta(
        amounts_df_decline, amounts_col, rows_df, results, target_col, year)
    assert_capacities(results, rows_df, capacity_col)


def test_with_floor_column(amounts_df_decline,
                           amounts_col,
                           rows_df,
                           target_col,
                           weights_col,
                           capacity_col):
    year = 2011
    am = AllocationModel(
        amounts_df_decline,
        amounts_col,
        target_col,
        weights_col,
        capacity_col,
        as_delta=True,
        compute_delta=True,
        floor_col='floor')
    results = am.allocate(rows_df, year)
    assert_totals_match_delta(
        amounts_df_decline, amounts_col, rows_df, results, target_col, year)
    assert_capacities(results, rows_df, capacity_col, 'floor')


def test_raise_too_many_years(amounts_df,
                              amounts_col,
                              target_col):
    year = 2010
    amounts_df.index = pd.Index([2010, 2010, 2011])
    with pytest.raises(ValueError):
        am = AllocationModel(amounts_df, amounts_col, target_col)


def test_not_enough_capacity(amounts_df,
                             amounts_col,
                             rows_df,
                             target_col,
                             weights_col,
                             capacity_col):
    year = 2010
    amounts_df[amounts_col][year] = 10000
    am = AllocationModel(
        amounts_df,
        amounts_col,
        target_col,
        weights_col,
        capacity_col)
    results = am.allocate(rows_df, year)
    assert_capacities(results, rows_df, capacity_col)
    # add check to make sure allocation = capacities?


def test_raise_not_enough_capacity_decline(amounts_df,
                                           amounts_col,
                                           rows_df,
                                           target_col,
                                           weights_col,
                                           capacity_col):
    year = 2010
    amounts_df[amounts_col][year] = -10000
    am = AllocationModel(
        amounts_df,
        amounts_col,
        target_col,
        weights_col,
        capacity_col)
    results = am.allocate(rows_df, year)
    assert_capacities(results, rows_df, capacity_col)
    # add check to make sure allocation = capacities?


def test_raise_float_amount_as_int(amounts_df,
                                   amounts_col,
                                   rows_df,
                                   target_col):
    year = 2010
    amounts_df[amounts_col] = pd.Series([100.56, 200.56, 300.56])
    am = AllocationModel(
        amounts_df,
        amounts_col,
        target_col)
    with pytest.raises(ValueError):
        am.allocate(rows_df, year)
