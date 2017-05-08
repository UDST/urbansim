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
    return ['taz', 'district']


@pytest.fixture(scope='function')
def seg_amounts_df():
    return pd.DataFrame(
        {
            'taz': [1, 2, 3, 3, 1, 2, 3, 3],
            'district': ['a', 'b', 'c', 'd', 'a', 'b', 'c', 'd'],
            'amount': [100, 200, 20, 20, 150, 250, 20, 10]
        },
        index=pd.Index([2010, 2010, 2010, 2010, 2011, 2011, 2011, 2011]))


@pytest.fixture(scope='function')
def seg_rows_df():
    return pd.DataFrame(
        {
            'taz': [1, 1, 1, 2, 2, 2, 3, 3],
            'district': ['a', 'a', 'a', 'b', 'b', 'b', 'c', 'd'],
            'weight': [50, 0, 25, 10, 20, 30, 10, 0],
            'capacity': [60, 70, 80, 100, 110, 120, 100, 100],
            'existing': [10, 11, 12, 20, 21, 22, 20, 20]
        })


@pytest.fixture(scope='function')
def agents_df():
    return pd.DataFrame(
        {
            'taz': [1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
            'district': ['a', 'b', 'b', 'b', 'b', 'b', 'c', 'c', 'c', 'c']
        },
        index=pd.Index(range(100, 110)))


@pytest.fixture(scope='function')
def locations_df():
    return pd.DataFrame(
        {
            'taz': [1, 1, 1, 2, 2, 2],
            'district': ['a', 'a', 'b', 'b', 'c', 'c'],
            'weight': [50, 0, 25, 10, 20, 30],
            'capacity': [1, 5, 5, 5, 5, 3],
            'existing': [1, 0, 0, 0, 0, 2]
        },
        index=pd.Index(range(300, 306)))


def assert_totals_match(amounts_df, amounts_col, results, target_col, year):
    curr_amount = amounts_df[amounts_col][year]
    assert curr_amount == results.sum()


def assert_totals_match_delta(amounts_df,
                              amounts_col,
                              orig_df,
                              results,
                              target_col,
                              year,
                              previous_year=9999):
    if previous_year == 9999:
        previous_year = year - 1
    amount = amounts_df[amounts_col][year]
    prev_amount = amounts_df[amounts_col][previous_year]
    expected_change = amount - prev_amount
    observed_change = results.sum() - orig_df[target_col].sum()
    assert expected_change == observed_change


def assert_segmentation_totals_match_delta(amounts_df,
                                           amounts_col,
                                           orig_df,
                                           results,
                                           target_col,
                                           segment_cols,
                                           year,
                                           previous_year=9999):
    # get the allocated change
    orig_sums = orig_df.groupby(segment_cols).sum()
    results_df = orig_df.copy()
    results_df[target_col] = results
    result_sums = results_df.groupby(segment_cols).sum()
    change_sums = result_sums - orig_sums

    # join the amounts for the two years
    if previous_year == 9999:
        previous_year = year - 1
    amount_join = pd.merge(
        left=amounts_df.loc[year],
        right=amounts_df.loc[previous_year],
        on=segment_cols,
        how='outer')

    # join the amounts w/ the observed change
    join2 = pd.merge(
        left=change_sums,
        right=amount_join,
        left_index=True,
        right_on=segment_cols,
        how='outer')

    amount_change = join2[amounts_col + '_x'] - join2[amounts_col + '_y']
    mismatches = len(join2[join2[target_col] != amount_change])
    assert mismatches == 0


def assert_capacities(results, orig_df, capacity_col, floor_col=None):
    c = orig_df[capacity_col]
    over = results > c
    if floor_col is not None:
        under = results < orig_df[floor_col]
    else:
        under = results < 0
    print results[over]
    assert not (over.any() or under.any())


def assert_agent_locations(agent_loc_ids, locations_df):
    assert agent_loc_ids.notnull().any
    assert np.in1d(agent_loc_ids, locations_df.index.values).all()


def test_null_data_frame(amounts_df, amounts_col, target_col):
    am = AllocationModel(amounts_df, amounts_col, target_col)
    assert am.allocate(None, 2010) is None


def test_noWeights_noCapacity(amounts_df, amounts_col, rows_df, target_col):
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
    assert_totals_match_delta(
        amounts_df, amounts_col, rows_df, results, target_col, year)


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
    assert_totals_match_delta(
        amounts_df, amounts_col, rows_df, results, target_col, year)


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
    assert_totals_match_delta(
        amounts_df, amounts_col, rows_df, results, target_col, year)
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
    results = amod.allocate(seg_rows_df, year)
    assert_segmentation_totals_match_delta(
        seg_amounts_df, amounts_col, seg_rows_df, results, target_col, segment_cols, year)


def test_missing_year(amounts_df, amounts_col, rows_df, target_col):
    year = 2020
    am = AllocationModel(amounts_df, amounts_col, target_col)
    orig_cnt = len(rows_df)
    orig_sum = rows_df[target_col].sum()
    results = am.allocate(rows_df, year)
    assert orig_cnt == len(results)
    assert orig_sum == results.sum()


def test_missing_year_delta(rows_df, target_col):
    # set up the allocation model
    amounts_col = 'amount'
    amounts_df = pd.DataFrame(
        {amounts_col: [150, 250, 300]},
        index=[2010, 2011, 2020])
    am = AllocationModel(
        amounts_df,
        amounts_col,
        target_col,
        as_delta=True,
        compute_delta=True)

    # get 2011 results to serve as baseline
    results_2011 = am.allocate(rows_df, 2011)
    rows_df[target_col] = results_2011

    # test 2020
    results_2020 = am.allocate(rows_df, 2020)
    assert_totals_match_delta(
        amounts_df, amounts_col, rows_df, results_2020, target_col, 2020, 2011)


def test_missing_year_delta_segmentation():
    # set up the allocation model
    amounts_df = pd.DataFrame(
        {
            'taz': [1, 2, 1, 2, 1, 2],
            'amount': [100, 200, 150, 250, 300, 500]
        },
        index=pd.Index([2010, 2010, 2011, 2011, 2020, 2020]))
    rows_df = pd.DataFrame(
        {
            'taz': [1, 1, 1, 2, 2, 2, 1, 1],
            'weight': [50, 0, 25, 10, 20, 30, 10, 0],
            'existing': [10, 11, 12, 20, 21, 22, 20, 20]
        })
    am = AllocationModel(
        amounts_df,
        'amount',
        'existing',
        'weight',
        as_delta=True,
        compute_delta=True,
        segment_cols=['taz'])

    # get 2011 results to serve as baseline
    results_2011 = am.allocate(rows_df, 2011)
    rows_df[target_col] = results_2011

    # test 2020
    results_2020 = am.allocate(rows_df, 2020)
    assert_segmentation_totals_match_delta(
        amounts_df, 'amount', rows_df, results_2020, 'existing', ['taz'], 2020, 2011)


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


def test_not_enough_capacity_decline(amounts_df,
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


def test_raise_float_amount(amounts_df, amounts_col, rows_df, target_col):
    year = 2010
    amounts_df[amounts_col] = pd.Series([100.56, 200.56, 300.56])
    with pytest.raises(ValueError):
        am = AllocationModel(amounts_df, amounts_col, target_col)


def test_agent_allocation(agents_df, locations_df):
    aa = AgentAllocationModel('existing', 'weight', 'capacity')
    loc_ids, loc_allo = aa.locate_agents(locations_df, agents_df, 2010)
    assert_agent_locations(loc_ids, locations_df)
    assert_capacities(loc_allo, locations_df, 'capacity')


def test_agent_allocation_segmented(agents_df, locations_df):
    segment_cols = ['taz', 'district']
    aa = AgentAllocationModel(
        'existing',
        'weight',
        'capacity',
        segment_cols=segment_cols)
    loc_ids, loc_allo = aa.locate_agents(locations_df, agents_df, 2010)
    assert_agent_locations(loc_ids, locations_df)
    assert_capacities(loc_allo, locations_df, 'capacity')

    agents_df['loc_id'] = loc_ids
    m = pd.merge(agents_df, locations_df, left_on='loc_id', right_index=True)
    for curr_seg_col in segment_cols:
        assert not (m[curr_seg_col + '_x'] != m[curr_seg_col + '_y']).any()


def test_agent_allocation_not_enough_capacity():
    locations_df = pd.DataFrame({'existing': [0], 'capacity': [8]})
    agents_df = pd.DataFrame(np.arange(10), columns=['test'])
    aa = AgentAllocationModel('existing', capacity_col='capacity')
    loc_ids, loc_allo = aa.locate_agents(locations_df, agents_df, 2010)
    assert len(loc_ids[loc_ids.isnull()]) == 2
    assert loc_allo.sum() == 8
