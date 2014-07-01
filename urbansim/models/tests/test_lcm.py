import numpy.testing as npt
import pandas as pd
import pytest
import yaml
from pandas.util import testing as pdt

from ...utils import testing

from .. import lcm


@pytest.fixture
def choosers():
    return pd.DataFrame(
        {'var1': range(5, 10),
         'thing_id': ['a', 'c', 'e', 'g', 'i']})


@pytest.fixture
def grouped_choosers(choosers):
    choosers['group'] = ['x', 'y', 'x', 'x', 'y']
    return choosers


@pytest.fixture
def alternatives():
    return pd.DataFrame(
        {'var2': range(10, 20),
         'var3': range(20, 30)},
        index=pd.Index([x for x in 'abcdefghij'], name='thing_id'))


def test_unit_choice_uniform(choosers, alternatives):
    probabilities = [1] * len(alternatives)
    choices = lcm.unit_choice(
        choosers.index, alternatives.index, probabilities)
    npt.assert_array_equal(choices.index, choosers.index)
    assert choices.isin(alternatives.index).all()


def test_unit_choice_some_zero(choosers, alternatives):
    probabilities = [0, 1, 0, 1, 1, 0, 1, 0, 0, 1]
    choices = lcm.unit_choice(
        choosers.index, alternatives.index, probabilities)
    npt.assert_array_equal(choices.index, choosers.index)
    npt.assert_array_equal(sorted(choices.values), ['b', 'd', 'e', 'g', 'j'])


def test_unit_choice_not_enough(choosers, alternatives):
    probabilities = [0, 0, 0, 0, 0, 1, 0, 1, 0, 0]
    choices = lcm.unit_choice(
        choosers.index, alternatives.index, probabilities)
    npt.assert_array_equal(choices.index, choosers.index)
    assert choices.isnull().sum() == 3
    npt.assert_array_equal(sorted(choices[~choices.isnull()]), ['f', 'h'])


def test_unit_choice_none_available(choosers, alternatives):
    probabilities = [0] * len(alternatives)
    choices = lcm.unit_choice(
        choosers.index, alternatives.index, probabilities)
    npt.assert_array_equal(choices.index, choosers.index)
    assert choices.isnull().all()


def test_mnl_lcm(choosers, alternatives):
    model_exp = 'var2 + var1:var3'
    sample_size = 5
    choosers_fit_filters = ['var1 != 5']
    choosers_predict_filters = ['var1 != 7']
    alts_fit_filters = ['var3 != 15']
    alts_predict_filters = ['var2 != 14']
    interaction_predict_filters = None
    estimation_sample_size = None
    choice_column = None
    name = 'Test LCM'

    model = lcm.MNLLocationChoiceModel(
        model_exp, sample_size,
        choosers_fit_filters, choosers_predict_filters,
        alts_fit_filters, alts_predict_filters,
        interaction_predict_filters, estimation_sample_size,
        choice_column, name)
    loglik = model.fit(choosers, alternatives, choosers.thing_id)
    model.report_fit()

    # hard to test things exactly because there's some randomness
    # involved, but can at least do a smoke test.
    assert len(loglik) == 3
    assert len(model.fit_parameters) == 2
    assert len(model.fit_parameters.columns) == 3

    choices = model.predict(choosers.iloc[1:], alternatives)

    pdt.assert_index_equal(choices.index, pd.Index([1, 3, 4]))
    assert choices.isin(alternatives.index).all()

    # check that we can do a YAML round-trip
    yaml_str = model.to_yaml()
    new_model = lcm.MNLLocationChoiceModel.from_yaml(yaml_str)

    assert new_model.fitted
    testing.assert_frames_equal(model.fit_parameters, new_model.fit_parameters)


def test_mnl_lcm_repeated_alts(choosers, alternatives):
    model_exp = 'var2 + var1:var3'
    sample_size = 5
    choosers_fit_filters = ['var1 != 5']
    choosers_predict_filters = ['var1 != 7']
    alts_fit_filters = ['var3 != 15']
    alts_predict_filters = ['var2 != 14']
    interaction_predict_filters = ['var1 * var2 > 50']
    estimation_sample_size = None
    choice_column = 'thing_id'
    name = 'Test LCM'

    model = lcm.MNLLocationChoiceModel(
        model_exp, sample_size,
        choosers_fit_filters, choosers_predict_filters,
        alts_fit_filters, alts_predict_filters,
        interaction_predict_filters, estimation_sample_size,
        choice_column, name)
    loglik = model.fit(choosers, alternatives, 'thing_id')
    model.report_fit()

    # hard to test things exactly because there's some randomness
    # involved, but can at least do a smoke test.
    assert len(loglik) == 3
    assert len(model.fit_parameters) == 2
    assert len(model.fit_parameters.columns) == 3

    repeated_index = alternatives.index.repeat([1, 2, 3, 2, 4, 3, 2, 1, 5, 8])
    repeated_alts = alternatives.loc[repeated_index].reset_index()

    choices = model.predict(choosers, repeated_alts)

    pdt.assert_index_equal(choices.index, pd.Index([0, 1, 3, 4]))
    assert choices.isin(alternatives.index).all()


def test_mnl_lcm_group(grouped_choosers, alternatives):
    model_exp = 'var2 + var1:var3'
    sample_size = 4

    group = lcm.MNLLocationChoiceModelGroup('group')
    group.add_model_from_params('x', model_exp, sample_size)
    group.add_model_from_params('y', model_exp, sample_size)

    assert group.fitted is False
    logliks = group.fit(grouped_choosers, alternatives, 'thing_id')
    assert group.fitted is True

    assert 'x' in logliks and 'y' in logliks
    assert isinstance(logliks['x'], dict) and isinstance(logliks['y'], dict)

    choices = group.predict(grouped_choosers, alternatives)

    assert len(choices.unique()) == len(choices)
    assert choices.isin(alternatives.index).all()


def test_mnl_lcm_segmented_raises():
    group = lcm.SegmentedMNLLocationChoiceModel('group', 2)

    with pytest.raises(ValueError):
        group.add_segment('x')


def test_mnl_lcm_segmented(grouped_choosers, alternatives):
    model_exp = 'var2 + var1:var3'
    sample_size = 4

    group = lcm.SegmentedMNLLocationChoiceModel(
        'group', sample_size, default_model_expr=model_exp)
    group.add_segment('x')
    group.add_segment('y', 'var3 + var1:var2')

    assert group.fitted is False
    logliks = group.fit(grouped_choosers, alternatives, 'thing_id')
    assert group.fitted is True

    assert 'x' in logliks and 'y' in logliks
    assert isinstance(logliks['x'], dict) and isinstance(logliks['y'], dict)

    choices = group.predict(grouped_choosers, alternatives)

    assert len(choices.unique()) == len(choices)
    assert choices.isin(alternatives.index).all()


def test_mnl_lcm_segmented_yaml(grouped_choosers, alternatives):
    model_exp = 'var2 + var1:var3'
    sample_size = 4

    group = lcm.SegmentedMNLLocationChoiceModel(
        'group', sample_size, default_model_expr=model_exp, name='test_seg')
    group.add_segment('x')
    group.add_segment('y', 'var3 + var1:var2')

    expected_dict = {
        'model_type': 'segmented_locationchoice',
        'name': 'test_seg',
        'segmentation_col': 'group',
        'sample_size': sample_size,
        'choosers_fit_filters': None,
        'choosers_predict_filters': None,
        'alts_fit_filters': None,
        'alts_predict_filters': None,
        'interaction_predict_filters': None,
        'estimation_sample_size': None,
        'choice_column': None,
        'default_config': {
            'model_expression': model_exp,
        },
        'fitted': False,
        'models': {
            'x': {
                'name': 'x',
                'fitted': False,
                'log_likelihoods': None,
                'fit_parameters': None
            },
            'y': {
                'name': 'y',
                'model_expression': 'var3 + var1:var2',
                'fitted': False,
                'log_likelihoods': None,
                'fit_parameters': None
            }
        }
    }

    assert yaml.load(group.to_yaml()) == expected_dict

    new_seg = lcm.SegmentedMNLLocationChoiceModel.from_yaml(group.to_yaml())
    assert yaml.load(new_seg.to_yaml()) == expected_dict

    group.fit(grouped_choosers, alternatives, 'thing_id')

    expected_dict['fitted'] = True
    expected_dict['models']['x']['fitted'] = True
    expected_dict['models']['y']['fitted'] = True
    del expected_dict['models']['x']['fit_parameters']
    del expected_dict['models']['x']['log_likelihoods']
    del expected_dict['models']['y']['fit_parameters']
    del expected_dict['models']['y']['log_likelihoods']

    actual_dict = yaml.load(group.to_yaml())
    assert isinstance(actual_dict['models']['x'].pop('fit_parameters'), dict)
    assert isinstance(actual_dict['models']['x'].pop('log_likelihoods'), dict)
    assert isinstance(actual_dict['models']['y'].pop('fit_parameters'), dict)
    assert isinstance(actual_dict['models']['y'].pop('log_likelihoods'), dict)

    assert actual_dict == expected_dict

    new_seg = lcm.SegmentedMNLLocationChoiceModel.from_yaml(group.to_yaml())
    assert new_seg.fitted is True


def test_segmented_lcm_removes_old_models(grouped_choosers, alternatives):
    model_exp = 'var2 + var1:var3'
    sample_size = 4

    group = lcm.SegmentedMNLLocationChoiceModel(
        'group', sample_size, default_model_expr=model_exp)
    group.add_segment('a')
    group.add_segment('b')
    group.add_segment('c')

    group.fit(grouped_choosers, alternatives, 'thing_id')

    assert sorted(group._group.models.keys()) == ['x', 'y']
