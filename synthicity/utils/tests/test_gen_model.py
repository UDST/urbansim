"""
Tests for the functionality that converts JSON configurations to
Python model files.

"""

from ..misc import gen_model


def check_estimate_simulate(basename, d):
    """
    Check results for an estimate/simulate model.

    """
    assert basename == 'autorun'
    assert 'estimate' in d
    assert 'simulate' in d

    assert 'def autorun_estimate' in d['estimate']
    assert 'def autorun_simulate' in d['simulate']

    # make sure no errors happen when compiling the code
    compile(d['estimate'], '<string>', mode='exec')
    compile(d['simulate'], '<string>', mode='exec')


def check_run(basename, d):
    """
    Check results for a 'run' model.

    """
    assert basename == 'autorun'
    assert 'run' in d

    assert 'def autorun_run' in d['run']

    # make sure no errors happen when compiling the code
    compile(d['run'], '<string>', mode='exec')


def test_hedonic_model():
    config = {
        'add_constant': True,
        'dep_var': 'averageweightedrent',
        'dep_var_transform': 'np.log',
        'ind_vars': ['accessibility', 'reliability', 'ln_stories'],
        'internalname': 'jobs',
        'merge': {
            'left_on': '_node_id',
            'right_index': True,
            'table': 'dset.nodes'
        },
        'model': 'hedonicmodel',
        'output_table': 'dset.buildings',
        'output_transform': 'np.exp',
        'output_varname': 'nonresidential_rent',
        'segment': ['general_type'],
        'table': 'dset.costar',
        'table_sim': 'dset.building_filter(residential=0)',
    }

    check_estimate_simulate(*gen_model(config))


def test_location_choice_model():
    config = {
        'alternatives': (
            'dset.nodes.join('
            'dset.variables.compute_nonres_building_proportions(dset, year))'),
        'dep_var': '_node_id',
        'est_sample_size': 10000,
        'ind_vars': [
            'total sqft',
            'ln_weighted_rent',
            'retpct',
            'indpct',
            'accessibility',
            'reliability'
        ],
        'internalname': 'jobs',
        'model': 'locationchoicemodel',
        'modeldependencies': 'nrh.json',
        'output_table': 'dset.nets',
        'output_varname': 'firms_building_ids',
        'sample_size': 100,
        'segment': ['naics11cat'],
        'table': 'dset.nets',
        'table_sim': 'dset.nets[dset.nets.lastmove > 2007]',
    }

    check_estimate_simulate(*gen_model(config))


def test_mini_model():
    config = {
        'add_constant': True,
        'ind_vars': ['var1', 'var2', 'var3'],
        'internalname': 'test',
        'model': 'minimodel',
        'output_table': 'dset.test_output',
        'output_varname': 'test_output_var',
        'table': 'dest.test_table',
        'table_sim': 'dest.test_table[dest.test_table > 9000]'
    }

    check_run(*gen_model(config))
