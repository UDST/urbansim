"""
Tests for the functionality that converts JSON configurations to
Python model files.

"""

from ..misc import gen_model


def test_hedonicmodel():
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

    basename, d = gen_model(config)

    assert basename == 'autorun'
    assert 'estimate' in d
    assert 'simulate' in d

    assert 'def autorun_estimate' in d['estimate']
    assert 'def autorun_simulate' in d['simulate']

    # make sure no errors happen when compiling the code
    compile(d['estimate'], '<string>', mode='exec')
    compile(d['simulate'], '<string>', mode='exec')


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
        'table_sim': 'dset.nets[dset.nets.lastmove>2007]',
    }

    basename, d = gen_model(config)

    assert basename == 'autorun'
    assert 'estimate' in d
    assert 'simulate' in d

    assert 'def autorun_estimate' in d['estimate']
    assert 'def autorun_simulate' in d['simulate']

    # make sure no errors happen when compiling the code
    compile(d['estimate'], '<string>', mode='exec')
    compile(d['simulate'], '<string>', mode='exec')
