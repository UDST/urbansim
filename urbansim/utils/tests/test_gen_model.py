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
            'dset.variables.compute_nonres_building_proportions(dset, year))'
        ),
        'dep_var': '_node_id',
        'filters': ['households.tenure == 1'],
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
        'merge': {
            'table': 'dset.nodes',
            'left_on': '_node_id',
            'right_index': True
        },
        'model': 'locationchoicemodel',
        'modeldependencies': 'nrh.json',
        'output_table': 'dset.nets',
        'output_varname': 'firms_building_ids',
        'patsy': (
            'np.log1p(unit_sqft) + sum_residential_units + '
            'ave_unit_sqft + ave_lot_sqft + ave_income + poor + sfdu + '
            'renters + np.log1p(res_rent) - 1'
        ),
        'relocation_rate': 0.04,
        'sample_size': 100,
        'segment': ['naics11cat'],
        'supply_constraint': (
            'dset.building_filter(residential=1).'
            "residential_units.sub(dset.households.groupby('building_id')."
            'size(), fill_value=0)'
        ),
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


def test_modelset():
    config = {
        'dataset': 'BayAreaDataset',
        'datastore': 'bayarea.h5',
        'model': 'modelset',
        'modelstorun': [
            'networks_run',
            'rrh_estimate',
            'rrh_simulate',
            'hlcmr_estimate'
        ],
        'pathinsertcwd': True
    }

    basename, d = gen_model(config)

    assert basename == 'autorun'
    assert 'run' in d

    # make sure no errors happen when compiling the code
    compile(d['run'], '<string>', mode='exec')


def test_networks():
    config = {
        'ind_vars': [
            'sum_residential_units',
            'ave_unit_sqft',
            'ave_lot_sqft',
            'ave_income',
            'population',
            'poor',
            'hhsize',
            'jobs',
            'sfdu',
            'renters'
        ],
        'model': 'networks',
        'networks': {
            'factors': [1.0],
            'filenames': ['osm_bayarea.jar'],
            'maxdistances': [2000],
            'twoway': [1]
        },
        'show': True,
        'update_xys': ['dset.households'],
        'var_lib': {
            'ave_income': 'ave_income_var',
            'ave_lot_sqft': 'ave_lot_sqft_var',
            'ave_unit_sqft': 'ave_unit_sqft_var',
            'hhsize': 'hhsize_var',
            'jobs': 'jobs_var',
            'poor': 'poor_var',
            'population': 'population_var',
            'renters': 'renters_var',
            'sfdu': 'sfdu_var',
            'sum_residential_units': 'sum_residential_units_var'
        },
        'writetocsv': 'nodes.csv',
        'writetotmp': 'nodes'
    }

    check_run(*gen_model(config))


def test_transition_model():
    config = {
        'growth_rate': 0.05,
        'internalname': 'households',
        'model': 'transitionmodel2',
        'output_varname': 'household_id',
        'table': 'dset.households',
        'zero_out_names': ['building_id']
    }

    check_run(*gen_model(config))
