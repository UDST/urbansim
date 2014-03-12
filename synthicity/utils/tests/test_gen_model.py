"""
Tests for the functionality that converts JSON configurations to
Python model files.

"""

from ..misc import gen_model


def test_hedonicmodel():
    config = {
        'model': 'hedonicmodel',
        'name': 'unit_test',
        'show': True,
        'internalname': 'test_name',
        'table': 'dset.test_table',
        "table_sim": "dset.test_sim(thing=False)",
        'merge': {
            'right_index': True,
            'left_on': '_node_id',
            'table': 'dset.merge'
        },
        'segment': ['test_segment'],
        'dep_var': 'test_dep_var',
        'dep_var_transform': 'np.log',
        'ind_vars': [
            'var1',
            'var2',
            'var3'
        ],
        'add_constant': True,
        'output_transform': 'np.exp',
        'output_varname': 'test_output_var',
        'output_table': 'dset.output_table'
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
