import os.path
import shutil
import tempfile

import simplejson as json
import yaml

from .. import modelcompile

TEST_CONFIG = {
    'growth_rate': 0.05,
    'internalname': 'households',
    'model': 'transitionmodel2',
    'output_varname': 'household_id',
    'table': 'dset.households',
    'zero_out_names': ['building_id']
}


def setup_module(module):
    module.TEST_DIR = tempfile.mkdtemp()


def teardown_module(module):
    shutil.rmtree(module.TEST_DIR)


def test_load_config_json():
    test_file = os.path.join(TEST_DIR, 'test_config.json')

    with open(test_file, 'w') as f:
        json.dump(TEST_CONFIG, f)

    config, basename = modelcompile.load_config(test_file)
    assert config == TEST_CONFIG
    assert basename == 'test_config.json'


def test_load_config_yaml():
    test_file = os.path.join(TEST_DIR, 'test_config.yaml')

    with open(test_file, 'w') as f:
        yaml.dump(
            TEST_CONFIG, f, default_flow_style=False, indent=4, width=50)

    config, basename = modelcompile.load_config(test_file)
    assert config == TEST_CONFIG
    assert basename == 'test_config.yaml'
