import os.path
import sys
import time
import urllib2
import urlparse
from collections import defaultdict

import simplejson as json
import yaml
from jinja2 import Environment, FileSystemLoader

# these are the lists of modes available for each model
MODES_D = defaultdict(lambda: ["estimate", "simulate"], {
    "minimodel": ["run"],
    "modelset": ["run"],
    "transitionmodel": ["run"],
    "transitionmodel2": ["run"],
    "networks": ["run"]
})


def droptable(d):
    d = d.copy()
    del d['table']
    return d

J2_ENV = Environment(
    loader=FileSystemLoader(
        os.path.join(os.path.dirname(__file__), 'templates')),
    trim_blocks=True)
J2_ENV.filters['droptable'] = droptable


def load_config(config):
    """
    Load a configuration from a JSON or YAML file.

    Parameters
    ----------
    config : str
        Path to config file.

    Returns
    -------
    conf : dict
        Configuration parameters as a dictionary.
    basename : str
        Base name (without path) of the config file.

    """
    base = os.path.basename(config)
    ext = os.path.splitext(base)[1]

    if ext == '.json':
        loader = json.load
    elif ext in {'.yaml', '.yml'}:
        loader = yaml.load
    else:
        raise ValueError('Only JSON and YAML configs are supported.')

    with open(config) as f:
        conf = loader(f)

    return conf, base


def gen_model(config, configname=None, mode=None):
    """
    Generate a Python model based on a configuration stored in a JSON file.

    Parameters
    ----------
    config : dict
        Dictionary of config parameters.
    configname : str, optional
        Name of configuration file from which config came,
        if it came from a file.
    mode : str, optional

    Returns
    -------
    basename : str
    d : dict

    """
    configname = configname or 'autorun'

    if 'model' not in config:
        print('Not generating {}'.format(configname))
        return '', {}

    model = config['model']
    d = {}
    modes = [mode] if mode else MODES_D[model]
    for mode in modes:
        assert mode in MODES_D[model]

        basename = os.path.splitext(configname)[0]
        dirname = os.path.dirname(configname)
        print('Running {} with mode {}'.format(basename, mode))

        if 'var_lib_file' in config:
            if 'var_lib_db' in config:
                # should not be hardcoded
                githubroot = ('https://raw.github.com/fscottfoti'
                              '/bayarea/master/configs/')
                var_lib = json.loads(
                    urllib2.urlopen(
                        urlparse.urljoin(
                            githubroot, config['var_lib_file'])).read())
            else:
                with open(
                    os.path.join(configs_dir(), config['var_lib_file'])
                ) as f:
                    var_lib = json.load(f)

            config['var_lib'] = config.get('var_lib', {})
            config['var_lib'].update(var_lib)

        config['modelname'] = basename
        config['template_mode'] = mode
        d[mode] = J2_ENV.get_template(model + '.py.template').render(**config)

    return basename, d

COMPILED_MODELS = {}


def run_model(config, dset, configname=None, mode="estimate"):
    basename, model = gen_model(config, configname, mode)
    model = model[mode]
    code = compile(model, '<string>', 'exec')
    ns = {}
    exec code in ns
    print(basename, mode)
    out = ns['%s_%s' % (basename, mode)](dset, 2010)
    return out
