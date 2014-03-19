from __future__ import print_function

import argparse
import os.path

from urbansim.urbansim import modelcompile
from urbansim.utils import misc


def model_save(config):
    print('Generating model for config {}.'.format(config))

    basename = None
    if isinstance(config, str):
        config, basename = modelcompile.load_config(config)

    basename, d = modelcompile.gen_model(config, configname=basename)

    for mode, code in d.items():
        outname = os.path.join(
            misc.models_dir(), '{}_{}.py'.format(basename, mode))

        print('Saving model {}.'.format(outname))

        with open(outname, 'w') as f:
            f.write(code)


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description=('Compile a set of Python files that run models '
                     'specified in JSON configuration files.'))
    parser.add_argument(
        'configs', nargs='+', help='JSON model configuration files.')
    return parser.parse_args(args)


def main(args=None):
    args = parse_args(args)

    for c in args.configs:
        model_save(c)
