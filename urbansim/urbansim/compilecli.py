from __future__ import print_function

import argparse
import os.path

from cliff.command import Command

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


class Compile(Command):
    def get_description(self):
        return ('Compile a set of Python files that run models '
                'specified in configuration files.')

    def get_parser(self, prog_name):
        parser = argparse.ArgumentParser(
            description=self.get_description(),
            prog=prog_name)
        parser.add_argument(
            'configs', nargs='+', help='Model configuration files.')
        return parser

    def take_action(self, args):
        for c in args.configs:
            model_save(c)
