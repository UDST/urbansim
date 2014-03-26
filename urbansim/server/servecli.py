from __future__ import print_function

import argparse
import sys

from cliff.command import Command

from urbansim.server import urbansimd


class Serve(Command):
    def get_description(self):
        return ('Start the UrbanSim server for use '
                'with the UrbanSim web portal')

    def get_parser(self, prog_name):
        parser = argparse.ArgumentParser(
            description=self.get_description(),
            prog=prog_name)
        parser.add_argument(
            'dataset', help='HDF5 file which contains UrbanSim data.')
        return parser

    def take_action(self, args):
        sys.path.insert(0, ".")
        import dataset
        dset = dataset.LocalDataset(args.dataset)

        urbansimd.start_service(dset)
