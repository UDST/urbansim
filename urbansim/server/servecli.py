from __future__ import print_function

import argparse
import os.path
import sys

from urbansim.server import urbansimd


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description=('Start the UrbanSim server for use with the UrbanSim web portal'))
    parser.add_argument(
        'dataset', help='HDF5 file which contains UrbanSim data.')
    return parser.parse_args(args)


def main(args=None):
    args = parse_args(args)

    sys.path.insert(0, ".")
    import dataset
    dset = dataset.LocalDataset(args.dataset)

    urbansimd.start_service(dset)
