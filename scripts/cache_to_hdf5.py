#!/usr/bin/env python

from __future__ import print_function

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd


def cache_to_df(dir_path):
    """
    Convert a directory of binary array data files to a Pandas DataFrame.

    Parameters
    ----------
    dir_path : str


    """
    table = {}
    for attrib in glob.glob(os.path.join(dir_path, '*')):
        attrib_name, attrib_ext = os.path.splitext(os.path.basename(attrib))
        if attrib_ext == '.lf8':
            attrib_data = np.fromfile(attrib, np.float64)
            table[attrib_name] = attrib_data

        elif attrib_ext == '.lf4':
            attrib_data = np.fromfile(attrib, np.float32)
            table[attrib_name] = attrib_data

        elif attrib_ext == '.li2':
            attrib_data = np.fromfile(attrib, np.int16)
            table[attrib_name] = attrib_data

        elif attrib_ext == '.li4':
            attrib_data = np.fromfile(attrib, np.int32)
            table[attrib_name] = attrib_data

        elif attrib_ext == '.li8':
            attrib_data = np.fromfile(attrib, np.int64)
            table[attrib_name] = attrib_data

        elif attrib_ext == '.ib1':
            attrib_data = np.fromfile(attrib, np.bool_)
            table[attrib_name] = attrib_data

        elif attrib_ext.startswith('.iS'):
            length_string = int(attrib_ext[3:])
            attrib_data = np.fromfile(attrib, ('a' + str(length_string)))
            table[attrib_name] = attrib_data

        else:
            print('Array {} is not a recognized data type'.format(attrib))

    df = pd.DataFrame(table)
    return df


DIRECTORIES = {
    'parcels', 'buildings', 'households', 'jobs', 'zones', 'travel_data',
    'annual_employment_control_totals', 'annual_household_control_totals',
    'annual_household_relocation_rates', 'annual_job_relocation_rates',
    'building_sqft_per_job', 'building_types', 'counties', 'target_vacancies',
    'development_event_history'
}


def convert_dirs(base_dir, hdf_name, complib=None, complevel=0):
    """
    Convert nested set of directories to

    """
    print('Converting directories in {}'.format(base_dir))

    dirs = glob.glob(os.path.join(base_dir, '*'))
    dirs = {d for d in dirs if os.path.basename(d) in DIRECTORIES}
    if not dirs:
        raise RuntimeError('No direcotries found matching known data.')

    store = pd.HDFStore(
        hdf_name, mode='w', complevel=complevel, complib=complib)

    for dirpath in dirs:
        dirname = os.path.basename(dirpath)

        print(dirname)
        df = cache_to_df(dirpath)

        if dirname == 'travel_data':
            keys = ['from_zone_id', 'to_zone_id']
        elif dirname == 'annual_employment_control_totals':
            keys = ['sector_id', 'year', 'home_based_status']
        elif dirname == 'annual_job_relocation_rates':
            keys = ['sector_id']
        elif dirname == 'annual_household_control_totals':
            keys = ['year']
        elif dirname == 'annual_household_relocation_rates':
            keys = ['age_of_head_max', 'age_of_head_min',
                    'income_min', 'income_max']
        elif dirname == 'building_sqft_per_job':
            keys = ['zone_id', 'building_type_id']
        elif dirname == 'counties':
            keys = ['county_id']
        elif dirname == 'development_event_history':
            keys = ['building_id']
        elif dirname == 'target_vacancies':
            keys = ['building_type_id', 'year']
        else:
            keys = [dirname[:-1] + '_id']

        if dirname != 'annual_household_relocation_rates':
            df = df.set_index(keys)

        for colname in df.columns:
            if df[colname].dtype == np.float64:
                df[colname] = df[colname].astype(np.float32)
            elif df[colname].dtype == np.int64:
                df[colname] = df[colname].astype(np.int32)
            else:
                df[colname] = df[colname]

        df.info()
        print(os.linesep)
        store.put(dirname, df)

    store.close()


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description=(
            'Convert nested set of directories containing binary '
            'array data to an HDF5 file made from Pandas DataFrames.'))
    parser.add_argument('base_dir', help='Base data directory for conversion.')
    parser.add_argument('hdf_name', help='Name of output HDF5 file.')
    parser.add_argument('-c', '--complib',
                        help=('Compression library to use, if any. '
                              "Can be one of "
                              "'zlib', 'bzip2', 'lzo', 'blosc'."))
    parser.add_argument('-l', '--complevel', type=int, default=0,
                        help='Compression level to use.')
    return parser.parse_args(args)


def main(args=None):
    args = parse_args(args)
    convert_dirs(args.base_dir, args.hdf_name, args.complib, args.complevel)


if __name__ == '__main__':
    sys.exit(main())
