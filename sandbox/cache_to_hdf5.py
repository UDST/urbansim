import os
import shutil
import time

import pandas as pd
import numpy as np


def cache_to_df(dir_path):
    table = {}
    for attrib in os.listdir(dir_path):
        attribdir = os.path.join(dir_path, attrib)
        if attrib.endswith('lf8'):
            attrib_data = np.fromfile(attribdir, 'float64')
            table[attrib[:-4]] = attrib_data
        elif attrib.endswith('lf4'):
            attrib_data = np.fromfile(attribdir, 'float32')
            table[attrib[:-4]] = attrib_data
        elif attrib.endswith('li2'):
            attrib_data = np.fromfile(attribdir, 'int16')
            table[attrib[:-4]] = attrib_data
        elif attrib.endswith('li4'):
            attrib_data = np.fromfile(attribdir, 'int32')
            table[attrib[:-4]] = attrib_data
        elif attrib.endswith('li8'):
            attrib_data = np.fromfile(attribdir, 'int64')
            table[attrib[:-4]] = attrib_data
        elif attrib.find('.iS') > 0:
            start_suffix = attrib.find('.iS')
            length_string = int(attrib[(start_suffix + 3):])
            attrib_data = np.fromfile(attribdir, ('a' + str(length_string)))
            table[attrib[:start_suffix]] = attrib_data
        elif attrib.endswith('.ib1'):
            attrib_data = np.fromfile(attribdir, 'bool')
            table[attrib[:-4]] = attrib_data
        else:
            print 'Array %s is not a recognized data type' % (attrib,)
    df = pd.DataFrame(table)
    return df

store = pd.HDFStore('mrcog.h5')
for dirname in os.listdir('base_year_data/2010/'):
    if dirname not in {'parcels', 'buildings', 'households', 'jobs', 'zones',
                       'travel_data', 'annual_employment_control_totals',
                       'annual_household_control_totals',
                       'annual_household_relocation_rates',
                       'annual_job_relocation_rates', 'building_sqft_per_job',
                       'building_types', 'counties', 'target_vacancies',
                       'development_event_history'}:
        continue
    table_path = 'base_year_data/2010/' + dirname
    print dirname
    df = cache_to_df(table_path)
    keys = [dirname[:-1] + "_id"]
    if dirname == "travel_data":
        keys = ["from_zone_id", "to_zone_id"]
    if dirname == "annual_employment_control_totals":
        keys = ["sector_id", "year", "home_based_status"]
    if dirname == "annual_job_relocation_rates":
        keys = ["sector_id"]
    if dirname == "annual_household_control_totals":
        keys = ["year"]
    if dirname == "annual_household_relocation_rates":
        keys = ["age_of_head_max", "age_of_head_min",
                "income_min", "income_max"]
    if dirname == "building_sqft_per_job":
        keys = ["zone_id", "building_type_id"]
    if dirname == "counties":
        keys = ["county_id"]
    if dirname == "development_event_history":
        keys = ["building_id"]
    if dirname == "target_vacancies":
        keys = ["building_type_id", "year"]
    if dirname != "annual_household_relocation_rates":
        df = df.set_index(keys)

    tbl = df
    print df
    newtbl = pd.DataFrame(index=tbl.index)
    for colname in df.columns:
        if tbl[colname].dtype == np.float64:
            newtbl[colname] = tbl[colname].astype('float32')
        elif tbl[colname].dtype == np.int64:
            newtbl[colname] = tbl[colname].astype('int32')
        else:
            newtbl[colname] = tbl[colname]

    df = newtbl
    print df
    store.put(dirname, df)
