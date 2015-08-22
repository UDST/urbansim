__author__ = 'JMartinez'
import orca
import pandas as pd
import numpy as np


######This module registers the necessary data tables into the orca pipeline#######

#Register the data source (drcog.h5) with the orca pipeline
orca.add_injectable("store",pd.HDFStore('c:/urbansim/data/Justin Data/test data/drcog.h5', mode='r'))

#register tables
@orca.table('buildings', cache=True)
def buildings(store):
    df = store['buildings']
    return df

@orca.table('parcels', cache=True)
def parcels(store):
    df = store['parcels']
    return df

@orca.table('households', cache=True)
def households(store):
    df = store['households']
    return df

@orca.table('zones', cache=True)
def zones(store):
    df = store['zones']
    return df

@orca.table('travel_data', cache=True)
def travel_data(store):
    df = store['travel_data']
    return df

@orca.table('establishments', cache=True)
def establishments(store):
    df = store['establishments']
    return df

#this would go in the assumptions.py file
@orca.table('sqft_per_job', cache=True)
def sqft_per_job(store):
    df = store['building_sqft_per_job']
    return df

@orca.table('buildings_merged', cache=False)
def main_table():
     return orca.merge_tables('buildings', tables=['buildings','parcels','zones'])

@orca.table('households_for_estimation', cache=True)
def households_for_estimation(store):
    return store['households_for_estimation']

@orca.table('zone_to_county', cache=True)
def zone_to_county():
    return pd.read_csv('C:/urbansim/data/TAZ_County_Table.csv').set_index('zone_id')


#automagic merging
orca.broadcast('zones', 'parcels', cast_index=True, onto_on='zone_id')
orca.broadcast('zones', 'households', cast_index=True, onto_on='zone_id')
orca.broadcast('parcels', 'buildings', onto_on='parcel_id', cast_index=True, onto_index=False)
orca.broadcast('zones', 'establishments', cast_index=True, onto_on='zone_id')
