__author__ = 'JMartinez'
import orca
import pandas as pd
import numpy as np


orca.add_injectable("store",pd.HDFStore('C:/Users/jmartinez/Documents/Projects/UrbanSim/data/drcog.h5', mode='r'))
#orca.add_injectable("store",pd.HDFStore('C:/urbansim/data/drcog.h5', mode='r'))
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
    #df.drop('zone_id', axis=1, inplace=True)
    return df

@orca.table('zones', cache=True)
def zones(store):
    df = store['zones']
    amenities = pd.read_csv('c:/users/jmartinez/documents/data/urbansim/regression/processed/amenities.csv',index_col=0)
    df = pd.merge(df, amenities, left_index=True, right_index=True)
    return df


@orca.table('establishments', cache=True)
def establishments(store):
    df = store['establishments']
    return df


@orca.table('hh_demand')
def hh_demand():
    return pd.read_csv('c:/urbansim_new/urbansim/urbansim_drcog/config/hh_demand.csv', index_col=0)

@orca.table('emp_demand')
def emp_demand():
    return pd.read_csv('c:/urbansim_new/urbansim/urbansim_drcog/config/emp_demand.csv', index_col=0)

#this would go in the assumptions.py file
@orca.table('sqft_per_job', cache=True)
def sqft_per_job():
    df = pd.read_csv('c:/urbansim/data/building_sqft_per_job.csv', index_col=[0,1])
    return df

@orca.table('households_for_estimation', cache=True)
def households_for_estimation(store):
    return store['households_for_estimation']

@orca.table('counties', cache=True)
def counties():
    return pd.read_csv('C:/urbansim/data/TAZ_County_Table.csv').set_index('zone_id')

@orca.table('household_relocation_rates')
def household_relocation_rates(store):
    df = store['annual_household_relocation_rates']
    df.probability_of_relocating = df.probability_of_relocating
    return df

@orca.table('job_relocation_rates')
def job_relocation_rates(store):
    df = store['annual_job_relocation_rates'] / 10
    return df

@orca.table('household_control_totals')
def household_control_totals():
    df = pd.read_csv('C:/Users/jmartinez/Documents/Projects/UrbanSim/Results/emp_sector/hh.csv', index_col=0)
    return df

@orca.table('employment_control_totals')
def employment_control_totals():
    df = pd.read_csv('C:/Users/jmartinez/Documents/Projects/UrbanSim/Results/emp_sector/employment_control_totals.csv', index_col=0)
    return df

@orca.table('travel_data', cache=True)
def travel_data(store):
    df = store['travel_data']
    return df

@orca.table('migration_data')
def migration_data():
    df = pd.read_csv('c:/urbansim/data/NetMigrationByAge.csv')
    df.columns = ['county', 'age','net_migration']
    df = df[15:90]
    return df


@orca.table('zoning')
def zoning(store):
    return store.zoning

@orca.table('zoning_baseline')
def zoning_baseline(store):
    return pd.merge(store.parcels, store.zoning, left_on='zoning_id', right_index=True)

@orca.table('fars')
def fars(store):
    return store.fars

@orca.table('refiner_targets')
def refiner_targets():
    return pd.read_csv('c:/urbansim/data/zone_demand_refine.csv', index_col=0)


@orca.table("emp_multipliers")
def multipliers():
    return pd.read_csv('c:/urbansim_new/urbansim/urbansim_drcog/config/emp_multipliers.csv', index_col=0)

@orca.table("hh_multipliers")
def multipliers():
    return pd.read_csv('c:/urbansim_new/urbansim/urbansim_drcog/config/hh_multipliers.csv', index_col=0)
#Travel data Table (to be computed before all simulations)

@orca.table('t_data_dist20', cache=True)
def dist30( travel_data):
    t_data=travel_data.to_frame(columns=['am_single_vehicle_to_work_travel_time']).reset_index(level=1)
    return t_data[['to_zone_id']][t_data.am_single_vehicle_to_work_travel_time<20]

@orca.table('t_data_dist30', cache=True)
def dist30( travel_data):
    t_data=travel_data.to_frame(columns=['am_single_vehicle_to_work_travel_time']).reset_index(level=1)
    return t_data[['to_zone_id']][t_data.am_single_vehicle_to_work_travel_time<30]

@orca.table('t_data_dist15', cache=True)
def dist30( travel_data):
    t_data=travel_data.to_frame(columns=['am_single_vehicle_to_work_travel_time']).reset_index(level=1)
    return t_data[['to_zone_id']][t_data.am_single_vehicle_to_work_travel_time<15]

@orca.table('t_data_dist45', cache=True)
def dist30( travel_data):
    t_data=travel_data.to_frame(columns=['am_single_vehicle_to_work_travel_time']).reset_index(level=1)
    return t_data[['to_zone_id']][t_data.am_single_vehicle_to_work_travel_time<45]


#broadcass

orca.broadcast('zones', 'parcels', cast_index=True, onto_on='zone_id')
orca.broadcast('parcels','buildings', cast_index=True, onto_on='parcel_id', onto_index=False)
orca.broadcast('buildings','households', cast_index=True, onto_on='building_id')
orca.broadcast('buildings', 'establishments', cast_index=True, onto_on ='building_id')
orca.broadcast('zoning', 'parcels', cast_index=True, onto_on='zoning_id')
orca.broadcast('fars', 'parcels', cast_index=True, onto_on='far_id')
orca.broadcast('counties','zones', cast_index=True, onto_index=True)
orca.broadcast('buildings','households_for_estimation', cast_index=True, onto_on='building_id')
orca.broadcast('counties', 'establishments', cast_index=True, onto_on='zone_id')
orca.broadcast('counties', 'households', cast_index=True, onto_on='zone_id')