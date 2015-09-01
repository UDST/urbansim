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

@orca.table('alts_hlcm', cache=True, cache_scope='iteration')
def alts_hlcm(buildings):
    columns = [
        'residential_units',
        'ln_dist_rail',
        'ln_avg_unit_price_zone',
        'median_age_of_head',
        'median_yearbuilt_post_1990',
        'percent_hh_with_child_x_hh_with_child',
        'percent_renter_hh_in_zone',
        'townhome',
        'multifamily',
        'jobs_within_45min',
        'income5xlt_x_avg_unit_price_zone',
        'median_yearbuilt_pre_1950',
        'ln_income_x_average_resunit_size',
        'wkrs_hhs_x_ln_jobs_within_30min',
        'mean_income',
        'cherry_creek_school_district',
        'percent_younghead_x_younghead',
        'ln_jobs_within_30min',
        'ln_emp_sector3_within_20min',
        'allpurpose_agglosum_floor',
    ]
    alts = buildings.to_frame(columns=columns)
    alts.fillna(0, inplace=True)
    return alts

@orca.table('alts_elcm', cache=True, cache_scope='iteration')
def alts_elcm(buildings):
    columns = [
        'non_residential_sqft',
        'ln_jobs_within_30min',
        'ln_avg_nonres_unit_price_zone',
        'median_year_built',
        'ln_residential_unit_density_zone',
        'ln_pop_within_20min',
        'nonres_far',
        'office',
        'retail_or_restaurant',
        'industrial_building',
        'employees_x_ln_non_residential_sqft_zone',
        'ln_dist_rail',
        'rail_within_mile',
        'ln_emp_sector1_within_15min',
        'ln_emp_sector2_within_15min',
        'ln_emp_sector3_within_15min',
        'ln_emp_sector4_within_15min',
        'ln_emp_sector5_within_15min',
        'ln_emp_sector6_within_15min'
    ]
    alts = buildings.to_frame(columns=columns)
    alts.fillna(0, inplace=True)
    return alts

@orca.table('alts_repm', cache=True, cache_scope='iteration')
def alts_repm(buildings):
    columns=[
        'ln_non_residential_sqft_zone',
        'building_type_id',
        'unit_price_residential',
        'improvement_value',
        'ln_pop_within_20min',
        'ln_units_per_acre',
        'mean_income',
        'year_built',
        'ln_dist_bus',
        'ln_dist_rail',
        'ln_avg_land_value_per_sqft_zone',
        'ln_residential_unit_density_zone',
        'ln_non_residential_sqft_zone',
        'allpurpose_agglosum_floor',
        'county8001',
        'county8005',
        'county8013',
        'county8014',
        'county8019',
        'county8035',
        'county8039',
        'county8047',
        'county8059',
        'county8123'

    ]
    alts = buildings.to_frame(columns=columns)
    alts.fillna(0, inplace=True)
    return alts

@orca.table('alts_nrepm', cache=True, cache_scope='iteration')
def alts_repm(buildings):
    columns=[
        'improvement_value',
        'building_type_id',
        'unit_price_non_residential',
        'ln_jobs_within_20min',
        'nonres_far',
        'year_built',
        'ln_dist_bus',
        'ln_dist_rail',
        'ln_avg_land_value_per_sqft_zone',
        'ln_residential_unit_density_zone',
        'ln_non_residential_sqft_zone',
        'allpurpose_agglosum_floor',
        'county8001',
        'county8005',
        'county8013',
        'county8014',
        'county8019',
        'county8035',
        'county8039',
        'county8047',
        'county8059',
        'county8123'

    ]
    alts = buildings.to_frame(columns=columns)
    alts.fillna(0, inplace=True)
    return alts

@orca.table('household_relocation_rates')
def household_relocation_rates(store):
    df = store['annual_household_relocation_rates']
    df.probability_of_relocating = df.probability_of_relocating / 100
    return df

@orca.table('job_relocation_rates')
def job_relocation_rates(store):
    return store['annual_job_relocation_rates']

@orca.table('household_control_totals')
def household_control_totals(store):
    df = store['annual_household_control_totals']
    df = df[['total_number_of_households']]
    return df

@orca.table('employment_control_totals')
def employment_control_totals(store):
    return store['annual_employment_control_totals']

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

#automagic merging
orca.broadcast('zones', 'parcels', cast_index=True, onto_on='zone_id')
orca.broadcast('zones', 'households', cast_index=True, onto_on='zone_id')
orca.broadcast('parcels', 'buildings', onto_on='parcel_id', cast_index=True, onto_index=False)
orca.broadcast('zones', 'establishments', cast_index=True, onto_on='zone_id')
orca.broadcast('zoning', 'parcels', cast_index=True, onto_on='zoning_id')
orca.broadcast('fars', 'parcels', cast_index=True, onto_on='far_id')
