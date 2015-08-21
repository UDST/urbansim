__author__ = 'JMartinez'
#####import section#####
import orca
import pandas as pd
import numpy as np
import urbansim.utils_drcog as utils


#####This module combines the all the of the functionality of urbansim to test a HLCM estimation#####


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

@orca.table('main_table', cache=False)
def main_table():
     return orca.merge_tables('buildings', tables=['buildings','parcels','zones'])

@orca.table('households_for_estimation')
def households_for_estimation(store):
    return store['households_for_estimation']


#automagic merging
orca.broadcast('zones', 'parcels', cast_index=True, onto_on='zone_id')
orca.broadcast('zones', 'households', cast_index=True, onto_on='zone_id')
orca.broadcast('parcels', 'buildings', onto_on='parcel_id', cast_index=True, onto_index=False)


#register columns ---this would go into the variables.py file
@orca.column('buildings','townhome')
def townhome(buildings):
    return (buildings.building_type_id == 24).astype('int32')

@orca.column('buildings','multifamily')
def multifamily(buildings):
    return (np.in1d(buildings.building_type_id, [2,3])).astype('int32')

@orca.column('parcels','ln_dist_rail')
def ln_dist_rail(parcels):
    return parcels.dist_rail.apply(np.log1p)

@orca.column('zones', 'ln_avg_unit_price_zone')
def ln_avg_unit_price_zone(buildings):
    return buildings.unit_price_residential[(buildings.residential_units)&(buildings.improvement_value>0)].groupby(buildings.zone_id).mean().apply(np.log1p)

@orca.column('zones', 'median_age_of_head')
def median_age_of_head(households):
    return households.age_of_head.groupby(households.zone_id).median()

@orca.column('zones', 'median_yearbuilt_post_1990')
def median_yearbuilt_post_1990(buildings):
    return buildings.year_built[buildings.year_built > 1990].groupby(buildings.zone_id).median().astype('int32')

@orca.column('zones', 'percent_renter_hh_in_zone')
def percent_renter_hh_in_zone(households):
    return (households.tenure[households.tenure==2].groupby(households.zone_id).size()*100) / (households.tenure.groupby(households.zone_id).size())

@orca.column('zones', 'jobs_within_45min')
def jobs_within_45min(establishments, travel_data):
    zonal_emp = establishments.employees.groupby(establishments.zone_id).sum()
    t_data = travel_data.to_frame().reset_index(level=1)
    t_data = t_data[t_data.am_single_vehicle_to_work_travel_time < 45.0]
    t_data.loc[:, 'attr'] = zonal_emp[t_data.to_zone_id].values
    return t_data.groupby(level=0).attr.apply(np.sum)
#test
#print orca.get_table('main_table').ln_dist_rail
#print orca.get_table('main_table').jobs_within_45min


######This section would go in utils_drcog.py and in models.py This is how we define the input schema for the models######
from urbansim.models import MNLDiscreteChoiceModel


#configure the model
model_expr = 'ln_dist_rail + ln_avg_unit_price_zone + median_age_of_head + median_yearbuilt_post_1990 + percent_renter_hh_in_zone + multifamily + townhome + jobs_within_45min'
sample_size = 100
prob_mode = 'single_chooser'
choice_mode = 'aggregate'
choice_col = 'building_id'
#e_sample_size = 3000

hlcm = MNLDiscreteChoiceModel(model_expr, sample_size, prob_mode, choice_mode, choice_column=choice_col)

current_choice = orca.get_table('households').building_id
choosers = orca.get_table('households_for_estimation').to_frame()
alternatives = orca.get_table('main_table').to_frame()[['residential_units','ln_dist_rail','ln_avg_unit_price_zone','median_age_of_head','median_yearbuilt_post_1990','percent_renter_hh_in_zone','multifamily','townhome','jobs_within_45min']]
alternatives.percent_renter_hh_in_zone.fillna(0, inplace=True)
alternatives.ln_avg_unit_price_zone.fillna(0, inplace=True)
alternatives.median_age_of_head.fillna(0, inplace=True)
alternatives.median_yearbuilt_post_1990.fillna(0, inplace=True)
_ = utils.deal_with_nas(alternatives)

alternatives = alternatives.loc[alternatives.residential_units > 0]

results = hlcm.fit(choosers, alternatives, current_choice='building_id')

hlcm.report_fit()
hlcm.to_yaml('c:/users/jmartinez/documents/hlcm_yaml.yaml')
