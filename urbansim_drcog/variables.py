__author__ = 'JMartinez'
import orca
import pandas as pd
import numpy as np
import dataset
from urbansim.utils import misc

#register calculated variables with data table objects

#####################
# PARCEL VARIABLES
#####################


@orca.column('parcels', 'in_denver')
def in_denver(parcels):
    return (parcels.county_id==8031).astype('int32')

@orca.column('parcels', 'ln_dist_bus')
def ln_dist_bus(parcels):
    return parcels.dist_bus.apply(np.log1p)

@orca.column('parcels', 'ln_land_value')
def ln_land_value(parcels):
    return parcels.land_value.apply(np.log1p)

@orca.column('parcels', 'land_value_per_sqft')
def land_value_per_sqft(parcels):
    return (parcels.land_value*1.0/parcels.parcel_sqft)

@orca.column('parcels', 'rail_within_mile')
def rail_within_mile(parcels):
    return (parcels.dist_rail<5280).astype('int32')

@orca.column('parcels', 'cherry_creek_school_district')
def cherry_creek_school_district(parcels):
    return (parcels.school_district == 8).astype('int32')

@orca.column('parcels', 'acres')
def acres(parcels):
    return parcels.parcel_sqft/43560.0

@orca.column('parcels', 'ln_acres')
def ln_acres(parcels):
    return (parcels.parcel_sqft/43560.0).apply(np.log1p)

@orca.column('parcels', 'nonres_far')
def nonres_far(buildings, parcels):
    return (buildings.non_residential_sqft.groupby(buildings.parcel_id).sum()/parcels.acres).apply(np.log1p)

@orca.column('parcels', 'ln_units_per_acre')
def ln_units_per_acre(buildings, parcels):
    return (buildings.residential_units.groupby(buildings.parcel_id).sum()/parcels.acres).apply(np.log1p)







#####################
# BUILDING VARIABLES
#####################
@orca.column('buildings', 'zone_id')
def zone_id(buildings, parcels):
    series = pd.Series(buildings.building_type_id.index)
    series.loc[:] = parcels.zone_id[buildings.parcel_id].values
    return series

@orca.column('buildings', 'non_residential_units')
def non_residential_units(store, sqft_per_job, establishments):
    b = store['buildings']
    b = pd.merge(b, sqft_per_job.to_frame(), left_on=['zone_id','building_type_id'], right_index=True, how='left')
    b.loc[:, 'non_residential_units'] = b.non_residential_sqft / b.building_sqft_per_job
    b.loc[:, 'base_year_jobs'] = establishments.employees.groupby(establishments.building_id).sum()
    return b[['non_residential_units', 'base_year_jobs']].max(axis=1)


@orca.column('buildings','townhome')
def townhome(buildings):
    return (buildings.building_type_id == 24).astype('int32')

@orca.column('buildings','multifamily')
def multifamily(buildings):
    return (np.in1d(buildings.building_type_id, [2,3])).astype('int32')

@orca.column('buildings', 'office')
def office(buildings):
    return (buildings.building_type_id==5).astype('int32')

@orca.column('buildings', 'retail_or_restaurant')
def retail_or_restaurant(buildings):
    return (np.in1d(buildings.building_type_id, [17,18])).astype('int32')

@orca.column('buildings', 'industrial_building')
def industrial_building(buildings):
    return (np.in1d(buildings.building_type_id, [9,22])).astype('int32')

@orca.column('buildings', 'residential_sqft')
def residential_sqft(buildings):
    return (buildings.bldg_sq_ft - buildings.non_residential_sqft)

@orca.column('buildings', 'btype_hlcm')
def btype_hlcm(buildings):
    return 1*(buildings.building_type_id==2) + 2*(buildings.building_type_id==3) + 3*(buildings.building_type_id==20) + 4*np.invert(np.in1d(buildings.building_type_id,[2,3,20]))

@orca.column('buildings','county8001')
def county8001(buildings):
    return (buildings.county_id == 8001).astype('int32')

@orca.column('buildings','county8005')
def county8005(buildings):
    return (buildings.county_id == 8005).astype('int32')

@orca.column('buildings','county8013')
def county8013(buildings):
    return (buildings.county_id == 8013).astype('int32')

@orca.column('buildings','county8014')
def county8014(buildings):
    return (buildings.county_id == 8014).astype('int32')

@orca.column('buildings','county8019')
def county8019(buildings):
    return (buildings.county_id == 8019).astype('int32')

@orca.column('buildings','county8031')
def county8031(buildings):
    return (buildings.county_id == 8031).astype('int32')

@orca.column('buildings','county8035')
def county8035(buildings):
    return (buildings.county_id == 8035).astype('int32')

@orca.column('buildings','county8039')
def county8039(buildings):
    return (buildings.county_id == 8039).astype('int32')

@orca.column('buildings','county8047')
def county8047(buildings):
    return (buildings.county_id == 8047).astype('int32')

@orca.column('buildings','county8059')
def county8059(buildings):
    return (buildings.county_id == 8059).astype('int32')

@orca.column('buildings','county8123')
def county8123(buildings):
    return (buildings.county_id == 8123).astype('int32')


#####zonal variables for each building

#####variables for HLCM
@orca.column('buildings', 'ln_dist_rail')
def ln_dist_rail(buildings, parcels):
    series = pd.Series(index=buildings.building_type_id.index)
    series.loc[:] = parcels.dist_rail[buildings.parcel_id].values
    return series

@orca.column('buildings', 'ln_avg_unit_price_zone')
def ln_avg_unit_price_zone(buildings, zones):
    series = pd.Series(index=buildings.building_type_id.index)
    zonal_avg_price_ln = buildings.unit_price_residential[(buildings.residential_units>0)&(buildings.improvement_value>0)].groupby(buildings.zone_id).mean().apply(np.log1p)
    series.loc[:] = zonal_avg_price_ln[buildings.zone_id].values
    return series

@orca.column('buildings', 'median_age_of_head')
def median_age_of_head(buildings, households):
    series = pd.Series(index=buildings.building_type_id.index)
    zonal_median_age = households.age_of_head.groupby(buildings.zone_id).mean()
    series.loc[:] = zonal_median_age[buildings.zone_id].values
    return series

@orca.column('buildings', 'median_yearbuilt_post_1990')
def median_yearbuilt_post_1990(buildings):
    return (buildings.year_built.groupby(buildings.zone_id).median() > 1990).astype('int32')

@orca.column('buildings', 'percent_hh_with_child_x_hh_with_child')
def percent_hh_with_child_x_hh_with_child(buildings, households):
    series = pd.Series(index=buildings.building_type_id.index)
    percent_hh_with_child = households.age_of_head[households.children>0].groupby(households.zone_id).size()*100.0 / households.age_of_head.groupby(households.zone_id).size()
    percent_hh_with_child_x_hh_with_child = percent_hh_with_child * households.hh_with_child.groupby(households.zone_id).size()
    series.loc[:] = percent_hh_with_child_x_hh_with_child[buildings.zone_id].values
    return series

@orca.column('buildings', 'percent_renter_hh_in_zone')
def percent_renter_hh_in_zone(buildings, households):
    series = pd.Series(index=buildings.building_type_id.index)
    zonal_pct_renter = households.tenure[households.tenure==2].groupby(households.zone_id).size()*100.0 / households.tenure.groupby(households.zone_id).size()
    series.loc[:] = zonal_pct_renter[buildings.zone_id].values
    return series

@orca.column('buildings', 'jobs_within_45min')
def jobs_within_45min(establishments, buildings, travel_data):
    series = pd.Series(index=buildings.building_type_id.index)
    zonal_emp = establishments.employees.groupby(establishments.zone_id).sum()
    t_data = travel_data.to_frame().reset_index(level=1)
    t_data = t_data[t_data.am_single_vehicle_to_work_travel_time < 45.0]
    t_data.loc[:, 'attr'] = zonal_emp[t_data.to_zone_id].values
    zonal_travel_time = t_data.groupby(level=0).attr.apply(np.sum)
    series.loc[:] = zonal_travel_time[buildings.zone_id].values
    return series



#####################
# HOUSEHOLDS FOR ESTIMATION VARIABLES
#####################

@orca.column('households_for_estimation', 'tenure')
def households_for_estimation(households_for_estimation):
    series = pd.Series(index=households_for_estimation.own.index)
    series.loc[households_for_estimation.own > 1] = 2
    series.loc[households_for_estimation.own <= 1] = 1
    return series

@orca.column('households_for_estimation', 'income')
def income(households_for_estimation):
    series = pd.Series(index=households_for_estimation.income_group.index)
    series.loc[households_for_estimation.income_group == 1] = 7500
    series.loc[households_for_estimation.income_group == 2] = 17500
    series.loc[households_for_estimation.income_group == 3] = 25000
    series.loc[households_for_estimation.income_group == 4] = 35000
    series.loc[households_for_estimation.income_group == 5] = 45000
    series.loc[households_for_estimation.income_group == 6] = 55000
    series.loc[households_for_estimation.income_group == 7] = 67500
    series.loc[households_for_estimation.income_group == 8] = 87500
    series.loc[households_for_estimation.income_group == 9] = 117500
    series.loc[households_for_estimation.income_group == 10] = 142500
    series.loc[households_for_estimation.income_group == 11] = 200000
    return series

@orca.column('households_for_estimation', 'zone_id')
def zone_id(buildings, households_for_estimation):
    series = pd.Series(index=households_for_estimation.income_group.index)
    series.loc[:] = buildings.zone_id[households_for_estimation.building_id].values
    return series

@orca.column('households_for_estimation', 'building_type_id')
def building_type_id(buildings, households_for_estimation):
    series = pd.Series(index=households_for_estimation.income_group.index)
    series.loc[:] = buildings.building_type_id[households_for_estimation.building_id].values
    return series

@orca.column('households_for_estimation', 'county_id')
def county_id(households_for_estimation, buildings):
    series = pd.Series(index=households_for_estimation.income_group.index)
    series.loc[:] = buildings.county_id[households_for_estimation.building_id].values
    return series

@orca.column('households_for_estimation', 'btype')
def btype(households_for_estimation):
    return 1*(households_for_estimation.building_type_id==2) + 2*(households_for_estimation.building_type_id==3) + 3*(households_for_estimation.building_type_id==20) + 4*np.invert(np.in1d(households_for_estimation.building_type_id,[2,3,20]))

@orca.column('households_for_estimation', 'income_3_tenure')
def income_3_tenure(households_for_estimation):
    return 1 * (households_for_estimation.income < 60000)*(households_for_estimation.tenure == 1) + 2 * np.logical_and(households_for_estimation.income >= 60000, households_for_estimation.income < 120000)*(households_for_estimation.tenure == 1) + 3*(households_for_estimation.income >= 120000)*(households_for_estimation.tenure == 1) + 4*(households_for_estimation.income < 40000)*(households_for_estimation.tenure == 2) + 5*(households_for_estimation.income >= 40000)*(households_for_estimation.tenure == 2)

@orca.column('households_for_estimation', 'younghead')
def younghead(households_for_estimation):
    return households_for_estimation.age_of_head<30

@orca.column('households_for_estimation', 'hh_with_child')
def hh_with_child(households_for_estimation):
    return households_for_estimation.children>0

@orca.column('households_for_estimation', 'ln_income')
def ln_income(households_for_estimation):
    return households_for_estimation.income.apply(np.log1p)

@orca.column('households_for_estimation', 'income5xlt')
def income5xlt(households_for_estimation):
    return households_for_estimation.income*5.0

@orca.column('households_for_estimation', 'income10xlt')
def income10xlt(households_for_estimation):
    return households_for_estimation.income*5.0

@orca.column('households_for_estimation', 'wkrs_hhs')
def wkrs_hhs(households_for_estimation):
    return households_for_estimation.workers*1.0/households_for_estimation.persons




#####################
# HOUSEHOLDS VARIABLES
#####################

@orca.column('households', 'zone_id')
def zone_id(buildings, households):
    series = pd.Series(index=households.age_of_head.index)
    series.loc[:] = buildings.zone_id[households.building_id].values
    return series

@orca.column('households', 'building_type_id')
def building_type_id(buildings, households):
    series = pd.Series(index=households.income_group.index)
    series.loc[:] = buildings.building_type_id[households.building_id].values
    return series

@orca.column('households', 'county_id')
def county_id(households, buildings):
    series = pd.Series(index=households.income_group.index)
    series.loc[:] = buildings.county_id[households.building_id].values
    return series

@orca.column('households', 'btype')
def btype(households):
    return 1*(households.building_type_id==2) + 2*(households.building_type_id==3) + 3*(households.building_type_id==20) + 4*np.invert(np.in1d(households.building_type_id,[2,3,20]))

@orca.column('households', 'income_3_tenure')
def income_3_tenure(households):
    return 1 * (households.income < 60000)*(households.tenure == 1) + 2 * np.logical_and(households.income >= 60000, households.income < 120000)*(households.tenure == 1) + 3*(households.income >= 120000)*(households.tenure == 1) + 4*(households.income < 40000)*(households.tenure == 2) + 5*(households.income >= 40000)*(households.tenure == 2)

@orca.column('households', 'younghead')
def younghead(households):
    return households.age_of_head<30

@orca.column('households', 'hh_with_child')
def hh_with_child(households):
    return households.children>0

@orca.column('households', 'ln_income')
def ln_income(households):
    return households.income.apply(np.log1p)

@orca.column('households', 'income5xlt')
def income5xlt(households):
    return households.income*5.0

@orca.column('households', 'income10xlt')
def income10xlt(households):
    return households.income*5.0

@orca.column('households', 'wkrs_hhs')
def wkrs_hhs(households):
    return households.workers*1.0/households.persons

#####################
# ESTABLISHMENT VARIABLES
#####################
@orca.column('establishments', 'zone_id')
def zone_id(establishments, buildings):
    series = pd.Series(index=establishments.employees.index)
    series.loc[:] = buildings.zone_id[establishments.building_id].values
    return series

@orca.column('establishments', 'county_id')
def county_id(buildings, establishments):
    series = pd.Series(index=establishments.employees.index)
    series.loc[:] = buildings.county_id[establishments.building_id].values
    return series

@orca.column('establishments', 'sector_id_six')
def sector_id_six(establishments):
    e = establishments
    return 1*(e.sector_id==61) + 2*(e.sector_id==71) + 3*np.in1d(e.sector_id,[11,21,22,23,31,32,33,42,48,49]) + 4*np.in1d(e.sector_id,[7221,7222,7224]) + 5*np.in1d(e.sector_id,[44,45,7211,7212,7213,7223]) + 6*np.in1d(e.sector_id,[51,52,53,54,55,56,62,81,92])

@orca.column('establishments', 'sector_id_retail_agg')
def sector_id_retail_agg(establishments):
    e = establishments
    return e.sector_id*np.logical_not(np.in1d(e.sector_id,[7211,7212,7213])) + 7211*np.in1d(e.sector_id,[7211,7212,7213])

@orca.column('establishments', 'nonres_sqft')
def nonres_sqft(buildings, establishments):
    series = pd.Series(index=establishments.employees.index)
    series.loc[:] = buildings.non_residential_sqft[establishments.building_id].values
    return series


