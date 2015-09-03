import orca
import pandas as pd
import numpy as np
import dataset
from urbansim.utils import misc
from urbansim.utils.misc import reindex

#register calculated variables with data table objects

#####################
# PARCEL VARIABLES
#####################


@orca.column('parcels', 'in_denver', cache=True)
def in_denver(parcels):
    return (parcels.county_id==8031).astype('int32')

@orca.column('parcels', 'ln_dist_bus', cache=True)
def ln_dist_bus(parcels):
    return parcels.dist_bus.apply(np.log1p)

@orca.column('parcels', 'ln_land_value', cache=True)
def ln_land_value(parcels):
    return parcels.land_value.apply(np.log1p)

@orca.column('parcels', 'land_value_per_sqft', cache=True)
def land_value_per_sqft(parcels):
    return (parcels.land_value*1.0/parcels.parcel_sqft)

@orca.column('parcels', 'rail_within_mile', cache=True)
def rail_within_mile(parcels):
    return (parcels.dist_rail<5280).astype('int32')

@orca.column('parcels', 'cherry_creek_school_district', cache=True)
def cherry_creek_school_district(parcels):
    return (parcels.school_district == 8).astype('int32')

@orca.column('parcels', 'acres', cache=True)
def acres(parcels):
    return parcels.parcel_sqft/43560.0

@orca.column('parcels', 'ln_acres', cache=True)
def ln_acres(parcels):
    return (parcels.parcel_sqft/43560.0).apply(np.log1p)

@orca.column('parcels', 'nonres_far', cache=True)
def nonres_far(buildings, parcels):
    return (buildings.non_residential_sqft.groupby(buildings.parcel_id).sum()/parcels.acres).apply(np.log1p)

@orca.column('parcels', 'ln_units_per_acre', cache=True)
def ln_units_per_acre(buildings, parcels):
    return (buildings.residential_units.groupby(buildings.parcel_id).sum()/parcels.acres).apply(np.log1p)


@orca.column('parcels','land_cost', cache=True)
def land_cost(parcels):
    return parcels.land_value

@orca.column('parcels','parcel_size', cache=True)
def land_cost(parcels):
    return parcels.parcel_sqft

@orca.column('parcels', 'ave_res_unit_size')
def ave_unit_size(parcels, buildings):
    series = pd.Series(index=parcels.index)
    zonal_sqft_per_unit = buildings.sqft_per_unit.groupby(buildings.zone_id).mean()
    series.loc[:] = zonal_sqft_per_unit[parcels.zone_id].values
    return series

@orca.column('parcels', 'ave_non_res_unit_size')
def ave_unit_size(parcels, sqft_per_job):
    series = pd.Series(index=parcels.index)
    sqft = sqft_per_job.to_frame().reset_index()
    zonal_sqft = sqft.groupby('zone_id').building_sqft_per_job.mean()
    series.loc[:] = zonal_sqft[parcels.zone_id].values
    return series

@orca.column('parcels', 'total_units', cache=True, cache_scope='iteration')
def total_units(parcels, buildings):
    return buildings.residential_units.groupby(buildings.parcel_id).sum().\
        reindex(parcels.index).fillna(0)

@orca.column('parcels', 'total_job_spaces', cache=True, cache_scope='iteration')
def total_job_spaces(parcels, buildings):
    return buildings.non_residential_units.groupby(buildings.parcel_id).sum().\
        reindex(parcels.index).fillna(0)

#####################
# BUILDING VARIABLES
#####################
@orca.column('buildings', 'vacant_residential_units')
def vacant_residential_units(buildings, households):
    return buildings.residential_units.sub(
        households.building_id.value_counts(), fill_value=0)

@orca.column('buildings', 'vacant_job_spaces')
def vacant_residential_units(buildings, establishments):
    return buildings.non_residential_units.sub(
        establishments.employees.groupby(establishments.building_id).sum(), fill_value=0)

@orca.column('buildings', 'zone_id', cache=True, cache_scope='iteration')
def zone_id(buildings, parcels):
    series = pd.Series(index=buildings.building_type_id.index)
    series.loc[:] = parcels.zone_id[buildings.parcel_id].values
    return series

@orca.column('buildings', 'county_id', cache=True, cache_scope='iteration')
def county_id(buildings, zone_to_county):
    series = pd.Series(index=buildings.building_type_id.index)
    series.loc[:] = zone_to_county.county_id[buildings.zone_id].values
    return series

@orca.column('buildings', 'non_residential_units', cache=True, cache_scope='iteration')
def non_residential_units(buildings, sqft_per_job, establishments):
    b = buildings.to_frame(columns=['zone_id', 'building_type_id','non_residential_sqft'])
    b = pd.merge(b, sqft_per_job.to_frame(), left_on=[b.zone_id,b.building_type_id], right_index=True, how='left')
    b.loc[:, 'non_residential_units'] = (b.non_residential_sqft / b.building_sqft_per_job).fillna(0).astype('int')
    b.loc[:, 'base_year_jobs'] = establishments.employees.groupby(establishments.building_id).sum()
    return b[['non_residential_units', 'base_year_jobs']].max(axis=1)


@orca.column('buildings','townhome', cache=True)
def townhome(buildings):
    return (buildings.building_type_id == 24).astype('int32')

@orca.column('buildings','multifamily', cache=True)
def multifamily(buildings):
    return (np.in1d(buildings.building_type_id, [2,3])).astype('int32')

@orca.column('buildings', 'office', cache=True)
def office(buildings):
    return (buildings.building_type_id==5).astype('int32')

@orca.column('buildings', 'retail_or_restaurant', cache=True)
def retail_or_restaurant(buildings):
    return (np.in1d(buildings.building_type_id, [17,18])).astype('int32')

@orca.column('buildings', 'industrial_building', cache=True)
def industrial_building(buildings):
    return (np.in1d(buildings.building_type_id, [9,22])).astype('int32')

@orca.column('buildings', 'residential_sqft', cache=True)
def residential_sqft(buildings):
    return (buildings.bldg_sq_ft - buildings.non_residential_sqft)

@orca.column('buildings', 'btype_hlcm', cache=True)
def btype_hlcm(buildings):
    return 1*(buildings.building_type_id==2) + 2*(buildings.building_type_id==3) + 3*(buildings.building_type_id==20) + 4*np.invert(np.in1d(buildings.building_type_id,[2,3,20]))

@orca.column('buildings','county8001', cache=True, cache_scope='iteration')
def county8001(buildings):
    return (buildings.county_id == 8001).astype('int32')

@orca.column('buildings','county8005', cache=True, cache_scope='iteration')
def county8005(buildings):
    return (buildings.county_id == 8005).astype('int32')

@orca.column('buildings','county8013', cache=True, cache_scope='iteration')
def county8013(buildings):
    return (buildings.county_id == 8013).astype('int32')

@orca.column('buildings','county8014', cache=True, cache_scope='iteration')
def county8014(buildings):
    return (buildings.county_id == 8014).astype('int32')

@orca.column('buildings','county8019', cache=True, cache_scope='iteration')
def county8019(buildings):
    return (buildings.county_id == 8019).astype('int32')

@orca.column('buildings','county8031', cache=True, cache_scope='iteration')
def county8031(buildings):
    return (buildings.county_id == 8031).astype('int32')

@orca.column('buildings','county8035', cache=True, cache_scope='iteration')
def county8035(buildings):
    return (buildings.county_id == 8035).astype('int32')

@orca.column('buildings','county8039', cache=True, cache_scope='iteration')
def county8039(buildings):
    return (buildings.county_id == 8039).astype('int32')

@orca.column('buildings','county8047', cache=True, cache_scope='iteration')
def county8047(buildings):
    return (buildings.county_id == 8047).astype('int32')

@orca.column('buildings','county8059', cache=True, cache_scope='iteration')
def county8059(buildings):
    return (buildings.county_id == 8059).astype('int32')

@orca.column('buildings','county8123', cache=True, cache_scope='iteration')
def county8123(buildings):
    return (buildings.county_id == 8123).astype('int32')


#####zonal variables for each building

#####variables for HLCM
@orca.column('buildings', 'ln_dist_rail', cache=True, cache_scope='iteration')
def ln_dist_rail(buildings, parcels):
    series = pd.Series(index=buildings.building_type_id.index)
    series.loc[:] = parcels.dist_rail[buildings.parcel_id].values
    return series

@orca.column('buildings', 'ln_avg_unit_price_zone', cache=True, cache_scope='iteration')
def ln_avg_unit_price_zone(buildings, zones):
    series = pd.Series(index=buildings.building_type_id.index)
    zonal_avg_price_ln = buildings.unit_price_residential[(buildings.residential_units>0)&(buildings.improvement_value>0)].groupby(buildings.zone_id).mean().apply(np.log1p)
    series.loc[:] = zonal_avg_price_ln[buildings.zone_id].values
    return series

@orca.column('buildings', 'median_age_of_head', cache=True, cache_scope='iteration')
def median_age_of_head(buildings, households):
    series = pd.Series(index=buildings.building_type_id.index)
    zonal_median_age = households.age_of_head.groupby(buildings.zone_id).mean()
    series.loc[:] = zonal_median_age[buildings.zone_id].values
    return series

@orca.column('buildings', 'median_yearbuilt_post_1990', cache=True, cache_scope='iteration')
def median_yearbuilt_post_1990(buildings):
    return (buildings.year_built.groupby(buildings.zone_id).median() > 1990).astype('int32')

@orca.column('buildings', 'percent_hh_with_child_x_hh_with_child', cache=True, cache_scope='iteration')
def percent_hh_with_child_x_hh_with_child(buildings, households):
    series = pd.Series(index=buildings.building_type_id.index)
    percent_hh_with_child = households.age_of_head[households.children>0].groupby(households.zone_id).size()*100.0 / households.age_of_head.groupby(households.zone_id).size()
    percent_hh_with_child_x_hh_with_child = percent_hh_with_child * households.hh_with_child.groupby(households.zone_id).size()
    series.loc[:] = percent_hh_with_child_x_hh_with_child[buildings.zone_id].values
    return series

@orca.column('buildings', 'percent_renter_hh_in_zone', cache=True, cache_scope='iteration')
def percent_renter_hh_in_zone(buildings, households):
    series = pd.Series(index=buildings.building_type_id.index)
    zonal_pct_renter = households.tenure[households.tenure==2].groupby(households.zone_id).size()*100.0 / households.tenure.groupby(households.zone_id).size()
    series.loc[:] = zonal_pct_renter[buildings.zone_id].values
    return series

@orca.column('buildings', 'jobs_within_45min',  cache=True, cache_scope='iteration')
def jobs_within_45min(buildings, t_data_dist45, establishments):
    zonal_emp=establishments.employees.groupby(establishments.zone_id).sum()
    t_data=t_data_dist45.to_frame()
    t_data.loc[:,'attr']=zonal_emp[t_data_dist45.to_zone_id].values
    zone_time_range=t_data.groupby(level=0).attr.apply(np.sum)
    return reindex(zone_time_range,buildings.zone_id)


@orca.column('buildings', 'jobs_within_30min',  cache=True, cache_scope='iteration')
def jobs_within_30min(buildings, t_data_dist30, establishments):
    zonal_emp=establishments.employees.groupby(establishments.zone_id).sum()
    t_data=t_data_dist30.to_frame()
    t_data.loc[:,'attr']=zonal_emp[t_data_dist30.to_zone_id].values
    zone_time_range=t_data.groupby(level=0).attr.apply(np.sum)
    return reindex(zone_time_range,buildings.zone_id)

@orca.column('buildings', 'ln_jobs_within_30min', cache=True, cache_scope='iteration')
def ln_jobs_within_30min(buildings):
    return np.log1p(buildings.jobs_within_30min)


@orca.column('buildings', 'jobs_within_20min', cache=True, cache_scope='iteration')
def jobs_within_20min(buildings, t_data_dist20, establishments):
    zonal_emp=establishments.employees.groupby(establishments.zone_id).sum()
    t_data=t_data_dist20.to_frame()
    t_data.loc[:,'attr']=zonal_emp[t_data_dist20.to_zone_id].values
    zone_time_range=t_data.groupby(level=0).attr.apply(np.sum)
    return reindex(zone_time_range,buildings.zone_id)

@orca.column('buildings', 'ln_jobs_within_20min', cache=True, cache_scope='iteration')
def ln_jobs_within_20min(buildings):
    return np.log1p(buildings.jobs_within_20min)

@orca.column('buildings', 'income5xlt_x_avg_unit_price_zone', cache=True, cache_scope='iteration')
def income5xlt_x_avg_unit_price_zone(households, buildings):
    zonal_avg_price = buildings.unit_price_residential[(buildings.residential_units>0)&(buildings.improvement_value>0)].groupby(buildings.zone_id).mean()
    income5xlt = households.income5xlt.groupby(households.zone_id).mean()
    income5xlt_x_avg_unit_price_zone = zonal_avg_price * income5xlt
    return reindex(income5xlt_x_avg_unit_price_zone,buildings.zone_id)


@orca.column('buildings', 'median_yearbuilt_pre_1950', cache=True)
def median_yearbuilt_pre_1950(buildings):
    return (buildings.year_built.groupby(buildings.zone_id).median() < 1950).astype('int32')

@orca.column('buildings', 'ln_income_x_average_resunit_size', cache=True, cache_scope='iteration')
def ln_income_x_average_resunit_size(households, buildings):
    ln_income = households.ln_income.groupby(households.zone_id).mean()
    avg_resunit_size = buildings.sqft_per_unit.groupby(buildings.zone_id).mean()
    ln_income_x_average_resunit_size = ln_income * avg_resunit_size
    return reindex(ln_income_x_average_resunit_size,buildings.zone_id)

@orca.column('buildings', 'wkrs_hhs_x_ln_jobs_within_30min', cache=True, cache_scope='iteration')
def wkrs_hhs_x_ln_jobs_within_30min(buildings, households):
    ln_jobs_within_30min = buildings.jobs_within_30min.apply(np.log1p)
    wkrs_hhs = households.wkrs_hhs.groupby(households.building_id).sum()
    return wkrs_hhs * ln_jobs_within_30min

@orca.column('buildings', 'mean_income', cache=True, cache_scope='iteration')
def mean_income(buildings, households):
    series = pd.Series(index=buildings.building_type_id.index)
    zonal_hh_income = households.income.groupby(households.zone_id).mean()
    series.loc[:] = zonal_hh_income[buildings.zone_id].values
    return series

@orca.column('buildings', 'cherry_creek_school_district', cache=True, cache_scope='iteration')
def cherry_creek_school_district(buildings, parcels):
    series = pd.Series(index=buildings.building_type_id.index)
    series.loc[:] = parcels.cherry_creek_school_district[buildings.parcel_id].values
    return series


@orca.column('buildings', 'percent_younghead_x_younghead', cache=True, cache_scope='iteration')
def percent_younghead_x_younghead(buildings, households):
    series = pd.Series(index=buildings.building_type_id.index)
    percent_younghead_x_younghead = (households.age_of_head[households.age_of_head < 30].groupby(households.zone_id).size() * 100.0 / households.age_of_head.groupby(households.zone_id).size()) * households.age_of_head[households.age_of_head < 30].groupby(households.zone_id).size()
    series.loc[:] = percent_younghead_x_younghead[buildings.zone_id].values
    return series
@orca.column('buildings', 'ln_emp_sector3_within_20min', cache=True, cache_scope='iteration')
def ln_emp_sector3_within_20min(buildings, t_data_dist20, establishments):
    zonal_emp=establishments.employees[establishments.sector_id_six == 3].groupby(establishments.zone_id).sum()
    t_data=t_data_dist20.to_frame()
    t_data.loc[:,'attr']=zonal_emp[t_data_dist20.to_zone_id].values
    zone_time_range=t_data.groupby(level=0).attr.apply(np.sum)
    return reindex(zone_time_range,buildings.zone_id).apply(np.log1p)

@orca.column('buildings', 'ln_emp_sector3_within_15min', cache=True, cache_scope='iteration')
def ln_emp_sector3_within_15min(buildings, t_data_dist15, establishments):
    zonal_emp=establishments.employees[establishments.sector_id_six == 3].groupby(establishments.zone_id).sum()
    t_data=t_data_dist15.to_frame()
    t_data.loc[:,'attr']=zonal_emp[t_data_dist15.to_zone_id].values
    zone_time_range=t_data.groupby(level=0).attr.apply(np.sum)
    return reindex(zone_time_range,buildings.zone_id).apply(np.log1p)

@orca.column('buildings', 'ln_emp_sector1_within_15min', cache=True, cache_scope='iteration')
def ln_emp_sector1_within_15min(buildings, t_data_dist15, establishments):
    zonal_emp=establishments.employees[establishments.sector_id_six == 1].groupby(establishments.zone_id).sum()
    t_data=t_data_dist15.to_frame()
    t_data.loc[:,'attr']=zonal_emp[t_data_dist15.to_zone_id].values
    zone_time_range=t_data.groupby(level=0).attr.apply(np.sum)
    return reindex(zone_time_range,buildings.zone_id).apply(np.log1p)

@orca.column('buildings', 'ln_emp_sector2_within_15min', cache=True, cache_scope='iteration')
def ln_emp_sector2_within_15min(buildings, t_data_dist15, establishments):
    zonal_emp=establishments.employees[establishments.sector_id_six == 2].groupby(establishments.zone_id).sum()
    t_data=t_data_dist15.to_frame()
    t_data.loc[:,'attr']=zonal_emp[t_data_dist15.to_zone_id].values
    zone_time_range=t_data.groupby(level=0).attr.apply(np.sum)
    return reindex(zone_time_range,buildings.zone_id).apply(np.log1p)

@orca.column('buildings', 'ln_emp_sector4_within_15min', cache=True, cache_scope='iteration')
def ln_emp_sector4_within_15min(buildings, t_data_dist15, establishments):
    zonal_emp=establishments.employees[establishments.sector_id_six == 4].groupby(establishments.zone_id).sum()
    t_data=t_data_dist15.to_frame()
    t_data.loc[:,'attr']=zonal_emp[t_data_dist15.to_zone_id].values
    zone_time_range=t_data.groupby(level=0).attr.apply(np.sum)
    return reindex(zone_time_range,buildings.zone_id).apply(np.log1p)

@orca.column('buildings', 'ln_emp_sector5_within_15min', cache=True, cache_scope='iteration')
def ln_emp_sector5_within_15min(buildings, t_data_dist15, establishments):
    zonal_emp=establishments.employees[establishments.sector_id_six == 5].groupby(establishments.zone_id).sum()
    t_data=t_data_dist15.to_frame()
    t_data.loc[:,'attr']=zonal_emp[t_data_dist15.to_zone_id].values
    zone_time_range=t_data.groupby(level=0).attr.apply(np.sum)
    return reindex(zone_time_range,buildings.zone_id).apply(np.log1p)

@orca.column('buildings', 'ln_emp_sector6_within_15min', cache=True, cache_scope='iteration')
def ln_emp_sector6_within_15min(buildings, t_data_dist15, establishments):
    zonal_emp=establishments.employees[establishments.sector_id_six == 6].groupby(establishments.zone_id).sum()
    t_data=t_data_dist15.to_frame()
    t_data.loc[:,'attr']=zonal_emp[t_data_dist15.to_zone_id].values
    zone_time_range=t_data.groupby(level=0).attr.apply(np.sum)
    return reindex(zone_time_range,buildings.zone_id).apply(np.log1p)

@orca.column('buildings' ,'allpurpose_agglosum_floor', cache=True, cache_scope='iteration')
def allpurpose_agglosum_floor(buildings, zones):
    series = pd.Series(index=buildings.building_type_id.index)
    allpurpose_agglosum_floor = (zones.allpurpose_agglosum>=0)*(zones.allpurpose_agglosum)
    series.loc[:] = allpurpose_agglosum_floor[buildings.zone_id].values
    return series

#####variables for ELCM
@orca.column('buildings', 'ln_avg_nonres_unit_price_zone', cache=True, cache_scope='iteration')
def ln_avg_nonres_unit_price_zone(buildings, zones):
    series = pd.Series(index=buildings.building_type_id.index)
    zonal_avg_price_ln = buildings.unit_price_non_residential[(buildings.non_residential_sqft>0)&(buildings.improvement_value>0)].groupby(buildings.zone_id).mean().apply(np.log1p)
    series.loc[:] = zonal_avg_price_ln[buildings.zone_id].values
    return series

@orca.column('buildings','median_year_built', cache=True, cache_scope='iteration')
def median_year_built(buildings):
    series = pd.Series(index=buildings.building_type_id.index)
    median_year_built = buildings.year_built.groupby(buildings.zone_id).median().astype('int32')
    series.loc[:] = median_year_built[buildings.zone_id].values
    return series

@orca.column('buildings', 'ln_residential_unit_density_zone', cache=True, cache_scope='iteration')
def ln_residential_unit_density_zone(buildings, zones):
    series = pd.Series(index=buildings.building_type_id.index)
    ln_residential_unit_density_zone = (buildings.residential_units.groupby(buildings.zone_id).sum() / zones.acreage).apply(np.log1p)
    series.loc[:] = ln_residential_unit_density_zone[buildings.zone_id].values
    return series

@orca.column('buildings', 'ln_pop_within_20min', cache=True, cache_scope='iteration')
def ln_pop_within_20min(buildings, t_data_dist20,households):
    zonal_pop=households.persons.groupby(households.zone_id).sum()
    t_data=t_data_dist20.to_frame()
    t_data.loc[:,'attr']=zonal_pop[t_data_dist20.to_zone_id].values
    zone_time_range=t_data.groupby(level=0).attr.apply(np.sum)
    return reindex(zone_time_range,buildings.zone_id).apply(np.log1p)


@orca.column('buildings', 'nonres_far', cache=True, cache_scope='iteration')
def nonres_far(buildings, parcels):
    series = pd.Series(index=buildings.building_type_id.index)
    series.loc[:] = parcels.nonres_far[buildings.parcel_id].values
    return series

@orca.column('buildings', 'employees_x_ln_non_residential_sqft_zone', cache=True, cache_scope='iteration')
def employees_x_ln_non_residential_sqft_zone(buildings, establishments):
    series = pd.Series(index=buildings.building_type_id.index)
    ln_non_residential_sqft_zone = buildings.non_residential_sqft.groupby(buildings.zone_id).sum().apply(np.log1p)
    employees = establishments.employees.groupby(establishments.zone_id).sum()
    employees_x_ln_non_residential_sqft_zone = employees * ln_non_residential_sqft_zone
    series.loc[:] = employees_x_ln_non_residential_sqft_zone[buildings.zone_id].values
    return series

@orca.column('buildings', 'ln_non_residential_sqft_zone')
def ln_non_residential_sqft_zone(buildings):
    series = pd.Series(index=buildings.building_type_id.index)
    ln_non_residential_sqft_zone = buildings.non_residential_sqft.groupby(buildings.zone_id).sum().apply(np.log1p)
    series.loc[:] = ln_non_residential_sqft_zone[buildings.zone_id].values
    return series


@orca.column('buildings', 'rail_within_mile', cache=True, cache_scope='iteration')
def rail_within_mile(buildings, parcels):
    series = pd.Series(index=buildings.building_type_id.index)
    series.loc[:] = parcels.rail_within_mile[buildings.parcel_id].values
    return series


#####variables for REPM
@orca.column('buildings','ln_units_per_acre', cache=True, cache_scope='iteration')
def ln_units_per_acre(buildings, parcels):
    series = pd.Series(index=buildings.building_type_id.index)
    series.loc[:] = parcels.ln_units_per_acre[buildings.parcel_id].values
    return series

@orca.column('buildings', 'ln_dist_bus', cache=True, cache_scope='iteration')
def ln_dist_bus(buildings, parcels):
    series = pd.Series(index=buildings.building_type_id.index)
    series.loc[:] = parcels.ln_dist_bus[buildings.parcel_id].values
    return series

@orca.column('buildings', 'ln_avg_land_value_per_sqft_zone', cache=True, cache_scope='iteration')
def ln_avg_land_value_per_sqft_zone(parcels, buildings):
    series = pd.Series(index=buildings.building_type_id.index)
    zonal_avg = parcels.land_value_per_sqft.groupby(parcels.zone_id).mean().apply(np.log1p)
    series.loc[:] = zonal_avg[buildings.zone_id].values
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

@orca.column('households_for_estimation', 'zone_id', cache=True, cache_scope='iteration')
def zone_id(buildings, households_for_estimation):
    series = pd.Series(index=households_for_estimation.income_group.index)
    series.loc[:] = buildings.zone_id[households_for_estimation.building_id].values
    return series

@orca.column('households_for_estimation', 'building_type_id')
def building_type_id(buildings, households_for_estimation):
    series = pd.Series(index=households_for_estimation.income_group.index)
    series.loc[:] = buildings.building_type_id[households_for_estimation.building_id].values
    return series

@orca.column('households_for_estimation', 'county_id', cache=True, cache_scope='iteration')
def county_id(households_for_estimation, buildings):
    series = pd.Series(index=households_for_estimation.income_group.index)
    series.loc[:] = buildings.county_id[households_for_estimation.building_id].values
    return series

@orca.column('households_for_estimation', 'btype')
def btype(households_for_estimation):
    return 1*(households_for_estimation.building_type_id==2) + 2*(households_for_estimation.building_type_id==3) + 3*(households_for_estimation.building_type_id==20) + 4*np.invert(np.in1d(households_for_estimation.building_type_id,[2,3,20]))

@orca.column('households_for_estimation', 'income_3_tenure', cache=True, cache_scope='iteration')
def income_3_tenure(households_for_estimation):
    return 1 * (households_for_estimation.income < 60000)*(households_for_estimation.tenure == 1) + 2 * np.logical_and(households_for_estimation.income >= 60000, households_for_estimation.income < 120000)*(households_for_estimation.tenure == 1) + 3*(households_for_estimation.income >= 120000)*(households_for_estimation.tenure == 1) + 4*(households_for_estimation.income < 40000)*(households_for_estimation.tenure == 2) + 5*(households_for_estimation.income >= 40000)*(households_for_estimation.tenure == 2)

@orca.column('households_for_estimation', 'younghead', cache=True)
def younghead(households_for_estimation):
    return households_for_estimation.age_of_head<30

@orca.column('households_for_estimation', 'hh_with_child', cache=True)
def hh_with_child(households_for_estimation):
    return households_for_estimation.children>0

@orca.column('households_for_estimation', 'ln_income', cache=True, cache_scope='iteration')
def ln_income(households_for_estimation):
    return households_for_estimation.income.apply(np.log1p)

@orca.column('households_for_estimation', 'income5xlt', cache=True, cache_scope='iteration')
def income5xlt(households_for_estimation):
    return households_for_estimation.income*5.0

@orca.column('households_for_estimation', 'income10xlt', cache=True, cache_scope='iteration')
def income10xlt(households_for_estimation):
    return households_for_estimation.income*5.0

@orca.column('households_for_estimation', 'wkrs_hhs', cache=True, cache_scope='iteration')
def wkrs_hhs(households_for_estimation):
    return households_for_estimation.workers*1.0/households_for_estimation.persons




#####################
# HOUSEHOLDS VARIABLES
#####################

@orca.column('households', 'zone_id', cache=True, cache_scope='iteration')
def zone_id(buildings, households):
    series = pd.Series(index=households.age_of_head.index)
    series.loc[:] = buildings.zone_id[households.building_id].values
    return series

@orca.column('households', 'building_type_id')
def building_type_id(buildings, households):
    series = pd.Series(index=households.age_of_head.index)
    series.loc[:] = buildings.building_type_id[households.building_id].values
    return series

@orca.column('households', 'county_id', cache=True, cache_scope='iteration')
def county_id(households, buildings):
    series = pd.Series(index=households.age_of_head.index)
    series.loc[:] = buildings.county_id[households.building_id].values
    return series

@orca.column('households', 'btype')
def btype(households):
    return 1*(households.building_type_id==2) + 2*(households.building_type_id==3) + 3*(households.building_type_id==20) + 4*np.invert(np.in1d(households.building_type_id,[2,3,20]))

@orca.column('households', 'income_3_tenure', cache=True, cache_scope='iteration')
def income_3_tenure(households):
    return 1 * (households.income < 60000)*(households.tenure == 1) + 2 * np.logical_and(households.income >= 60000, households.income < 120000)*(households.tenure == 1) + 3*(households.income >= 120000)*(households.tenure == 1) + 4*(households.income < 40000)*(households.tenure == 2) + 5*(households.income >= 40000)*(households.tenure == 2)

@orca.column('households', 'younghead', cache=True)
def younghead(households):
    return households.age_of_head<30

@orca.column('households', 'hh_with_child', cache=True)
def hh_with_child(households):
    return households.children>0

@orca.column('households', 'ln_income', cache=True, cache_scope='iteration')
def ln_income(households):
    return households.income.apply(np.log1p)

@orca.column('households', 'income5xlt', cache=True, cache_scope='iteration')
def income5xlt(households):
    return households.income*5.0

@orca.column('households', 'income10xlt', cache=True, cache_scope='iteration')
def income10xlt(households):
    return households.income*5.0

@orca.column('households', 'wkrs_hhs', cache=True, cache_scope='iteration')
def wkrs_hhs(households):
    return households.workers*1.0/households.persons

#####################
# ESTABLISHMENT VARIABLES
#####################
@orca.column('establishments', 'zone_id', cache=True, cache_scope='iteration')
def zone_id(establishments, buildings):
    series = pd.Series(index=establishments.employees.index)
    series.loc[:] = buildings.zone_id[establishments.building_id].values
    return series

@orca.column('establishments', 'county_id', cache=True, cache_scope='iteration')
def county_id(buildings, establishments):
    series = pd.Series(index=establishments.employees.index)
    series.loc[:] = buildings.county_id[establishments.building_id].values
    return series

@orca.column('establishments', 'sector_id_six', cache=True)
def sector_id_six(establishments):
    e = establishments
    return 1*(e.sector_id==61) + 2*(e.sector_id==71) + 3*np.in1d(e.sector_id,[11,21,22,23,31,32,33,42,48,49]) + 4*np.in1d(e.sector_id,[7221,7222,7224]) + 5*np.in1d(e.sector_id,[44,45,7211,7212,7213,7223]) + 6*np.in1d(e.sector_id,[51,52,53,54,55,56,62,81,92])

@orca.column('establishments', 'sector_id_retail_agg', cache=True)
def sector_id_retail_agg(establishments):
    e = establishments
    return e.sector_id*np.logical_not(np.in1d(e.sector_id,[7211,7212,7213])) + 7211*np.in1d(e.sector_id,[7211,7212,7213])

@orca.column('establishments', 'nonres_sqft', cache=True, cache_scope='iteration')
def nonres_sqft(buildings, establishments):
    series = pd.Series(index=establishments.employees.index)
    series.loc[:] = buildings.non_residential_sqft[establishments.building_id].values
    return series