__author__ = 'JMartinez'
import orca
import pandas as pd
import numpy as np
from urbansim.utils.misc import reindex


#joined table-wrappers


#####################
# PARCEL VARIABLES
#####################

@orca.column('parcels', 'in_denver', cache=True)
def in_denver(parcels):
    return (parcels.county_id==8031).astype('int32')

@orca.column('parcels', 'ln_dist_bus', cache=True)
def ln_dist_bus(parcels):
    return parcels.dist_bus.apply(np.log1p)

@orca.column('parcels', 'ln_dist_rail', cache=True)
def ln_dist_rail(parcels):
    return parcels.dist_rail.apply(np.log1p)

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

@orca.column('parcels', 'ln_units_per_acre', cache=True)
def ln_units_per_acre(buildings, parcels):
    return (buildings.residential_units.groupby(buildings.parcel_id).sum()/parcels.acres).apply(np.log1p)

@orca.column('parcels', 'nonres_far', cache=True)
def nonres_far(buildings, parcels):
    return (buildings.non_residential_sqft.groupby(buildings.parcel_id).sum()/parcels.acres).apply(np.log1p)

@orca.column('parcels','parcel_size', cache=True)
def land_cost(parcels):
    return parcels.parcel_sqft

@orca.column('parcels','land_cost', cache=True)
def land_cost(parcels):
    return parcels.land_value

@orca.column('parcels', 'ave_res_unit_size')
def ave_unit_size(parcels, buildings):
    building_zone = pd.Series(index=buildings.index)
    b_zone_id = parcels.zone_id.loc[buildings.parcel_id]
    building_zone.loc[:] = b_zone_id.values
    series = pd.Series(index=parcels.index)
    zonal_sqft_per_unit = buildings.sqft_per_unit.groupby(building_zone).mean()
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
#
@orca.column('buildings', 'non_residential_units', cache=True, cache_scope='iteration')
def non_residential_units(sqft_per_job, establishments):
    b = orca.merge_tables('buildings', tables=['buildings','parcels'], columns=['zone_id', 'building_type_id','non_residential_sqft'])
    b = pd.merge(b, sqft_per_job.to_frame(), left_on=[b.zone_id,b.building_type_id], right_index=True, how='left')
    b.loc[:, 'non_residential_units'] = np.ceil((b.non_residential_sqft / b.building_sqft_per_job).fillna(0)).astype('int')
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
    df = orca.merge_tables('buildings', tables=['buildings','parcels'], columns=['county_id'])
    return (df.county_id == 8001).astype('int32')

@orca.column('buildings','county8005', cache=True, cache_scope='iteration')
def county8005(buildings):
    df = orca.merge_tables('buildings', tables=['buildings','parcels'], columns=['county_id'])
    return (df.county_id == 8005).astype('int32')

@orca.column('buildings','county8013', cache=True, cache_scope='iteration')
def county8013(buildings):
    df = orca.merge_tables('buildings', tables=['buildings','parcels'], columns=['county_id'])
    return (df.county_id == 8013).astype('int32')

@orca.column('buildings','county8014', cache=True, cache_scope='iteration')
def county8014(buildings):
    df = orca.merge_tables('buildings', tables=['buildings','parcels'], columns=['county_id'])
    return (df.county_id == 8014).astype('int32')

@orca.column('buildings','county8019', cache=True, cache_scope='iteration')
def county8019(buildings):
    df = orca.merge_tables('buildings', tables=['buildings','parcels'], columns=['county_id'])
    return (df.county_id == 8019).astype('int32')

@orca.column('buildings','county8031', cache=True, cache_scope='iteration')
def county8031(buildings):
    df = orca.merge_tables('buildings', tables=['buildings','parcels'], columns=['county_id'])
    return (df.county_id == 8031).astype('int32')

@orca.column('buildings','county8035', cache=True, cache_scope='iteration')
def county8035(buildings):
    df = orca.merge_tables('buildings', tables=['buildings','parcels'], columns=['county_id'])
    return (df.county_id == 8035).astype('int32')

@orca.column('buildings','county8039', cache=True, cache_scope='iteration')
def county8039(buildings):
    df = orca.merge_tables('buildings', tables=['buildings','parcels'], columns=['county_id'])
    return (df.county_id == 8039).astype('int32')

@orca.column('buildings','county8047', cache=True, cache_scope='iteration')
def county8047(buildings):
    df = orca.merge_tables('buildings', tables=['buildings','parcels'], columns=['county_id'])
    return (df.county_id == 8047).astype('int32')

@orca.column('buildings','county8059', cache=True, cache_scope='iteration')
def county8059(buildings):
    df = orca.merge_tables('buildings', tables=['buildings','parcels'], columns=['county_id'])
    return (df.county_id == 8059).astype('int32')

@orca.column('buildings','county8123', cache=True, cache_scope='iteration')
def county8123(buildings):
    df = orca.merge_tables('buildings', tables=['buildings','parcels'], columns=['county_id'])
    return (df.county_id == 8123).astype('int32')


@orca.column('buildings', 'percent_hh_with_child_x_hh_with_child', cache=True, cache_scope='iteration')
def percent_hh_with_child_x_hh_with_child(buildings, households, parcels, zones):
    building_zones = pd.Series(index=buildings.index)
    hh_zones = pd.Series(index=households.index)
    b_zone_id = parcels.zone_id.loc[buildings.parcel_id]
    building_zones.loc[:] = b_zone_id.values
    hh_zone_id = building_zones.loc[households.building_id]
    hh_zones.loc[:] = hh_zone_id.values
    percent_hh_with_child = zones.percent_hh_with_child
    percent_hh_with_child_x_hh_with_child = percent_hh_with_child * households.hh_with_child.groupby(hh_zones).size()

    series = pd.Series(index=buildings.index)
    series.loc[:] = percent_hh_with_child_x_hh_with_child[building_zones].values
    return series

@orca.column('buildings', 'employees_x_ln_non_residential_sqft_zone', cache=True, cache_scope='iteration')
def employees_x_ln_non_residential_sqft_zone(buildings, establishments, parcels, zones):
    building_zones = pd.Series(index=buildings.index)
    e_zones = pd.Series(index=establishments.index)
    b_zone_id = parcels.zone_id.loc[buildings.parcel_id]
    building_zones.loc[:] = b_zone_id.values
    e_zone_id = building_zones.loc[establishments.building_id]
    e_zones.loc[:] = e_zone_id.values

    ln_non_residential_sqft_zone = buildings.non_residential_sqft.groupby(building_zones).sum().apply(np.log1p)
    employees = establishments.employees.groupby(e_zones).sum()
    employees_x_ln_non_residential_sqft_zone = employees * ln_non_residential_sqft_zone
    series = pd.Series(index=buildings.index)
    series.loc[:] = employees_x_ln_non_residential_sqft_zone[building_zones].values
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

@orca.column('households_for_estimation', 'btype')
def btype(households_for_estimation, buildings):
    df = orca.merge_tables('households_for_estimation', tables=['households_for_estimation','buildings'], columns=['building_type_id'])
    return 1*(df.building_type_id==2) + 2*(df.building_type_id==3) + 3*(df.building_type_id==20) + 4*np.invert(np.in1d(df.building_type_id,[2,3,20]))



#####################
# HOUSEHOLDS VARIABLES
#####################

@orca.column('households', 'btype')
def btype(households):
    df = orca.merge_tables('households', tables=['households','buildings'], columns=['building_type_id'])
    return 1*(df.building_type_id==2) + 2*(df.building_type_id==3) + 3*(df.building_type_id==20) + 4*np.invert(np.in1d(df.building_type_id,[2,3,20]))


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

@orca.column('establishments', 'sector_id_six', cache=True)
def sector_id_six(establishments):
    e = establishments
    return 1*(e.sector_id==61) + 2*(e.sector_id==71) + 3*np.in1d(e.sector_id,[11,21,22,23,31,32,33,42,48,49]) + 4*np.in1d(e.sector_id,[7221,7222,7224]) + 5*np.in1d(e.sector_id,[44,45,7211,7212,7213,7223]) + 6*np.in1d(e.sector_id,[51,52,53,54,55,56,62,81,92])


@orca.column('establishments', 'sector_id_retail_agg', cache=True)
def sector_id_retail_agg(establishments):
    e = establishments
    return e.sector_id*np.logical_not(np.in1d(e.sector_id,[7211,7212,7213])) + 7211*np.in1d(e.sector_id,[7211,7212,7213])


#####################
# ZONE VARIABLES
#####################

@orca.column('zones', 'zonal_hh', cache=True, cache_scope='iteration')
def zonal_hh():
    df = orca.merge_tables('households', tables=['households','buildings', 'parcels'], columns=['building_type_id'])
    return df.groupby('zone_id').size()

@orca.column('zones', 'zonal_emp', cache=True, cache_scope='iteration')
def zonal_emp():
    df = orca.merge_tables('establishments', tables=['establishments','buildings', 'parcels'], columns=['employees'])
    return df.groupby('zone_id').employees.sum()

@orca.column('zones', 'residential_sqft_zone', cache=True, cache_scope='iteration')
def residential_sqft_zone():
    df = orca.merge_tables('buildings', tables=['buildings','parcels'], columns=['residential_sqft'])
    return df.groupby('zone_id').residential_sqft.sum()

@orca.column('zones', 'zonal_pop', cache=True, cache_scope='iteration')
def zonal_pop():
    df = orca.merge_tables('households', tables=['households','buildings', 'parcels'], columns=['persons'])
    return df.groupby('zone_id').persons.sum()

@orca.column('zones', 'residential_units_zone', cache=True, cache_scope='iteration')
def residential_units_zone():
    df = orca.merge_tables('buildings', tables=['buildings','parcels'], columns=['residential_units'])
    return df.groupby('zone_id').residential_units.sum()

@orca.column('zones', 'ln_residential_units_zone', cache=True, cache_scope='iteration')
def residential_units_zone():
    df = orca.merge_tables('buildings', tables=['buildings','parcels'], columns=['residential_units'])
    return df.groupby('zone_id').residential_units.sum().apply(np.log1p)

@orca.column('zones', 'ln_residential_unit_density_zone', cache=True, cache_scope='iteration')
def ln_residential_unit_density_zone(zones):
    df = orca.merge_tables('buildings', tables=['buildings','parcels'], columns=['residential_units'])
    return (df.groupby('zone_id').residential_units.sum()/zones.acreage).apply(np.log1p)


@orca.column('zones', 'non_residential_sqft_zone', cache=True, cache_scope='iteration')
def non_residential_sqft_zone():
    df = orca.merge_tables('buildings', tables=['buildings','parcels'], columns=['non_residential_sqft'])
    return df.groupby('zone_id').non_residential_sqft.sum()

@orca.column('zones', 'ln_non_residential_sqft_zone', cache=True, cache_scope='iteration')
def ln_non_residential_sqft_zone():
    df = orca.merge_tables('buildings', tables=['buildings','parcels'], columns=['non_residential_sqft'])
    return df.groupby('zone_id').non_residential_sqft.sum().apply(np.log1p)

@orca.column('zones', 'percent_sf', cache=True, cache_scope='iteration')
def ln_non_residential_sqft_zone():
    b = orca.merge_tables('buildings', tables=['buildings','parcels'], columns=['btype_hlcm','residential_units'])
    return b[b.btype_hlcm==3].groupby('zone_id').residential_units.sum()*100.0/(b.groupby('zone_id').residential_units.sum())


@orca.column('zones', 'avg_unit_price_zone', cache=True, cache_scope='iteration')
def avg_unit_price_zone():
    b = orca.merge_tables('buildings', tables=['buildings','parcels'], columns=['residential_units','improvement_value','unit_price_residential'])
    return  b[(b.residential_units>0)*(b.improvement_value>0)].groupby('zone_id').unit_price_residential.mean()


@orca.column('zones', 'ln_avg_unit_price_zone', cache=True, cache_scope='iteration')
def avg_unit_price_zone(zones):
    return zones.avg_unit_price_zone.apply(np.log1p)


@orca.column('zones', 'avg_nonres_unit_price_zone', cache=True, cache_scope='iteration')
def avg_unit_price_zone():
    b = orca.merge_tables('buildings', tables=['buildings','parcels'], columns=['non_residential_sqft','improvement_value','unit_price_non_residential'])

    return  b[(b.non_residential_sqft>0)*(b.improvement_value>0)].groupby('zone_id').unit_price_non_residential.mean()


@orca.column('zones', 'ln_avg_nonres_unit_price_zone', cache=True, cache_scope='iteration')
def avg_unit_price_zone(zones):
    return zones.avg_nonres_unit_price_zone.apply(np.log1p)

@orca.column('zones', 'median_age_of_head', cache=True, cache_scope='iteration')
def median_age_of_head():
    hh = orca.merge_tables('households', tables=['households','buildings','parcels'], columns=['age_of_head'])
    return hh.groupby('zone_id').age_of_head.median()

@orca.column('zones', 'mean_income', cache=True, cache_scope='iteration')
def mean_income():
    hh = orca.merge_tables('households', tables=['households','buildings','parcels'], columns=['income'])
    return hh.groupby('zone_id').income.mean()

@orca.column('zones', 'median_year_built', cache=True, cache_scope='iteration')
def median_year_built():
    b = orca.merge_tables('buildings', tables=['buildings','parcels'], columns=['year_built'])
    return b.groupby('zone_id').year_built.median().astype('int32')

@orca.column('zones', 'ln_avg_land_value_per_sqft_zone', cache=True, cache_scope='iteration')
def ln_avg_land_value_per_sqft_zone(parcels):

    return parcels.land_value_per_sqft.groupby(parcels.zone_id).mean().apply(np.log1p)


@orca.column('zones', 'median_yearbuilt_post_1990', cache=True, cache_scope='iteration')
def median_yearbuilt_post_1990():
    b = orca.merge_tables('buildings', tables=['buildings','parcels'], columns=['year_built'])
    return (b.groupby('zone_id').year_built.median()>1990).astype('int32')

@orca.column('zones', 'median_yearbuilt_pre_1950', cache=True, cache_scope='iteration')
def median_yearbuilt_pre_1950():
    b = orca.merge_tables('buildings', tables=['buildings','parcels'], columns=['year_built'])
    return (b.groupby('zone_id').year_built.median()<1950).astype('int32')

@orca.column('zones', 'percent_hh_with_child', cache=True, cache_scope='iteration')
def percent_hh_with_child(zones):
    hh = orca.merge_tables('households', tables=['households','buildings','parcels'], columns=['children'])
    return hh[hh.children>0].groupby('zone_id').size()*100.0/zones.zonal_hh

@orca.column('zones', 'percent_renter_hh_in_zone', cache=True, cache_scope='iteration')
def percent_renter_hh_in_zone(zones):
    hh = orca.merge_tables('households', tables=['households','buildings','parcels'], columns=['tenure'])
    return hh[hh.tenure==2].groupby('zone_id').size()*100.0/zones.zonal_hh

@orca.column('zones', 'percent_younghead', cache=True, cache_scope='iteration')
def percent_younghead(zones):
    hh = orca.merge_tables('households', tables=['households','buildings','parcels'], columns=['age_of_head'])
    return hh[hh.age_of_head<30].groupby('zone_id').size()*100.0/zones.zonal_hh

@orca.column('zones', 'average_resunit_size', cache=True, cache_scope='iteration')
def average_resunit_size():
    b = orca.merge_tables('buildings', tables=['buildings','parcels'], columns=['sqft_per_unit'])
    return  b.groupby('zone_id').sqft_per_unit.mean()



@orca.column('zones', 'emp_sector_agg', cache=True, cache_scope='iteration')
def emp_sector_agg():
    e = orca.merge_tables('establishments', tables=['establishments','buildings','parcels'], columns=['sector_id','employees'])
    return  e[e.sector_id==1].groupby('zone_id').employees.sum()

@orca.column('zones', 'emp_sector1', cache=True, cache_scope='iteration')
def emp_sector1():
    e = orca.merge_tables('establishments', tables=['establishments','buildings','parcels'], columns=['sector_id_six','employees'])
    return  e[e.sector_id_six==1].groupby('zone_id').employees.sum()

@orca.column('zones', 'emp_sector2', cache=True, cache_scope='iteration')
def emp_sector2():
    e = orca.merge_tables('establishments', tables=['establishments','buildings','parcels'], columns=['sector_id_six','employees'])
    return  e[e.sector_id_six==2].groupby('zone_id').employees.sum()

@orca.column('zones', 'emp_sector3', cache=True, cache_scope='iteration')
def emp_sector3():
    e = orca.merge_tables('establishments', tables=['establishments','buildings','parcels'], columns=['sector_id_six','employees'])
    return  e[e.sector_id_six==3].groupby('zone_id').employees.sum()

@orca.column('zones', 'emp_sector4', cache=True, cache_scope='iteration')
def emp_sector4():
    e = orca.merge_tables('establishments', tables=['establishments','buildings','parcels'], columns=['sector_id_six','employees'])
    return  e[e.sector_id_six==4].groupby('zone_id').employees.sum()

@orca.column('zones', 'emp_sector5', cache=True, cache_scope='iteration')
def emp_sector5():
    e = orca.merge_tables('establishments', tables=['establishments','buildings','parcels'], columns=['sector_id_six','employees'])
    return  e[e.sector_id_six==5].groupby('zone_id').employees.sum()

@orca.column('zones', 'emp_sector6', cache=True, cache_scope='iteration')
def emp_sector6():
    e = orca.merge_tables('establishments', tables=['establishments','buildings','parcels'], columns=['sector_id_six','employees'])
    return  e[e.sector_id_six==6].groupby('zone_id').employees.sum()

@orca.column('zones', 'emp_sector6', cache=True, cache_scope='iteration')
def emp_sector6():
    e = orca.merge_tables('establishments', tables=['establishments','buildings','parcels'], columns=['sector_id_six','employees'])
    return  e[e.sector_id_six==6].groupby('zone_id').employees.sum()

@orca.column('zones', 'jobs_within_45min', cache=True, cache_scope='iteration')
def jobs_within_45min(zones, t_data_dist45):
    b = orca.merge_tables('buildings', tables=['buildings','parcels'], columns=['zone_id'])
    t_data=t_data_dist45.to_frame()
    t_data.loc[:,'attr']=zones.zonal_emp[t_data_dist45.to_zone_id].values
    zone_time_range=t_data.groupby(level=0).attr.apply(np.sum)
    return reindex(zone_time_range, b.zone_id)

@orca.column('zones', 'ln_jobs_within_45min', cache=True, cache_scope='iteration')
def ln_jobs_within_45min(zones):
    return zones.jobs_within_45min.apply(np.log1p)

@orca.column('zones', 'jobs_within_30min', cache=True, cache_scope='iteration')
def jobs_within_30min(zones, t_data_dist30):
    b = orca.merge_tables('buildings', tables=['buildings','parcels'], columns=['zone_id'])
    t_data=t_data_dist30.to_frame()
    t_data.loc[:,'attr']=zones.zonal_emp[t_data_dist30.to_zone_id].values
    zone_time_range=t_data.groupby(level=0).attr.apply(np.sum)
    return reindex(zone_time_range, b.zone_id)

@orca.column('zones', 'ln_jobs_within_30min', cache=True, cache_scope='iteration')
def ln_jobs_within_30min(zones):
    return zones.jobs_within_30min.apply(np.log1p)

@orca.column('zones', 'jobs_within_20min', cache=True, cache_scope='iteration')
def jobs_within_20min(zones, t_data_dist20):
    b = orca.merge_tables('buildings', tables=['buildings','parcels'], columns=['zone_id'])
    t_data=t_data_dist20.to_frame()
    t_data.loc[:,'attr']=zones.zonal_emp[t_data_dist20.to_zone_id].values
    zone_time_range=t_data.groupby(level=0).attr.apply(np.sum)
    return reindex(zone_time_range, b.zone_id)

@orca.column('zones', 'ln_jobs_within_20min', cache=True, cache_scope='iteration')
def ln_jobs_within_20min(zones):
    return zones.jobs_within_20min.apply(np.log1p)

@orca.column('zones', 'jobs_within_15min', cache=True, cache_scope='iteration')
def jobs_within_15min(zones, t_data_dist15):
    b = orca.merge_tables('buildings', tables=['buildings','parcels'], columns=['zone_id'])
    t_data=t_data_dist15.to_frame()
    t_data.loc[:,'attr']=zones.zonal_emp[t_data_dist15.to_zone_id].values
    zone_time_range=t_data.groupby(level=0).attr.apply(np.sum)
    return reindex(zone_time_range, b.zone_id)

@orca.column('zones', 'ln_jobs_within_15min', cache=True, cache_scope='iteration')
def ln_jobs_within_15min(zones):
    return zones.jobs_within_15min.apply(np.log1p)

@orca.column('zones', 'ln_emp_sector1_within_15min', cache=True, cache_scope='iteration')
def ln_emp_sector1_within_15min(t_data_dist15):
    b = orca.merge_tables('buildings', tables=['buildings','parcels'], columns=['zone_id'])
    e = orca.merge_tables('establishments', tables=['establishments','buildings','parcels'], columns=['sector_id_six','zone_id','employees'])
    e = e.loc[e.sector_id_six == 1]
    zonal_emp = e.groupby('zone_id').employees.sum()
    t_data=t_data_dist15.to_frame()
    t_data.loc[:,'attr']=zonal_emp[t_data_dist15.to_zone_id].values
    zone_time_range=t_data.groupby(level=0).attr.apply(np.sum)
    return reindex(zone_time_range,b.zone_id).apply(np.log1p)

@orca.column('zones', 'ln_emp_sector2_within_15min', cache=True, cache_scope='iteration')
def ln_emp_sector2_within_15min(t_data_dist15):
    b = orca.merge_tables('buildings', tables=['buildings','parcels'], columns=['zone_id'])
    e = orca.merge_tables('establishments', tables=['establishments','buildings','parcels'], columns=['sector_id_six','zone_id','employees'])
    e = e.loc[e.sector_id_six == 2]
    zonal_emp = e.groupby('zone_id').employees.sum()
    t_data=t_data_dist15.to_frame()
    t_data.loc[:,'attr']=zonal_emp[t_data_dist15.to_zone_id].values
    zone_time_range=t_data.groupby(level=0).attr.apply(np.sum)
    return reindex(zone_time_range,b.zone_id).apply(np.log1p)

@orca.column('zones', 'ln_emp_sector3_within_15min', cache=True, cache_scope='iteration')
def ln_emp_sector4_within_15min(t_data_dist15):
    b = orca.merge_tables('buildings', tables=['buildings','parcels'], columns=['zone_id'])
    e = orca.merge_tables('establishments', tables=['establishments','buildings','parcels'], columns=['sector_id_six','zone_id','employees'])
    e = e.loc[e.sector_id_six == 3]
    zonal_emp = e.groupby('zone_id').employees.sum()
    t_data=t_data_dist15.to_frame()
    t_data.loc[:,'attr']=zonal_emp[t_data_dist15.to_zone_id].values
    zone_time_range=t_data.groupby(level=0).attr.apply(np.sum)
    return reindex(zone_time_range,b.zone_id).apply(np.log1p)

@orca.column('zones', 'ln_emp_sector4_within_15min', cache=True, cache_scope='iteration')
def ln_emp_sector4_within_15min(t_data_dist15):
    b = orca.merge_tables('buildings', tables=['buildings','parcels'], columns=['zone_id'])
    e = orca.merge_tables('establishments', tables=['establishments','buildings','parcels'], columns=['sector_id_six','zone_id','employees'])
    e = e.loc[e.sector_id_six == 4]
    zonal_emp = e.groupby('zone_id').employees.sum()
    t_data=t_data_dist15.to_frame()
    t_data.loc[:,'attr']=zonal_emp[t_data_dist15.to_zone_id].values
    zone_time_range=t_data.groupby(level=0).attr.apply(np.sum)
    return reindex(zone_time_range,b.zone_id).apply(np.log1p)

@orca.column('zones', 'ln_emp_sector5_within_15min', cache=True, cache_scope='iteration')
def ln_emp_sector5_within_15min(t_data_dist15):
    b = orca.merge_tables('buildings', tables=['buildings','parcels'], columns=['zone_id'])
    e = orca.merge_tables('establishments', tables=['establishments','buildings','parcels'], columns=['sector_id_six','zone_id','employees'])
    e = e.loc[e.sector_id_six == 5]
    zonal_emp = e.groupby('zone_id').employees.sum()
    t_data=t_data_dist15.to_frame()
    t_data.loc[:,'attr']=zonal_emp[t_data_dist15.to_zone_id].values
    zone_time_range=t_data.groupby(level=0).attr.apply(np.sum)
    return reindex(zone_time_range,b.zone_id).apply(np.log1p)

@orca.column('zones', 'ln_emp_sector6_within_15min', cache=True, cache_scope='iteration')
def ln_emp_sector6_within_15min(t_data_dist15):
    b = orca.merge_tables('buildings', tables=['buildings','parcels'], columns=['zone_id'])
    e = orca.merge_tables('establishments', tables=['establishments','buildings','parcels'], columns=['sector_id_six','zone_id','employees'])
    e = e.loc[e.sector_id_six == 6]
    zonal_emp = e.groupby('zone_id').employees.sum()
    t_data=t_data_dist15.to_frame()
    t_data.loc[:,'attr']=zonal_emp[t_data_dist15.to_zone_id].values
    zone_time_range=t_data.groupby(level=0).attr.apply(np.sum)
    return reindex(zone_time_range,b.zone_id).apply(np.log1p)

@orca.column('zones', 'ln_emp_sector3_within_20min', cache=True, cache_scope='iteration')
def ln_emp_sector3_within_20min(t_data_dist20):
    b = orca.merge_tables('buildings', tables=['buildings','parcels'], columns=['zone_id'])
    e = orca.merge_tables('establishments', tables=['establishments','buildings','parcels'], columns=['sector_id_six','zone_id','employees'])
    e = e.loc[e.sector_id_six == 3]
    zonal_emp = e.groupby('zone_id').employees.sum()
    t_data=t_data_dist20.to_frame()
    t_data.loc[:,'attr']=zonal_emp[t_data_dist20.to_zone_id].values
    zone_time_range=t_data.groupby(level=0).attr.apply(np.sum)
    return reindex(zone_time_range,b.zone_id).apply(np.log1p)

@orca.column('zones', 'ln_emp_sector5_within_20min', cache=True, cache_scope='iteration')
def ln_emp_sector5_within_20min(t_data_dist20):
    b = orca.merge_tables('buildings', tables=['buildings','parcels'], columns=['zone_id'])
    e = orca.merge_tables('establishments', tables=['establishments','buildings','parcels'], columns=['sector_id_six','zone_id','employees'])
    e = e.loc[e.sector_id_six == 5]
    zonal_emp = e.groupby('zone_id').employees.sum()
    t_data=t_data_dist20.to_frame()
    t_data.loc[:,'attr']=zonal_emp[t_data_dist20.to_zone_id].values
    zone_time_range=t_data.groupby(level=0).attr.apply(np.sum)
    return reindex(zone_time_range,b.zone_id).apply(np.log1p)


@orca.column('zones', 'allpurpose_agglosum_floor', cache=True, cache_scope='iteration')
def allpurpose_agglosum_floor(zones):
    return (zones.allpurpose_agglosum>=0)*(zones.allpurpose_agglosum)

@orca.column('zones', 'ln_pop_within_20min', cache=True, cache_scope='iteration')
def ln_pop_within_20min(zones, t_data_dist20):
    b = orca.merge_tables('buildings', tables=['buildings','parcels'], columns=['zone_id'])
    zonal_pop=zones.zonal_pop
    t_data=t_data_dist20.to_frame()
    t_data.loc[:,'attr']=zonal_pop[t_data_dist20.to_zone_id].values
    zone_time_range=t_data.groupby(level=0).attr.apply(np.sum)
    return reindex(zone_time_range, b.zone_id).apply(np.log1p)

@orca.column('buildings', 'wkrs_hhs_x_ln_jobs_within_30min', cache=True, cache_scope='iteration')
def wkrs_hhs_x_ln_jobs_within_30min(buildings, households, parcels, zones):
    building_data = pd.DataFrame(index=buildings.index)
    p_zone_id = parcels.zone_id
    b_zone_id = p_zone_id.loc[buildings.parcel_id]
    building_data.loc[:, 'zone_id'] = b_zone_id.values
    building_data.loc[:, 'ln_jobs_within_30min'] = zones.ln_jobs_within_30min.loc[building_data.zone_id].values
    wkrs_hhs = households.wkrs_hhs.groupby(households.building_id).sum()
    return wkrs_hhs * building_data.ln_jobs_within_30min

@orca.column('buildings', 'ln_income_x_average_resunit_size', cache=True, cache_scope='iteration')
def ln_income_x_average_resunit_size(households, buildings, parcels):
    building_data = pd.Series(index=buildings.index)
    p_zone_id = parcels.zone_id
    b_zone_id = p_zone_id.loc[buildings.parcel_id]
    building_data.loc[:] = b_zone_id.values
    hh_data = pd.Series(index=households.index)
    hh_zone_id = building_data.loc[households.building_id]
    hh_data.loc[:] = hh_zone_id.values

    ln_income = households.ln_income.groupby(hh_data).mean()
    avg_resunit_size = buildings.sqft_per_unit.groupby(building_data).mean()
    ln_income_x_average_resunit_size = ln_income * avg_resunit_size
    return reindex(ln_income_x_average_resunit_size,building_data)

@orca.column('buildings', 'percent_younghead_x_younghead', cache=True, cache_scope='iteration')
def percent_younghead_x_younghead(buildings, households, zones,parcels):
    building_data = pd.Series(index=buildings.index)
    p_zone_id = parcels.zone_id
    b_zone_id = p_zone_id.loc[buildings.parcel_id]
    building_data.loc[:] = b_zone_id.values
    hh_data = pd.Series(index=households.index)
    hh_zone_id = building_data.loc[households.building_id]
    hh_data.loc[:] = hh_zone_id.values

    percent_younghead = zones.percent_younghead
    younghead = households.age_of_head.groupby(hh_data).size()
    percent_younghead_x_younghead = percent_younghead * younghead
    return reindex(percent_younghead_x_younghead, building_data)