import random
import orca
import new_dataset
import utils_drcog
import new_variables
import assumptions
import numpy as np
import pandas as pd
from sqlalchemy import engine


@orca.injectable('year')
def year(iter_var):
    return iter_var


@orca.step('hlcm_simulate')
def hlcm_simulate(households, zones, counties):
    return utils_drcog.lcm_simulate('c:/urbansim_new/urbansim/urbansim_drcog/config/zonal_hlcm_yaml.yaml',
                                    households, zones, counties, 'zone_id')

@orca.step('elcm_simulate')
def elcm_simulate(establishments, zones, counties):
    return utils_drcog.elcm_simulate('c:/urbansim_new/urbansim/urbansim_drcog/config/zonal_elcm_yaml.yaml',
                                    establishments, zones, counties, 'zone_id')

@orca.step('hh_relocation')
def hh_relocation(households, household_relocation_rates):
    return utils_drcog.relocation_model(households, household_relocation_rates, 'zone_id')

@orca.step('emp_relocation')
def emp_relocation(establishments, job_relocation_rates):
    return utils_drcog.emp_relocation_model(establishments, job_relocation_rates, 'zone_id')

@orca.step('hh_transition')
def hh_transition(households, household_control_totals, year):
    if(year <= 2040):
        return utils_drcog.hh_transition(households,household_control_totals, 'zone_id', year)
    else:
        return

@orca.step('emp_transition')
def emp_transition(employment_control_totals, year):
    if(year <= 2040):
        return utils_drcog.emp_transition(employment_control_totals, 'zone_id', year)
    else:
        return

@orca.step('rsh_simulate')
def rsh_simulate(buildings, parcels, zones):
    return utils_drcog.hedonic_simulate('c:/urbansim_new/urbansim/urbansim_drcog/config/repm_yaml.yaml',
                                        buildings, parcels, zones, 'unit_price_residential')

@orca.step('nrh_simulate')
def nrh_simulate(buildings, parcels, zones):
    return utils_drcog.hedonic_simulate('c:/urbansim_new/urbansim/urbansim_drcog/config/nrepm_yaml.yaml',
                                        buildings, parcels, zones, 'unit_price_non_residential')

@orca.step('feasibility')
def feasibility(parcels):
    return utils_drcog.run_feasibility(parcels, assumptions.parcel_avg_price, assumptions.parcel_is_allowed, residential_to_yearly=False)


@orca.step('residential_developer')
def residential_developer(feasibility, households, buildings, parcels, year):
    utils_drcog.run_developer(["residential","mixedresidential"],
                        households,
                        buildings,
                        "residential_units",
                        parcels.parcel_size,
                        parcels.ave_res_unit_size,
                        parcels.total_units,
                        feasibility,
                        year=year,
                        target_vacancy=0.20,
                        form_to_btype_callback=random_type,
                        add_more_columns_callback=add_extra_columns,
                        bldg_sqft_per_job=400.0, price_col='unit_price_residential')

@orca.step('non_residential_developer')
def non_residential_developer(feasibility, establishments, buildings, parcels, year):
    employees = establishments.employees
    agents = employees.ix[np.repeat(employees.index.values, employees.values)]
    utils_drcog.run_developer(["office", "retail", "industrial"],
                        agents,
                        buildings,
                        "non_residential_units",
                        parcels.parcel_size,
                        parcels.ave_non_res_unit_size,
                        parcels.total_job_spaces,
                        feasibility,
                        year=year,
                        target_vacancy=0.50,
                        form_to_btype_callback=random_type,
                        add_more_columns_callback=add_extra_columns_non_res,
                        residential=False,
                        bldg_sqft_per_job=400.0, price_col='unit_price_non_residential')

@orca.step('indicator_export')
def indicator_export(zones, year):
    utils_drcog.export_indicators(zones, year)

@orca.step('res_supply_demand')
def res_supply_demand(zones, hh_demand):
    utils_drcog.supply_demand('c:/urbansim_new/urbansim/urbansim_drcog/config/zonal_hlcm_yaml.yaml',
                              hh_demand, zones, 'avg_unit_price_zone', units_col='residential_units_zone', iterations=1)

@orca.step('non_res_supply_demand')
def non_res_supply_demand(zones, emp_demand):
    utils_drcog.supply_demand('c:/urbansim_new/urbansim/urbansim_drcog/config/zonal_elcm_yaml.yaml',
                              emp_demand, zones, 'avg_nonres_unit_price_zone', units_col='non_residential_sqft_zone')

@orca.step('scenario_zoning_change')
def scenario_zoning_change(parcels, fars):
    eng = engine.create_engine('postgresql://model_team:m0d3lte@m@postgresql:5432/urbansim', echo=False)
    eng_sandbox = engine.create_engine('postgresql://model_team:m0d3lte@m@postgresql:5432/sandbox', echo=False)
    df = pd.read_sql('parcels_scenario', eng, index_col='parcel_id')
    parcels.update_col_from_series('far_id', df.far_id)

    #change zoning to add allowable types
    zoning = pd.read_sql('parcels_spatial_fars', eng_sandbox, index_col='parcel_id',
                         columns=['parcel_id', 'zone_id','type1','type2','type3','type4',
                                  'type5','type6','type7','type8','type9','type10','type11',
                                  'type12','type14','type15','type16','type17','type18','type19',
                                  'type20','type21','type22','type23','type24','type25'])
    #zoning = zoning.drop('geom', axis=1)
    #change multifamily types to allowed
    orca.add_table('zoning_baseline', zoning)

@orca.step('households_to_buildings')
def household_to_buildings(households, buildings):
    unplaced_hh = households.to_frame(columns=['zone_id','building_id'])
    unplaced_hh = unplaced_hh.loc[unplaced_hh.building_id == -1]

    b = orca.merge_tables('buildings', tables=['parcels','buildings'], columns=['zone_id','residential_units', 'vacant_residential_units'])
    vacant_units = b.loc[b.vacant_residential_units > 0, 'vacant_residential_units']
    units = b.loc[np.repeat(vacant_units.index.values, vacant_units.values.astype('int'))].reset_index()

    print len(units)
    print len(unplaced_hh)

def random_type(form):
    form_to_btype = orca.get_injectable("form_to_btype")
    return random.choice(form_to_btype[form])

def add_extra_columns(df):
    for col in ['improvement_value','land_area','sqft_per_unit','tax_exempt','bldg_sq_ft','unit_price_non_residential','unit_price_residential']:
        df[col] = 0
    return df

def add_extra_columns_non_res(df):
    for col in ['improvement_value','land_area','sqft_per_unit','tax_exempt','bldg_sq_ft','unit_price_non_residential','unit_price_residential']:
        df[col] = 0
    df.rename(columns={'job_spaces':'non_residential_units'}, inplace=True)
    return df

@orca.step('buildings_to_uc')
def buildings_to_uc(buildings, year):
    #if settings['urbancanvas']:
        # Export newly predicted buildings (from proforma or scheduled_development_events) to Urban Canvas
        from urbansim.urbancanvas import urbancanvas2
        b = buildings.to_frame(buildings.local_columns)

        # Only buildings for this simulation year
        new_buildings = b[b.year_built == year]

        # Required columns
        if 'development_type_id' in new_buildings.columns:
            new_buildings = new_buildings.rename(columns ={'development_type_id':'building_type_id'})
        new_buildings = new_buildings[[ 'building_type_id', 'improvement_value',
                                       'non_residential_sqft', 'parcel_id', 'residential_units', 'sqft_per_unit', 'stories', 'year_built', 'bldg_sq_ft', ]]  # These are the fields you need.  And index should be named 'building_id'.
        for col in ['parcel_id', 'residential_units', 'non_residential_sqft', 'year_built',
                    'stories', 'bldg_sq_ft', 'sqft_per_unit']:
            new_buildings[col] = new_buildings[col].fillna(0).astype('int32')

        # Export to database
        urbancanvas2.buildings_to_uc(new_buildings, year)
    #else:
    #    print 'Urban Canvas setting not on.'


@orca.step('scheduled_development_events')
def scheduled_development_events(buildings, scheduled_development_events, year):

    sched_dev = scheduled_development_events.to_frame()
    print sched_dev
    sched_dev = sched_dev[sched_dev.year_built==year]
    sched_dev['residential_sqft'] = sched_dev.sqft_per_unit*sched_dev.residential_units
    sched_dev['job_spaces'] = sched_dev.non_residential_sqft/400
    sched_dev = sched_dev.rename(columns={'development_type_id': 'building_type_id',
                                          'nonres_rent_per_sqft': 'unit_price_non_residential' })
    sched_dev['tax_exempt'] = 0
    sched_dev['land_area'] = 0
    sched_dev['bldg_sq_ft'] = sched_dev['residential_sqft'] + sched_dev['non_residential_sqft']
    sched_dev['unit_price_residential'] = sched_dev['res_price_per_sqft']*\
                                          sched_dev['residential_sqft']/sched_dev.residential_units

    if len(sched_dev) > 0:
        max_bid = buildings.index.values.max()
        idx = np.arange(max_bid + 1,max_bid+len(sched_dev)+1)
        sched_dev['building_id'] = idx
        sched_dev = sched_dev.set_index('building_id')
        from urbansim.developer.developer import Developer
        merge = Developer(pd.DataFrame({})).merge
        b = buildings.to_frame(buildings.local_columns)
        all_buildings = merge(b,sched_dev[b.columns])
        orca.add_table("buildings", all_buildings)