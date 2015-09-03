__author__ = 'JMartinez'
import random
import orca
import dataset
import utils_drcog
import variables
import assumptions
import numpy as np
import pandas as pd


@orca.injectable('year')
def year(iter_var):
    return iter_var


@orca.step('hlcm_simulate')
def hlcm_simulate(households, buildings):
    return utils_drcog.lcm_simulate('c:/urbansim_new/urbansim/urbansim_drcog/config/hlcm_yaml.yaml',
                                    households, buildings, 'building_id', 'residential_units',
                                    'vacant_residential_units')

@orca.step('elcm_simulate')
def elcm_simulate(establishments, buildings):
    return utils_drcog.lcm_simulate('c:/urbansim_new/urbansim/urbansim_drcog/config/elcm_yaml.yaml',
                                    establishments, buildings, 'building_id', 'non_residential_units',
                                    'vacant_job_spaces')

@orca.step('hh_relocation')
def hh_relocation(households, household_relocation_rates):
    return utils_drcog.relocation_model(households, household_relocation_rates, 'building_id')

@orca.step('emp_relocation')
def emp_relocation(establishments, job_relocation_rates):
    return utils_drcog.emp_relocation_model(establishments, job_relocation_rates, 'building_id')

@orca.step('hh_transition')
def hh_transition(households, household_control_totals, year):
    if(year <= 2040):
        return utils_drcog.hh_transition(households,household_control_totals, 'building_id', year)
    else:
        return

@orca.step('emp_transition')
def emp_transition(establishments, employment_control_totals, year):
    if(year <= 2040):
        return utils_drcog.emp_transition(establishments, employment_control_totals, 'building_id', year)
    else:
        return

@orca.step('rsh_simulate')
def rsh_simulate(buildings):
    return utils_drcog.hedonic_simulate('c:/urbansim_new/urbansim/urbansim_drcog/config/repm_yaml.yaml',
                                        buildings, 'unit_price_residential')

@orca.step('nrh_simulate')
def nrh_simulate(buildings):
    return utils_drcog.hedonic_simulate('c:/urbansim_new/urbansim/urbansim_drcog/config/nrepm_yaml.yaml',
                                        buildings, 'unit_price_non_residential')

@orca.step('feasibility')
def feasibility(parcels):
    return utils_drcog.run_feasibility(parcels, assumptions.parcel_avg_price, assumptions.parcel_is_allowed, residential_to_yearly=False)


@orca.step('residential_developer')
def residential_developer(feasibility, households, buildings, parcels, year):
    utils_drcog.run_developer("residential",
                        households,
                        buildings,
                        "residential_units",
                        parcels.parcel_size,
                        parcels.ave_res_unit_size,
                        parcels.total_units,
                        feasibility,
                        year=year,
                        target_vacancy=.15,
                        form_to_btype_callback=random_type,
                        add_more_columns_callback=add_extra_columns,
                        bldg_sqft_per_job=400.0)

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
                        target_vacancy=.15,
                        form_to_btype_callback=random_type,
                        add_more_columns_callback=add_extra_columns_non_res,
                        residential=False,
                        bldg_sqft_per_job=400.0)

@orca.step('indicator_export')
def indicator_export(zones,buildings, households, establishments, year, store, zone_to_county):
    utils_drcog.export_indicators(zones, buildings, households, establishments, year, store, zone_to_county)

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
