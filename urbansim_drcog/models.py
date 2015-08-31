__author__ = 'JMartinez'
import random
import orca
import dataset
import utils_drcog
import variables


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
    return utils_drcog.lcm_simulate('c:/urbansim_new/urbansim/urbansim_drcog/config/hlcm_yaml.yaml',
                                    establishments, buildings, 'building_id', 'non_residential_units',
                                    'vacant_job_spaces')

@orca.step('hh_relocation')
def hh_relocation(households, household_relocation_rates):
    return utils_drcog.hh_relocation_model(households, household_relocation_rates, 'building_id')

@orca.step('hh_transition')
def hh_transition(households, household_control_totals, year):
    return utils_drcog.hh_transition(households,household_control_totals, 'building_id', year)

@orca.step('emp_transition')
def emp_transition(establishments, employment_control_totals, year):
    return utils_drcog.emp_transition(establishments, employment_control_totals, 'building_id', year)