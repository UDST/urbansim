__author__ = 'JMartinez'
import models
import utils_drcog
import orca
import numpy as np
import pandas as pd
from urbansim.utils import sampling



#set threshold
thresh = 250
base_data = pd.read_csv('C:/users/jmartinez/documents/projects/urbansim/results/zone_summary09032015.csv', index_col='zone_id')
year = 2015
init_slope = 158

#for i in range(10):

#
orca.run([
          'rsh_simulate',
          'nrh_simulate',
          'emp_transition',
          'emp_relocation',
          'elcm_simulate',
          'hh_transition',
          'hh_relocation',
          'hlcm_simulate',
          'feasibility',
          'residential_developer',
          'non_residential_developer',
          'indicator_export',

          ], iter_vars=[2015])

# orca.run(['emp_transition', 'emp_relocation','elcm_simulate'], iter_vars=[2015])


#test if current results exceed threshold
emp_diff = base_data.loc[base_data.sim_year == 2015].emp_sim.fillna(0).sub(orca.get_table('zone_summary').emp_sim.fillna(0))
hh_diff = base_data.loc[base_data.sim_year == 2015].hh_sim.fillna(0).sub(orca.get_table('zone_summary').hh_sim.fillna(0))

pct_emp_diff = 1+ base_data.loc[base_data.sim_year == 2015].emp_sim.fillna(0).sub(orca.get_table('zone_summary').emp_sim.fillna(0))/orca.get_table('zone_summary').emp_sim.fillna(0)
pct_hh_diff = 1+base_data.loc[base_data.sim_year == 2015].hh_sim.fillna(0).sub(orca.get_table('zone_summary').hh_sim.fillna(0))/orca.get_table('zone_summary').hh_sim.fillna(0)

emp_thresh = emp_diff.loc[emp_diff.abs() > thresh]
hh_thresh = hh_diff.loc[hh_diff.abs() > thresh]

#for records that exceed threshold, update multiplier
mult = orca.get_table('multipliers').to_frame()
hh_mult = pct_hh_diff.loc[hh_diff.abs() >= thresh]
emp_mult = pct_emp_diff.loc[emp_diff.abs() >= thresh]
mult.loc[hh_mult.index, "hh_multiplier"] = hh_mult.values
mult.loc[emp_mult.index, "emp_multiplier"] = emp_mult.values

#@orca.add_table('multipliers', mult, cache=True, cache_scope="iteration")
#mult.to_csv('c:/urbansim_new/urbansim/urbansim_drcog/config/new_multipliers.csv')

orca.get_table('zone_summary').to_frame().to_csv('c:/users/jmartinez/documents/calibration_test_zones.csv')
orca.get_table('county_summary').to_frame().to_csv('c:/users/jmartinez/documents/calibration_test_county.csv')

#calculate difference from simulated to base



