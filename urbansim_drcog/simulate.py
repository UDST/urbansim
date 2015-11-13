__author__ = 'JMartinez'
import models
import utils_drcog
import orca
import calibration
import numpy as np
from urbansim.utils import sampling
np.random.seed(1)
orca.run([



          'scenario_zoning_change',
          'feasibility',
          'residential_developer',
          'non_residential_developer',
          'emp_transition',
          'emp_relocation',
          'elcm_simulate',
          'hh_transition',
          'hh_relocation',
          'hlcm_simulate',
          'households_to_buildings',
          'indicator_export'
          ], iter_vars=[2040])


# orca.run(['emp_transition',
#           'emp_relocation',
#           'elcm_simulate',
#           'hh_transition',
#           'hh_relocation',
#           'hlcm_simulate',
#
#           'indicator_export',
#           ], iter_vars=[2040])

#orca.run(['hh_transition','hh_relocation','hlcm_simulate','res_supply_demand'], iter_vars=[2040])


# orca.run([
#           'feasibility', 'residential_developer', 'non_residential_developer'
#           ], iter_vars=[2011])