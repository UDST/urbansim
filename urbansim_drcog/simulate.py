__author__ = 'JMartinez'
import models
import utils_drcog
import orca
import calibration
import numpy as np
from urbansim.utils import sampling
np.random.seed(1)
# orca.run([
#           'rsh_simulate',
#           'nrh_simulate',
#           'feasibility',
#           'residential_developer',
#           'non_residential_developer',
#           'emp_transition',
#           'emp_relocation',
#           'elcm_simulate',
#           'hh_transition'
#           'hh_relocation',
#           'hlcm_simulate',
#
#           'indicator_export
#           ], iter_vars=[2040])


# orca.run(['emp_transition',
#           'emp_relocation',
#           'elcm_simulate',
#           'hh_transition',
#           'hh_relocation',
#           'hlcm_simulate',
#           'indicator_export',
#           'rsh_simulate',
#           'nrh_simulate',
#           'feasibility',
#           'residential_developer',
#           'non_residential_developer',
#           ], iter_vars=[2040])

orca.run(['hh_transition','hh_relocation','hlcm_simulate','res_supply_demand'], iter_vars=[2040])


# orca.run([
#           'feasibility', 'residential_developer', 'non_residential_developer'
#           ], iter_vars=[2011])