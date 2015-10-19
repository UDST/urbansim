__author__ = 'JMartinez'
import models
import utils_drcog
import orca
import calibration
import numpy as np
from urbansim.utils import sampling

# orca.run([
#           'rsh_simulate',
#           'nrh_simulate',
#           'emp_transition',
#           'emp_relocation',
#           'elcm_simulate',
#           'hh_transition',
#           'hh_relocation',
#           'hlcm_simulate',
#           'feasibility',
#           'residential_developer',
#           'non_residential_developer',
#           'indicator_export'
#           ], iter_vars=[2015])
np.random.seed(1)
orca.run(['calibrate', 'alter_multiplier'], iter_vars=np.linspace(50,130, num=81).tolist())

#orca.run(['emp_transition','emp_relocation', 'elcm_simulate'], iter_vars=[2011])


# orca.run([
#           'feasibility', 'residential_developer', 'non_residential_developer'
#           ], iter_vars=[2011])