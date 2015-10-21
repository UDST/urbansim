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

# orca.run(['calibrate', 'alter_multiplier'], iter_vars=np.linspace(0,1, num=11).tolist())

orca.run(['emp_transition','emp_relocation'], iter_vars=[2015])


# orca.run([
#           'feasibility', 'residential_developer', 'non_residential_developer'
#           ], iter_vars=[2011])