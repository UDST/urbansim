__author__ = 'JMartinez'
import models
import utils_drcog
import orca
import numpy as np
from urbansim.utils import sampling

# orca.run(['rsh_simulate',
#           'nrh_simulate',
#           'emp_transition',
#           'emp_relocation',
#           'elcm_simulate',
#           'hh_transition',
#           'hh_relocation',
#           'hlcm_simulate',
#           'feasibility',
#           'residential_developer',
#           'non_residential_developer'], iter_vars=[2011])

orca.run(['feasibility','residential_developer'], iter_vars=[2011])