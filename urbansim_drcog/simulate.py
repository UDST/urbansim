__author__ = 'JMartinez'
import models
import utils_drcog
import orca
import numpy as np
from urbansim.utils import sampling

orca.run(['rsh_simulate',
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
          'indicator_export'], iter_vars=[2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025,
                                          2026,2027,2028,2029,2030,2031,2032,2033,2034,2035,2036,2037,2038,2039,2040])

# orca.run(['feasibility','non_residential_developer'], iter_vars=[2011])