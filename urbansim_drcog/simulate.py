__author__ = 'JMartinez'
import models
import utils_drcog
import orca
import numpy as np
from urbansim.utils import sampling

#orca.run(['emp_transition','hh_transition','hh_relocation','hlcm_simulate'], iter_vars=[2011])

hh = orca.get_table('establishments').to_frame()
prob_dist = np.absolute(np.random.randn(hh.shape[0]))
prob = prob_dist / prob_dist.sum()
result = sampling.sample_rows(1500, hh, accounting_column='employees', replace=False)
print result.employees.sum()

# b = orca.get_table('buildings').to_frame(['zone_id','vacant_residential_units', 'residential_units'])
#
#
# vacant = b.groupby('zone_id').vacant_residential_units.sum().loc[[3,2632,10,2622,11,2,2631,6,13]]
#
# total = b.groupby('zone_id').residential_units.sum().loc[[3,2632,10,2622,11,2,2631,6,13 ]]
#
# print vacant
# print total
# print vacant/total