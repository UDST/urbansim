__author__ = 'JMartinez'
import pandas as pd
import numpy as np
import orca
from urbansim_drcog import new_dataset
from urbansim_drcog import new_variables
from urbansim.models import MNLDiscreteChoiceModel

print "Loading alts"
alts = orca.get_table('zones').to_frame(columns=[
    'ln_avg_unit_price_zone',
    'nb_pop',
    'nb_emp',
    'ln_residential_unit_density_zone',
    'ln_non_residential_sqft_zone',
    'mean_income',
    'percent_hh_with_child',
    'percent_renter_hh_in_zone',
    'percent_younghead'
])
alts = alts.fillna(0)

#choosers = orca.merge_tables('households', tables=['households','buildings','parcels'])
print "Loading choosers"
choosers = orca.get_table('households_for_estimation').to_frame()

rhs = "ln_avg_unit_price_zone +nb_pop + nb_emp + ln_residential_unit_density_zone + ln_non_residential_sqft_zone +" \
      "mean_income + percent_hh_with_child + percent_renter_hh_in_zone + percent_younghead"

sample_size = 100

probability_mode = "single_chooser"

choice_mode = "aggregate"

elcm = MNLDiscreteChoiceModel(rhs, sample_size=sample_size, probability_mode=probability_mode, choice_mode=choice_mode,
                              choice_column='zone_id', name='Zonal ELCM Model')

print "fitting model"
results = elcm.fit(choosers, alts, current_choice='zone_id')
elcm.to_yaml('c:/urbansim_new/urbansim/urbansim_drcog/config/zonal_hlcm_yaml.yaml')