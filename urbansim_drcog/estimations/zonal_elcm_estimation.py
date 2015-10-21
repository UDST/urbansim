__author__ = 'JMartinez'
import pandas as pd
import numpy as np
import orca
from urbansim_drcog import new_dataset
from urbansim_drcog import new_variables
from urbansim.models import MNLDiscreteChoiceModel

alts = orca.get_table('zones').to_frame(columns=[
    'ln_avg_nonres_unit_price_zone',
    'nb_pop',
    'nb_emp',
    'ln_residential_unit_density_zone',
    'ln_non_residential_sqft_zone',
    'percent_sf'
])
alts = alts.fillna(0)

choosers = orca.merge_tables('establishments', tables=['establishments','buildings','parcels'])

rhs = "ln_avg_nonres_unit_price_zone +nb_pop + nb_emp + ln_residential_unit_density_zone + ln_non_residential_sqft_zone +" \
      "percent_sf"

sample_size = 100

probability_mode = "single_chooser"

choice_mode = "aggregate"

elcm = MNLDiscreteChoiceModel(rhs, sample_size=sample_size, probability_mode=probability_mode, choice_mode=choice_mode,
                              choice_column='zone_id', name='Zonal ELCM Model')

results = elcm.fit(choosers, alts, current_choice='zone_id')
elcm.to_yaml('c:/urbansim_new/urbansim/urbansim_drcog/config/zonal_elcm_yaml.yaml')