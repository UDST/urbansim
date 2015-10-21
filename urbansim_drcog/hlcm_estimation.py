__author__ = 'JMartinez'
import pandas as pd
import numpy as np
import orca
import new_dataset
import new_variables
from urbansim.models import SegmentedMNLDiscreteChoiceModel


seg_col = 'income_3_tenure'
sample_size = 100
prob_mode = 'single_chooser'
choice_mode = 'aggregate'
choice_col = 'zone_id'
default_model_expr = 'ln_avg_unit_price_zone + median_age_of_head + median_yearbuilt_post_1990 ' \
                     '+ percent_renter_hh_in_zone + townhome + jobs_within_45min'

remove_alts = False
#alts_filter = ['residential_units > 0']

hlcm = SegmentedMNLDiscreteChoiceModel(segmentation_col=seg_col, sample_size=sample_size,
                                       probability_mode=prob_mode, choice_mode=choice_mode,
                                       choice_column=choice_col, default_model_expr=default_model_expr)


alts = orca.get_table('zones').to_frame()
alts = alts.fillna(0)


### define models
# model1 = 'ln_dist_rail + ln_avg_unit_price_zone + median_age_of_head + median_yearbuilt_post_1990' \
#          ' + percent_hh_with_child_x_hh_with_child + percent_renter_hh_in_zone' \
#          ' +  townhome + multifamily + jobs_within_45min + parks_3mi + golf_courses_3mi + schools_3mi + ' \
#          'fast_food_3mi + restauraunt_3mi + supermarket_3mi + cafes_3mi'
#
# model2 = 'ln_dist_rail + income5xlt_x_avg_unit_price_zone + median_age_of_head + median_yearbuilt_post_1990' \
#          ' + median_yearbuilt_pre_1950 + percent_hh_with_child_x_hh_with_child + percent_renter_hh_in_zone' \
#          ' + multifamily + ln_income_x_average_resunit_size + wkrs_hhs_x_ln_jobs_within_30min + parks_3mi + golf_courses_3mi + schools_3mi + ' \
#          'fast_food_3mi + restauraunt_3mi + supermarket_3mi + cafes_3mi'
#
# model3 = 'ln_dist_rail + income5xlt_x_avg_unit_price_zone + median_age_of_head + mean_income + median_yearbuilt_post_1990' \
#          ' + median_yearbuilt_pre_1950 + ln_income_x_average_resunit_size + percent_renter_hh_in_zone' \
#          ' + cherry_creek_school_district + percent_younghead_x_younghead + ln_jobs_within_30min + parks_3mi + golf_courses_3mi + schools_3mi + ' \
#          'fast_food_3mi + restauraunt_3mi + supermarket_3mi + cafes_3mi'
#
# model4 = 'ln_dist_rail + percent_hh_with_child_x_hh_with_child + percent_renter_hh_in_zone + ' \
#          ' + percent_younghead_x_younghead + ln_emp_sector3_within_20min + allpurpose_agglosum_floor + parks_3mi + golf_courses_3mi + schools_3mi + ' \
#          'fast_food_3mi + restauraunt_3mi + supermarket_3mi + cafes_3mi'
#
# model5 = 'income5xlt_x_avg_unit_price_zone + median_age_of_head + mean_income + median_yearbuilt_post_1990' \
#          ' + percent_hh_with_child_x_hh_with_child + percent_renter_hh_in_zone + townhome + multifamily' \
#          ' + percent_younghead_x_younghead + wkrs_hhs_x_ln_jobs_within_30min + parks_3mi + golf_courses_3mi + schools_3mi + ' \
#          'fast_food_3mi + restauraunt_3mi + supermarket_3mi + cafes_3mi'

model1 = 'ln_avg_unit_price_zone + ' \
         ' + percent_renter_hh_in_zone' \
         ' +  jobs_within_45min + parks_3mi + golf_courses_3mi + schools_3mi + ' \
         'fast_food_3mi + restauraunt_3mi + cafes_3mi + ln_emp_sector3_within_20min + ln_emp_sector5_within_20min'

model2 = 'ln_avg_unit_price_zone + percent_renter_hh_in_zone' \
         ' + parks_3mi + golf_courses_3mi + schools_3mi + ' \
         'fast_food_3mi + restauraunt_3mi + cafes_3mi'

model3 = 'ln_avg_unit_price_zone + percent_renter_hh_in_zone' \
         ' + ln_jobs_within_30min + parks_3mi + golf_courses_3mi + schools_3mi + ' \
         'fast_food_3mi + restauraunt_3mi + cafes_3mi'

model4 = 'ln_avg_unit_price_zone + percent_renter_hh_in_zone + ' \
         ' + ln_emp_sector3_within_20min + parks_3mi + golf_courses_3mi + schools_3mi + ' \
         'fast_food_3mi + restauraunt_3mi + cafes_3mi + ln_emp_sector5_within_20min'

model5 = 'ln_avg_unit_price_zone + ' \
         ' + percent_renter_hh_in_zone' \
         ' + parks_3mi + golf_courses_3mi + schools_3mi + ' \
         'fast_food_3mi + restauraunt_3mi + cafes_3mi'


choosers = orca.get_table('households_for_estimation').to_frame()

###add segments to model

hlcm.add_segment(1, model_expression=model1)
hlcm.add_segment(2, model_expression=model2)
hlcm.add_segment(3, model_expression=model3)
hlcm.add_segment(4, model_expression=model4)
hlcm.add_segment(5, model_expression=model5)


results = hlcm.fit(choosers, alts, current_choice='building_id')
hlcm.to_yaml('c:/urbansim_new/urbansim/urbansim_drcog/config/hlcm_yaml.yaml')