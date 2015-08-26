__author__ = 'JMartinez'
import pandas as pd
import numpy as np
import orca
import dataset
import variables
from urbansim.models import SegmentedMNLDiscreteChoiceModel


seg_col = 'income_3_tenure'
sample_size = 100
prob_mode = 'single_chooser'
choice_mode = 'aggregate'
choice_col = 'building_id'
default_model_expr = 'ln_dist_rail + ln_avg_unit_price_zone + median_age_of_head + median_yearbuilt_post_1990 ' \
                     '+ percent_renter_hh_in_zone + multifamily + townhome + jobs_within_45min'

remove_alts = False

hlcm = SegmentedMNLDiscreteChoiceModel(segmentation_col=seg_col, sample_size=sample_size,
                                       probability_mode=prob_mode, choice_mode=choice_mode,
                                       choice_column=choice_col, default_model_expr=default_model_expr)

alts = pd.DataFrame(index=orca.get_table('buildings').building_type_id.index)

##add columns to alts table for RHS of equation
alts.loc[:, 'residential_units'] = orca.get_table('buildings').residential_units
alts.loc[:, 'ln_dist_rail'] = orca.get_table('buildings').ln_dist_rail
alts.loc[:, 'ln_avg_unit_price_zone'] = orca.get_table('buildings').ln_avg_unit_price_zone
alts.loc[:, 'median_age_of_head'] = orca.get_table('buildings').median_age_of_head
alts.loc[:, 'median_yearbuilt_post_1990'] = orca.get_table('buildings').median_yearbuilt_post_1990
alts.loc[:, 'percent_hh_with_child_x_hh_with_child'] = orca.get_table('buildings').percent_hh_with_child_x_hh_with_child
alts.loc[:, 'percent_renter_hh_in_zone'] = orca.get_table('buildings').percent_renter_hh_in_zone
alts.loc[:, 'townhome'] = orca.get_table('buildings').townhome
alts.loc[:, 'multifamily'] = orca.get_table('buildings').multifamily
alts.loc[:, 'jobs_within_45min'] = orca.get_table('buildings').jobs_within_45min
alts.loc[:, 'income5xlt_x_avg_unit_price_zone'] = orca.get_table('buildings').income5xlt_x_avg_unit_price_zone
alts.loc[:, 'median_yearbuilt_pre_1950'] = orca.get_table('buildings').median_yearbuilt_pre_1950
alts.loc[:, 'ln_income_x_average_resunit_size'] = orca.get_table('buildings').ln_income_x_average_resunit_size
alts.loc[:, 'wkrs_hhs_x_ln_jobs_within_30min'] = orca.get_table('buildings').wkrs_hhs_x_ln_jobs_within_30min
alts.loc[:, 'mean_income'] = orca.get_table('buildings').mean_income
alts.loc[:, 'cherry_creek_school_district'] = orca.get_table('buildings').cherry_creek_school_district
alts.loc[:, 'percent_younghead_x_younghead'] = orca.get_table('buildings').percent_younghead_x_younghead
alts.loc[:, 'ln_jobs_within_30min'] = orca.get_table('buildings').ln_jobs_within_30min
alts.loc[:, 'ln_emp_sector3_within_20min'] = orca.get_table('buildings').ln_emp_sector3_within_20min
alts.loc[:, 'allpurpose_agglosum_floor'] = orca.get_table('buildings').allpurpose_agglosum_floor

alts = alts.loc[alts.residential_units > 0]
###drop nas
alts.ln_dist_rail.fillna(0, inplace=True)
alts.ln_avg_unit_price_zone.fillna(0, inplace=True)
alts.median_age_of_head.fillna(0, inplace=True)
alts.median_yearbuilt_post_1990.fillna(0, inplace=True)
alts.percent_hh_with_child_x_hh_with_child.fillna(0, inplace=True)
alts.percent_renter_hh_in_zone.fillna(0, inplace=True)
alts.townhome.fillna(0, inplace=True)
alts.multifamily.fillna(0, inplace=True)
alts.jobs_within_45min.fillna(0, inplace=True)
alts.income5xlt_x_avg_unit_price_zone.fillna(0, inplace=True)
alts.median_yearbuilt_pre_1950.fillna(0, inplace=True)
alts.ln_income_x_average_resunit_size.fillna(0, inplace=True)
alts.wkrs_hhs_x_ln_jobs_within_30min.fillna(0, inplace=True)
alts.mean_income.fillna(0, inplace=True)
alts.cherry_creek_school_district.fillna(0, inplace=True)
alts.percent_younghead_x_younghead.fillna(0, inplace=True)
alts.ln_jobs_within_30min.fillna(0, inplace=True)
alts.ln_emp_sector3_within_20min.fillna(0, inplace=True)
alts.allpurpose_agglosum_floor.fillna(0, inplace=True)


### define models
model1 = 'ln_dist_rail + ln_avg_unit_price_zone + median_age_of_head + median_yearbuilt_post_1990' \
         ' + percent_hh_with_child_x_hh_with_child + percent_renter_hh_in_zone' \
         ' +  townhome + multifamily + jobs_within_45min'

model2 = 'ln_dist_rail + income5xlt_x_avg_unit_price_zone + median_age_of_head + median_yearbuilt_post_1990' \
         ' + median_yearbuilt_pre_1950 + percent_hh_with_child_x_hh_with_child + percent_renter_hh_in_zone' \
         ' + multifamily + ln_income_x_average_resunit_size + wkrs_hhs_x_ln_jobs_within_30min'

model3 = 'ln_dist_rail + income5xlt_x_avg_unit_price_zone + median_age_of_head + mean_income + median_yearbuilt_post_1990' \
         ' + median_yearbuilt_pre_1950 + ln_income_x_average_resunit_size + percent_renter_hh_in_zone' \
         ' + cherry_creek_school_district + percent_younghead_x_younghead + ln_jobs_within_30min'

model4 = 'ln_dist_rail + percent_hh_with_child_x_hh_with_child + percent_renter_hh_in_zone + ' \
         ' + percent_younghead_x_younghead + ln_emp_sector3_within_20min + allpurpose_agglosum_floor'

model5 = 'income5xlt_x_avg_unit_price_zone + median_age_of_head + mean_income + median_yearbuilt_post_1990' \
         ' + percent_hh_with_child_x_hh_with_child + percent_renter_hh_in_zone + townhome + multifamily' \
         ' + percent_younghead_x_younghead + wkrs_hhs_x_ln_jobs_within_30min'


choosers = orca.get_table('households_for_estimation').to_frame()

###add segments to model
test = orca.get_table('households_for_estimation').to_frame().groupby('income_3_tenure')

hlcm.add_segment(1, model_expression=model1)
hlcm.add_segment(2, model_expression=model2)
hlcm.add_segment(3, model_expression=model3)
hlcm.add_segment(4, model_expression=model4)
hlcm.add_segment(5, model_expression=model5)


results = hlcm.fit(choosers, alts, current_choice='building_id')
hlcm.to_yaml('c:/users/jmartinez/documents/hlcm_yaml.yaml')