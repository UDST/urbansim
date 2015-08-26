__author__ = 'JMartinez'
import pandas as pd
import numpy as np
import orca
import dataset
import variables
from urbansim.models import SegmentedMNLDiscreteChoiceModel

###create estimation table
establishments = orca.get_table('establishments').to_frame()
sample_index = np.random.choice(establishments.index, 3000, replace=False)
establishments = establishments.loc[sample_index]
establishments_for_estimation = establishments[(establishments.building_id>0)&(establishments.home_based_status==0)&(establishments.nonres_sqft>0)]


###define model parameters

sample_size = 100
seg_col = 'sector_id_retail_agg'
prob_mode = 'single_chooser'
choice_mode = 'aggregate'
choice_col = 'building_id'
default_model = 'ln_jobs_within_30min+ln_avg_nonres_unit_price_zone+median_year_built+ln_residential_unit_density_zone' \
                '+ln_pop_within_20min+nonres_far+office+retail_or_restaurant+industrial_building' \
                '+employees_x_ln_non_residential_sqft_zone+ln_dist_rail+rail_within_mile'

alts_filter = ['non_residential_sqft > 0']

remove_alts = False

elcm = SegmentedMNLDiscreteChoiceModel(segmentation_col=seg_col, sample_size=sample_size,
                                       probability_mode=prob_mode, choice_mode=choice_mode,
                                       choice_column=choice_col, default_model_expr=default_model,
                                       alts_fit_filters=alts_filter)


alts = pd.DataFrame(index=orca.get_table('buildings').building_type_id.index)
choosers = establishments_for_estimation

##add columns to alts table for RHS of equation
alts.loc[:, 'non_residential_sqft'] = orca.get_table('buildings').non_residential_sqft
alts.loc[:, 'ln_jobs_within_30min'] = orca.get_table('buildings').ln_jobs_within_30min
alts.loc[:, 'ln_avg_nonres_unit_price_zone'] = orca.get_table('buildings').ln_avg_nonres_unit_price_zone
alts.loc[:, 'median_year_built'] = orca.get_table('buildings').median_year_built
alts.loc[:, 'ln_residential_unit_density_zone'] = orca.get_table('buildings').ln_residential_unit_density_zone
alts.loc[:, 'ln_pop_within_20min'] = orca.get_table('buildings').ln_pop_within_20min
alts.loc[:, 'nonres_far'] = orca.get_table('buildings').nonres_far
alts.loc[:, 'office'] = orca.get_table('buildings').office
alts.loc[:, 'retail_or_restaurant'] = orca.get_table('buildings').retail_or_restaurant
alts.loc[:, 'industrial_building'] = orca.get_table('buildings').industrial_building
alts.loc[:, 'employees_x_ln_non_residential_sqft_zone'] = orca.get_table('buildings').employees_x_ln_non_residential_sqft_zone
alts.loc[:, 'ln_dist_rail'] = orca.get_table('buildings').ln_dist_rail
alts.loc[:, 'rail_within_mile'] = orca.get_table('buildings').rail_within_mile
alts.loc[:, 'ln_emp_sector1_within_15min'] = orca.get_table('buildings').ln_emp_sector1_within_15min
alts.loc[:, 'ln_emp_sector2_within_15min'] = orca.get_table('buildings').ln_emp_sector2_within_15min
alts.loc[:, 'ln_emp_sector3_within_15min'] = orca.get_table('buildings').ln_emp_sector3_within_15min
alts.loc[:, 'ln_emp_sector4_within_15min'] = orca.get_table('buildings').ln_emp_sector4_within_15min
alts.loc[:, 'ln_emp_sector5_within_15min'] = orca.get_table('buildings').ln_emp_sector5_within_15min
alts.loc[:, 'ln_emp_sector6_within_15min'] = orca.get_table('buildings').ln_emp_sector6_within_15min


alts.ln_jobs_within_30min.fillna(0, inplace=True)
alts.ln_avg_nonres_unit_price_zone.fillna(0, inplace=True)
alts.median_year_built.fillna(0, inplace=True)
alts.ln_residential_unit_density_zone.fillna(0, inplace=True)
alts.ln_pop_within_20min.fillna(0, inplace=True)
alts.nonres_far.fillna(0, inplace=True)
alts.office.fillna(0, inplace=True)
alts.retail_or_restaurant.fillna(0, inplace=True)
alts.industrial_building.fillna(0, inplace=True)
alts.employees_x_ln_non_residential_sqft_zone.fillna(0, inplace=True)
alts.ln_dist_rail.fillna(0, inplace=True)
alts.rail_within_mile.fillna(0, inplace=True)
alts.ln_emp_sector1_within_15min.fillna(0, inplace=True)
alts.ln_emp_sector2_within_15min.fillna(0, inplace=True)
alts.ln_emp_sector3_within_15min.fillna(0, inplace=True)
alts.ln_emp_sector4_within_15min.fillna(0, inplace=True)
alts.ln_emp_sector5_within_15min.fillna(0, inplace=True)
alts.ln_emp_sector6_within_15min.fillna(0, inplace=True)


##define models
model11 = 'ln_jobs_within_30min+ln_avg_nonres_unit_price_zone+median_year_built+ln_residential_unit_density_zone' \
                '+ln_pop_within_20min+nonres_far+office+retail_or_restaurant+industrial_building' \
                '+employees_x_ln_non_residential_sqft_zone+ln_dist_rail+rail_within_mile'

model21 = 'ln_jobs_within_30min+ln_avg_nonres_unit_price_zone+median_year_built+ln_residential_unit_density_zone' \
                '+ln_pop_within_20min+nonres_far+office+retail_or_restaurant+industrial_building' \
                '+employees_x_ln_non_residential_sqft_zone+ln_emp_sector3_within_15min+rail_within_mile'

model22 = 'ln_jobs_within_30min+ln_avg_nonres_unit_price_zone+median_year_built+ln_residential_unit_density_zone' \
                '+ln_pop_within_20min+nonres_far+office+retail_or_restaurant+industrial_building' \
                '+employees_x_ln_non_residential_sqft_zone+ln_emp_sector3_within_15min+rail_within_mile'

model23 = 'ln_jobs_within_30min+ln_avg_nonres_unit_price_zone+median_year_built+ln_residential_unit_density_zone' \
                '+ln_pop_within_20min+nonres_far+office+retail_or_restaurant+industrial_building' \
                '+employees_x_ln_non_residential_sqft_zone+ln_emp_sector3_within_15min+rail_within_mile'

model31 = 'ln_jobs_within_30min+ln_avg_nonres_unit_price_zone+median_year_built+ln_residential_unit_density_zone' \
                '+ln_pop_within_20min+nonres_far+office+retail_or_restaurant+industrial_building' \
                '+employees_x_ln_non_residential_sqft_zone+ln_emp_sector3_within_15min+rail_within_mile'

model32 = 'ln_jobs_within_30min+ln_avg_nonres_unit_price_zone+median_year_built+ln_residential_unit_density_zone' \
                '+ln_pop_within_20min+nonres_far+office+retail_or_restaurant+industrial_building' \
                '+employees_x_ln_non_residential_sqft_zone+ln_emp_sector3_within_15min+rail_within_mile'

model33 = 'ln_jobs_within_30min+ln_avg_nonres_unit_price_zone+median_year_built+ln_residential_unit_density_zone' \
                '+ln_pop_within_20min+nonres_far+office+retail_or_restaurant+industrial_building' \
                '+employees_x_ln_non_residential_sqft_zone+ln_emp_sector3_within_15min+rail_within_mile'

model42 = 'ln_jobs_within_30min+ln_avg_nonres_unit_price_zone+median_year_built+ln_residential_unit_density_zone' \
                '+ln_pop_within_20min+nonres_far+office+retail_or_restaurant+industrial_building' \
                '+employees_x_ln_non_residential_sqft_zone+ln_emp_sector2_within_15min+rail_within_mile'

model44 = 'ln_jobs_within_30min+ln_avg_nonres_unit_price_zone+median_year_built+ln_residential_unit_density_zone' \
                '+ln_pop_within_20min+nonres_far+office+retail_or_restaurant+industrial_building' \
                '+employees_x_ln_non_residential_sqft_zone+ln_emp_sector5_within_15min+rail_within_mile'

model45 = 'ln_jobs_within_30min+ln_avg_nonres_unit_price_zone+median_year_built+ln_residential_unit_density_zone' \
                '+ln_pop_within_20min+nonres_far+office+retail_or_restaurant+industrial_building' \
                '+employees_x_ln_non_residential_sqft_zone+ln_emp_sector5_within_15min+rail_within_mile'

model48 = 'ln_jobs_within_30min+ln_avg_nonres_unit_price_zone+median_year_built+ln_residential_unit_density_zone' \
                '+ln_pop_within_20min+nonres_far+office+retail_or_restaurant+industrial_building' \
                '+employees_x_ln_non_residential_sqft_zone+ln_emp_sector3_within_15min+rail_within_mile'

model49 = 'ln_jobs_within_30min+ln_avg_nonres_unit_price_zone+median_year_built+ln_residential_unit_density_zone' \
                '+ln_pop_within_20min+nonres_far+office+retail_or_restaurant+industrial_building' \
                '+employees_x_ln_non_residential_sqft_zone+ln_emp_sector3_within_15min+rail_within_mile'

model51 = 'ln_jobs_within_30min+ln_avg_nonres_unit_price_zone+median_year_built+ln_residential_unit_density_zone' \
                '+ln_pop_within_20min+nonres_far+office+retail_or_restaurant+industrial_building' \
                '+employees_x_ln_non_residential_sqft_zone+ln_emp_sector6_within_15min+rail_within_mile'

model52 = 'ln_jobs_within_30min+ln_avg_nonres_unit_price_zone+median_year_built+ln_residential_unit_density_zone' \
                '+ln_pop_within_20min+nonres_far+office+retail_or_restaurant+industrial_building' \
                '+employees_x_ln_non_residential_sqft_zone+ln_emp_sector6_within_15min+rail_within_mile'

model53 = 'ln_jobs_within_30min+ln_avg_nonres_unit_price_zone+median_year_built+ln_residential_unit_density_zone' \
                '+ln_pop_within_20min+nonres_far+office+retail_or_restaurant+industrial_building' \
                '+employees_x_ln_non_residential_sqft_zone+ln_emp_sector6_within_15min+rail_within_mile'

model54 = 'ln_jobs_within_30min+ln_avg_nonres_unit_price_zone+median_year_built+ln_residential_unit_density_zone' \
                '+ln_pop_within_20min+nonres_far+office+retail_or_restaurant+industrial_building' \
                '+employees_x_ln_non_residential_sqft_zone+ln_emp_sector6_within_15min+rail_within_mile'

model55 = 'ln_jobs_within_30min+ln_avg_nonres_unit_price_zone+median_year_built+ln_residential_unit_density_zone' \
                '+ln_pop_within_20min+nonres_far+office+retail_or_restaurant+industrial_building' \
                '+employees_x_ln_non_residential_sqft_zone+ln_emp_sector6_within_15min+rail_within_mile'

model56 = 'ln_jobs_within_30min+ln_avg_nonres_unit_price_zone+median_year_built+ln_residential_unit_density_zone' \
                '+ln_pop_within_20min+nonres_far+office+retail_or_restaurant+industrial_building' \
                '+employees_x_ln_non_residential_sqft_zone+ln_emp_sector6_within_15min+rail_within_mile'

model61 = 'ln_jobs_within_30min+ln_avg_nonres_unit_price_zone+median_year_built+ln_residential_unit_density_zone' \
                '+ln_pop_within_20min+nonres_far+office+retail_or_restaurant+industrial_building' \
                '+employees_x_ln_non_residential_sqft_zone+ln_emp_sector6_within_15min+rail_within_mile'

model62 = 'ln_jobs_within_30min+ln_avg_nonres_unit_price_zone+median_year_built+ln_residential_unit_density_zone' \
                '+ln_pop_within_20min+nonres_far+office+retail_or_restaurant+industrial_building' \
                '+employees_x_ln_non_residential_sqft_zone+ln_emp_sector6_within_15min+rail_within_mile'

model71 = 'ln_jobs_within_30min+ln_avg_nonres_unit_price_zone+median_year_built+ln_residential_unit_density_zone' \
                '+ln_pop_within_20min+nonres_far+office+retail_or_restaurant+industrial_building' \
                '+employees_x_ln_non_residential_sqft_zone+ln_emp_sector2_within_15min+rail_within_mile'

model81 = 'ln_jobs_within_30min+ln_avg_nonres_unit_price_zone+median_year_built+ln_residential_unit_density_zone' \
                '+ln_pop_within_20min+nonres_far+office+retail_or_restaurant+industrial_building' \
                '+employees_x_ln_non_residential_sqft_zone+ln_emp_sector6_within_15min+rail_within_mile'

model92 = 'ln_jobs_within_30min+ln_avg_nonres_unit_price_zone+median_year_built+ln_residential_unit_density_zone' \
                '+ln_pop_within_20min+nonres_far+office+retail_or_restaurant+industrial_building' \
                '+employees_x_ln_non_residential_sqft_zone+ln_emp_sector6_within_15min+rail_within_mile'
model7211 = 'ln_jobs_within_30min+ln_avg_nonres_unit_price_zone+median_year_built+ln_residential_unit_density_zone' \
                '+ln_pop_within_20min+nonres_far+office+retail_or_restaurant+industrial_building' \
                '+employees_x_ln_non_residential_sqft_zone+ln_emp_sector5_within_15min+rail_within_mile'

model7221 = 'ln_jobs_within_30min+ln_avg_nonres_unit_price_zone+median_year_built+ln_residential_unit_density_zone' \
                '+ln_pop_within_20min+nonres_far+office+retail_or_restaurant+industrial_building' \
                '+employees_x_ln_non_residential_sqft_zone+ln_emp_sector4_within_15min+rail_within_mile'

model7222 = 'ln_jobs_within_30min+ln_avg_nonres_unit_price_zone+median_year_built+ln_residential_unit_density_zone' \
                '+ln_pop_within_20min+nonres_far+office+retail_or_restaurant+industrial_building' \
                '+employees_x_ln_non_residential_sqft_zone+ln_emp_sector4_within_15min+rail_within_mile'

model7223 = 'ln_jobs_within_30min+ln_avg_nonres_unit_price_zone+median_year_built+ln_residential_unit_density_zone' \
                '+ln_pop_within_20min+nonres_far+office+retail_or_restaurant+industrial_building' \
                '+employees_x_ln_non_residential_sqft_zone+ln_emp_sector5_within_15min+rail_within_mile'

model7224 = 'ln_jobs_within_30min+ln_avg_nonres_unit_price_zone+median_year_built+ln_residential_unit_density_zone' \
                '+ln_pop_within_20min+nonres_far+office+retail_or_restaurant+industrial_building' \
                '+employees_x_ln_non_residential_sqft_zone+ln_emp_sector4_within_15min+rail_within_mile'


elcm.add_segment(11, model_expression=model11)
elcm.add_segment(21, model_expression=model21)
elcm.add_segment(22, model_expression=model22)
elcm.add_segment(23, model_expression=model23)
elcm.add_segment(31, model_expression=model31)
elcm.add_segment(32, model_expression=model32)
elcm.add_segment(33, model_expression=model33)
elcm.add_segment(42, model_expression=model42)
elcm.add_segment(44, model_expression=model44)
elcm.add_segment(45, model_expression=model45)
elcm.add_segment(48, model_expression=model48)
elcm.add_segment(49, model_expression=model49)
elcm.add_segment(51, model_expression=model51)
elcm.add_segment(52, model_expression=model52)
elcm.add_segment(53, model_expression=model53)
elcm.add_segment(54, model_expression=model54)
elcm.add_segment(55, model_expression=model55)
elcm.add_segment(56, model_expression=model56)
elcm.add_segment(61, model_expression=model61)
elcm.add_segment(62, model_expression=model62)
elcm.add_segment(71, model_expression=model71)
elcm.add_segment(81, model_expression=model81)
elcm.add_segment(92, model_expression=model92)
elcm.add_segment(7211, model_expression=model7211)
elcm.add_segment(7221, model_expression=model7221)
elcm.add_segment(7223, model_expression=model7223)
elcm.add_segment(7224, model_expression=model7224)

results = elcm.fit(choosers, alts, current_choice='building_id')
elcm.to_yaml('c:/users/jmartinez/documents/elcm_yaml.yaml')