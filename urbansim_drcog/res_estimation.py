__author__ = 'JMartinez'
import orca
from urbansim.models import SegmentedRegressionModel
import numpy as np

seg_col = 'building_type_id'
min_seg_size = 150
transform = np.exp
filters_fit = ['improvement_value>20000']
default_model = 'np.log1p(unit_price_residential) ~ ln_pop_within_20min+ln_units_per_acre+mean_income+year_built+ln_dist_bus+' \
                'ln_dist_rail+ln_avg_land_value_per_sqft_zone+ln_residential_unit_density_zone+ln_non_residential_sqft_zone+' \
                'allpurpose_agglosum_floor+county8001+county8005+county8013+county8014+county8019+county8035+county8039+' \
                'county8047+county8059+county8123'

repm = SegmentedRegressionModel(seg_col,fit_filters=filters_fit, predict_filters=filters_fit,
                                default_ytransform=transform, min_segment_size=min_seg_size, default_model_expr=default_model)


buildings = orca.get_table('alts_repm').to_frame()
buildings = buildings[np.in1d(buildings.building_type_id,[2,3,20,24])]
sample_index = np.random.choice(buildings.index, 10000, replace=False)
buildings = buildings.loc[sample_index]


results = repm.fit(buildings)
repm.to_yaml('c:/urbansim_new/urbansim/urbansim_drcog/config/repm_yaml.yaml')