__author__ = 'JMartinez'
import orca
import pandas as pd
import numpy as np
import dataset
from urbansim.utils import misc

#register calculated variables with data table objects

#####################
# PARCEL VARIABLES
#####################


@orca.column('parcels', 'in_denver')
def in_denver(parcels):
    return (parcels.county_id==8031).astype('int32')

@orca.column('parcels','ln_dist_rail')
def ln_dist_rail(parcels):
    return parcels.dist_rail.apply(np.log1p)

@orca.column('parcels', 'ln_dist_bus')
def ln_dist_bus(parcels):
    return parcels.dist_bus.apply(np.log1p)

@orca.column('parcels', 'ln_land_value')
def ln_land_value(parcels):
    return parcels.land_value.apply(np.log1p)

@orca.column('parcels', 'land_value_per_sqft')
def land_value_per_sqft(parcels):
    return (parcels.land_value*1.0/parcels.parcel_sqft)

@orca.column('parcels', 'rail_within_mile')
def rail_within_mile(parcels):
    return (parcels.dist_rail<5280).astype('int32')

@orca.column('parcels', 'cherry_creek_school_district')
def cherry_creek_school_district(parcels):
    return (parcels.school_district == 8).astype('int32')

@orca.column('parcels', 'acres')
def acres(parcels):
    return parcels.parcel_sqft/43560.0

@orca.column('parcels', 'ln_acres')
def ln_acres(parcels):
    return (parcels.parcel_sqft/43560.0).apply(np.log1p)

