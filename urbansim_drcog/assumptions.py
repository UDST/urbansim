__author__ = 'JMartinez'
import pandas as pd
import numpy as np
import orca
from urbansim.utils import misc


orca.add_injectable('use_btype',
                    {'retail': [17, 18],
                     'office': [5],
                     'industrial': [9,22],
                     'residential': [2,3,20,24]})

orca.add_injectable('form_to_btype',
                    {'retail': [17, 18],
                     'office': [5],
                     'industrial': [9,22],
                     'residential': [2,3,20,24],
                     'mixedresidential':[6],
                     'mixedoffice':[11]})


def parcel_avg_price(use):
    #if use is residential translate unit price to price per sqft
    buildings = orca.get_table('buildings')
    use_btype = orca.get_injectable('use_btype')
    if use == 'residential':
        price = (buildings.unit_price_residential.loc[np.in1d(buildings.building_type_id, use_btype[use])] /
                 buildings.residential_sqft.loc[np.in1d(buildings.building_type_id, use_btype[use])]).groupby(buildings.zone_id).mean()
    else:
        price = buildings.unit_price_non_residential.loc[np.in1d(buildings.building_type_id, use_btype[use])].groupby(buildings.zone_id).mean()

    return misc.reindex(price, orca.get_table('parcels').zone_id)


def parcel_is_allowed(form):
    form_to_btype = orca.get_injectable("form_to_btype")
    # we have zoning by building type but want
    # to know if specific forms are allowed
    allowed = [orca.get_table('zoning_baseline')
               ['type%d' % typ] == 1 for typ in form_to_btype[form]]
    return pd.concat(allowed, axis=1).max(axis=1).\
        reindex(orca.get_table('parcels').index).fillna(False)
