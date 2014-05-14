import pandas as pd
import numpy as np
import time
import os
from urbansim.utils import misc, spotproforma


def get_possible_rents_by_use(dset):
    parcels = dset.parcels
    nodeavgrents = dset.nodes_prices
    price = nodeavgrents.ave_residential_price * -1
    nodeavgrents['ave_residential_rent'] = np.pmt(spotproforma.INTERESTRATE,
                                                  spotproforma.PERIODS,
                                                  price)
    del nodeavgrents['ave_residential_price']

    # need predictions of rents for each parcel
    avgrents = pd.DataFrame(index=parcels.index)
    for btype in ['residential', 'office', 'retail', 'industrial']:
        avgrents[btype] = nodeavgrents['ave_%s_rent' %
                                       btype].ix[parcels._node_id].values
        if btype != 'residential':
            avgrents[btype] *= 1.2
    print avgrents.describe()

    return avgrents


# BIG QUESTION - should the above "possible rents" be greater than the
# here computed actual rents?  probably, right?
def current_rent_per_parcel(far_predictions, avgrents):
    # this is bad - need the right rents for each type
    # my thinking here is that I don't want to go around tearing down
    # buildings to convert to other uses - have to think about
    # this more
    return far_predictions.total_sqft * avgrents.residential * .8


RENTMULTIPLIER = 1.0  # this is essentially a calibration constant
DEV = None


def feasibility_run(dset, year=2010):

    global DEV
    if DEV is None:
        print "Running pro forma"
        DEV = spotproforma.Developer()

    parcels = dset.fetch('parcels').join(
        dset.fetch('zoning_for_parcels'), how="left")

    avgrents = get_possible_rents_by_use(dset)

    # compute total_sqft on the parcel, total current rent, and current far
    far_predictions = pd.DataFrame(index=parcels.index)
    far_predictions['total_sqft'] = dset.buildings.groupby(
        'parcel_id').building_sqft.sum().fillna(0)
    far_predictions['total_units'] = dset.buildings.groupby(
        'parcel_id').residential_units.sum().fillna(0)
    far_predictions['year_built'] = dset.buildings.groupby(
        'parcel_id').year_built.min().fillna(1960)
    far_predictions['currentrent'] = current_rent_per_parcel(
        far_predictions, avgrents)
    far_predictions['parcelsize'] = parcels.shape_area * 10.764
    # some parcels have unrealisticly small sizes
    far_predictions.parcelsize[far_predictions.parcelsize < 300] = 300

    print "Get zoning:", time.ctime()
    zoning = dset.fetch('zoning').dropna(subset=['max_far'])

    # only keeps those with zoning
    parcels = pd.merge(parcels, zoning, left_on='zoning', right_index=True)

    # need to map building types in zoning to allowable forms in the developer
    # model
    type_d = {
        'residential': [1, 2, 3],
        'industrial': [7, 8, 9],
        'retail': [10, 11],
        'office': [4],
        'mixedresidential': [12],
        'mixedoffice': [14],
    }

    # we have zoning by like 16 building types and rents/far predictions by
    # 4 building types so we have to convert one into the other - would
    # probably be better to have rents segmented by the same 16 building
    # types if we had good observations for that
    for form, btypes in type_d.iteritems():

        btypes = type_d[form]
        for btype in btypes:

            print form, btype
            # is type allowed
            tmp = parcels[parcels['type%d' % btype] == 't'][
                ['max_far', 'max_height']]
            # at what far
            far_predictions['type%d_zonedfar' % btype] = tmp['max_far']
            # at what height
            far_predictions['type%d_zonedheight' % btype] = tmp['max_height']

            # need to use max_dua here!!
            if btype == 1:
                far_predictions['type%d_zonedfar' % btype] = .75
            elif btype == 2:
                far_predictions['type%d_zonedfar' % btype] = 1.2

            # do the lookup in the developer model - this is where the
            # profitability is computed
            far_predictions['type%d_feasiblefar' % btype], \
                far_predictions['type%d_profit' % btype] = \
                DEV.lookup(form, avgrents[spotproforma.uses].as_matrix(),
                           far_predictions.currentrent * RENTMULTIPLIER,
                           far_predictions.parcelsize,
                           far_predictions['type%d_zonedfar' % btype],
                           far_predictions['type%d_zonedheight' % btype])

            # don't redevelop historic buildings
            far_predictions['type%d_feasiblefar' % btype][
                far_predictions.year_built < 1945] = 0.0
            far_predictions['type%d_profit' % btype][
                far_predictions.year_built < 1945] = 0.0

    far_predictions = far_predictions.join(avgrents)
    print "Feasibility and zoning\n", far_predictions.describe()
    far_predictions['currentrent'] /= spotproforma.CAPRATE
    fname = os.path.join(misc.data_dir(), 'far_predictions.csv')
    far_predictions.to_csv(fname, index_col='parcel_id', float_format="%.2f")
    dset.save_tmptbl("feasibility", far_predictions)

    print "Finished developer", time.ctime()
