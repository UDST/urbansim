import time

import numpy as np
import pandas as pd
from numpy import array as arr

pd.set_option('display.precision', 3)

# these are the parcel sizes we test, turns out nothing is dependent on
# size right now
parcelsizes = np.array([10000.0])
# these ar ethe fars we test
fars = np.array(
    [.25, .5, .75, 1.0, 1.5, 1.8, 2.0, 3.0, 4.0, 5.0, 7.0, 9.0, 11.0])

# these are the uses we test and the mixes (forms) of those uses
uses = ['retail', 'industrial', 'office', 'residential']
forms = {
    'retail': arr([1.0, 0.0, 0.0, 0.0]),
    'industrial': arr([0.0, 1.0, 0.0, 0.0]),
    'office': arr([0.0, 0.0, 1.0, 0.0]),
    'residential': arr([0.0, 0.0, 0.0, 1.0]),
    'mixedresidential': arr([0.1, 0.0, 0.0, 0.9]),
    'mixedoffice': arr([0.0, 0.0, 0.7, 0.3])
}

PROFITFACTOR = 1.0  # ratio times break even rent
EFFICIENCY = .7  # interior building efficient
PARCELUSE = .8  # efficiency of footprint on parcel
INTERESTRATE = .05  # interest rate
CAPRATE = INTERESTRATE  # cp rate
PERIODS = 20  # number of periods (years)

# per uses above and per thousands of sqft
parking_rates = arr([2.0, .6, 1.0, 1.0])
SQFTPERRATE = 1000.0

parking_configs = ['surface', 'deck', 'underground']

costs = np.transpose(np.array([
    [160.0, 175.0, 200.0, 230.0],  # retail
    [140.0, 175.0, 200.0, 230.0],  # industrial
    [160.0, 175.0, 200.0, 230.0],  # office
    [170.0, 190.0, 210.0, 240.0],  # multifamily
]))
# max height for each of the costs above from left to right, same for all uses
heightsforcosts = [15, 55, 120, np.inf]

# parking costs!
parking_sqft_d = {'surface': 300.0, 'deck': 250.0, 'underground': 250.0}
parking_cost_d = {'surface': 30, 'deck': 90, 'underground': 110}

HEIGHTPERSTORY = 10.0
MAXRETAILHEIGHT = 2.0
MAXINDUSTRIALHEIGHT = 2.0


def building_cost(use_mix, stories, df):
    height = stories * HEIGHTPERSTORY

    df['stories'] = stories

    cost = np.searchsorted(heightsforcosts, height)
    cost[np.isnan(height)] = 0

    cost = np.dot(np.squeeze(costs[cost.astype('int32')]), use_mix)
    cost[np.isnan(stories).flatten()] = np.nan

    cost = np.reshape(cost, (-1, 1))
    df['costsqft'] = cost

    return cost

tiledparcelsizes = np.reshape(np.repeat(parcelsizes, fars.size), (-1, 1))


class Developer:

    def __init__(self):
        self.generate_lookup()

    # run the developer model on all parcel inputs
    # (not actually running on parcels here)
    # these are hypothetical buildings
    def generate_lookup(self):
        keys = forms.keys()
        keys.sort()
        df_d = {}
        for name in keys:
            uses_distrib = forms[name]

            for parking_config in parking_configs:

                df = pd.DataFrame(index=fars)
                df['far'] = fars
                df['pclsz'] = tiledparcelsizes

                building_bulk = np.reshape(
                    parcelsizes, (-1, 1)) * np.reshape(fars, (1, -1))
                building_bulk = np.reshape(building_bulk, (-1, 1))

                # need to converge in on exactly how much far is available for
                # deck pkg
                if parking_config == 'deck':
                    orig_bulk = building_bulk
                    while 1:
                        parkingstalls = building_bulk * \
                            np.sum(uses_distrib * parking_rates) / SQFTPERRATE
                        if np.where(
                                np.absolute(
                                    orig_bulk - building_bulk -
                                    parkingstalls *
                                    parking_sqft_d[parking_config])
                                > 10.0)[0].size == 0:
                            break
                        building_bulk = orig_bulk - parkingstalls * \
                            parking_sqft_d[parking_config]

                df['build'] = building_bulk

                parkingstalls = building_bulk * \
                    np.sum(uses_distrib * parking_rates) / SQFTPERRATE
                parking_cost = (parking_cost_d[parking_config] *
                                parkingstalls *
                                parking_sqft_d[parking_config])

                df['spaces'] = parkingstalls

                if parking_config == 'underground':
                    df['parksqft'] = parkingstalls * \
                        parking_sqft_d[parking_config]
                    stories = building_bulk / tiledparcelsizes
                if parking_config == 'deck':
                    df['parksqft'] = parkingstalls * \
                        parking_sqft_d[parking_config]
                    stories = ((building_bulk + parkingstalls *
                                parking_sqft_d[parking_config]) /
                               tiledparcelsizes)
                if parking_config == 'surface':
                    stories = building_bulk / \
                        (tiledparcelsizes - parkingstalls *
                         parking_sqft_d[parking_config])
                    df['parksqft'] = parkingstalls * \
                        parking_sqft_d[parking_config]
                    stories[np.where(stories < 0.0)] = np.nan

                stories /= PARCELUSE
                # acquisition cost!
                cost = building_cost(
                    uses_distrib, stories, df) * building_bulk + parking_cost
                df['parkcost'] = parking_cost
                df['cost'] = cost

                yearly_cost_per_sqft = np.pmt(
                    INTERESTRATE, PERIODS, cost) / (building_bulk * EFFICIENCY)
                df['yearly_pmt'] = yearly_cost_per_sqft

                break_even_weighted_rent = PROFITFACTOR * \
                    yearly_cost_per_sqft * -1.0
                if name == 'retail':
                    break_even_weighted_rent[
                        np.where(fars > MAXRETAILHEIGHT)] = np.nan
                if name == 'industrial':
                    break_even_weighted_rent[
                        np.where(fars > MAXINDUSTRIALHEIGHT)] = np.nan
                df['even_rent'] = break_even_weighted_rent

                df_d[(name, parking_config)] = df

        self.df_d = df_d

        min_even_rents_d = {}
        BIG = 999999

        for name in keys:
            min_even_rents = None
            for parking_config in parking_configs:
                even_rents = df_d[(name, parking_config)][
                    'even_rent'].fillna(BIG)
                min_even_rents = even_rents if min_even_rents is None \
                    else np.minimum(min_even_rents, even_rents)

            min_even_rents = min_even_rents.replace(BIG, np.nan)
            # this is the minimum cost per sqft for this form and far
            min_even_rents_d[name] = min_even_rents

        self.min_even_rents_d = min_even_rents_d

    # this function does the developer model lookups for all the actual parcels
    # form must be one of the forms specified here
    # rents is a matrix of rents of shape (numparcels x numuses)
    # land_costs is the current yearly rent on each parcel
    # parcel_size is the size of the parcel
    def lookup(self, form, rents, land_costs, parcel_sizes,
               max_fars, max_heights):

        print form, time.ctime()
        rents = np.dot(rents, forms[form])  # get weighted rent for this form

        even_rents = self.min_even_rents_d[form]
        print "sqft cost\n", even_rents

        # min between max_fars and max_heights
        max_heights[np.isnan(max_heights)] = 9999.0
        max_fars[np.isnan(max_fars)] = 0.0
        max_heights = max_heights / HEIGHTPERSTORY * PARCELUSE
        max_fars = np.minimum(max_heights, max_fars)

        # zero out fars not allowed by zoning
        fars = np.tile(even_rents.index.values, (len(parcel_sizes.index), 1))
        fars[fars > np.reshape(max_fars.values, (-1, 1)) + .01] = np.nan

        # parcel sizes * possible fars
        building_bulks = fars * np.reshape(parcel_sizes.values, (-1, 1))

        # cost to build the new building
        building_costs = building_bulks * \
            np.reshape(even_rents.values, (1, -1)) / CAPRATE

        # add cost to buy the current building
        building_costs = building_costs + \
            np.reshape(land_costs.values, (-1, 1)) / CAPRATE

        # rent to make for the new building
        building_revenue = building_bulks * \
            np.reshape(rents, (-1, 1)) / CAPRATE

        # profit for each form
        profit = building_revenue - building_costs

        # index maximum total profit
        # i got really weird behavior out of numpy on this line - leave it even
        # though it's ugly
        maxprofitind = np.argmax(np.nan_to_num(profit.astype('float')), axis=1)
        # value of the maximum total profit
        maxprofit = profit[np.arange(maxprofitind.size), maxprofitind]

        # far of the max profit
        maxprofit_fars = pd.Series(
            even_rents.index.values[maxprofitind].astype('float'),
            index=parcel_sizes.index)
        maxprofit = pd.Series(
            maxprofit.astype('float'), index=parcel_sizes.index)
        # remove unprofitable buildings
        maxprofit[maxprofit < 0] = np.nan
        # remove far of unprofitable building
        maxprofit_fars[pd.isnull(maxprofit)] = np.nan

        '''
  # this code does detailed debugging for a specific parcel
  label = 151358
  i = parcel_sizes.index.get_loc(label)
  print "size\n", parcel_sizes.loc[label]
  print "max_far\n", max_fars[i]
  print "bulks\n", building_bulks[i,:]
  print "costs\n", building_costs[i,:]
  print "revenue\n", building_revenue[i,:]
  print "profit\n", profit[i,:]
  print "land_costs\n", land_costs.loc[label]
  print "maxprofitind", maxprofitind[i]
  print "maxprofit", maxprofit.loc[label]
  print "max_profit_fars", maxprofit_fars.loc[label]
  '''

        print "maxprofit_fars\n", maxprofit_fars.value_counts()

        return maxprofit_fars.astype('float32'), maxprofit

    # this code creates the debugging plots to understand the behavior of
    # the hypothetical building model
    def debug_output(self):
        import matplotlib.pyplot as plt

        df_d = self.df_d
        keys = df_d.keys()
        keys.sort()
        for key in keys:
            print "\n", key, "\n"
            print df_d[key]
        for key in self.min_even_rents_d.keys():
            print "\n", key, "\n"
            print self.min_even_rents_d[key]

        keys = forms.keys()
        keys.sort()
        c = 1
        share = None
        fig = plt.figure(figsize=(12, 3 * len(keys)))
        fig.suptitle('Profitable rents by use', fontsize=40)
        for name in keys:
            sumdf = None
            for parking_config in parking_configs:
                df = df_d[(name, parking_config)]
                if sumdf is None:
                    sumdf = pd.DataFrame(df['far'])
                sumdf[parking_config] = df['even_rent']
            far = sumdf['far']
            del sumdf['far']

            if share is None:
                share = plt.subplot(len(keys) / 2, 2, c)
            else:
                plt.subplot(len(keys) / 2, 2, c, sharex=share, sharey=share)

            handles = plt.plot(far, sumdf)
            plt.ylabel('even_rent')
            plt.xlabel('FAR')
            plt.title('Rents for use type %s' % name)
            plt.legend(
                handles, parking_configs, loc='lower right',
                title='Parking type')
            c += 1
        plt.savefig('even_rents.png', bbox_inches=0)

if __name__ == '__main__':
    dev = Developer()
    dev.debug_output()
