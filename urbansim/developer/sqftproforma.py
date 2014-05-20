import time

import numpy as np
import pandas as pd


class SqFtProFormaConfig:

    parcel_sizes = [10000.0]
    fars = [.25, .5, .75, 1.0, 1.5, 1.8, 2.0, 3.0, 4.0, 5.0, 7.0, 9.0, 11.0]
    uses = ['retail', 'industrial', 'office', 'residential']
    forms = {
        'retail': {
            "retail": 1.0
        },
        'industrial': {
            "industrial": 1.0
        },
        'office': {
            "office": 1.0
        },
        'residential': {
            "residential": 1.0
        },
        'mixedresidential': {
            "retail": .1,
            "residential": .9
        },
        'mixedoffice': {
            "office": 0.7,
            "residential": 0.3
        }
    }

    PROFITFACTOR = 1.0
    EFFICIENCY = .7
    PARCELUSE = .8
    INTERESTRATE = .05
    CAPRATE = INTERESTRATE
    PERIODS = 20

    parking_rates = {
        "retail": 2.0,
        "industrial": .6,
        "office": 1.0,
        "residential": 1.0
    }
    SQFTPERRATE = 1000.0

    parking_configs = ['surface', 'deck', 'underground']

    costs = {
        "retail": [160.0, 175.0, 200.0, 230.0],
        "industrial": [140.0, 175.0, 200.0, 230.0],
        "office": [160.0, 175.0, 200.0, 230.0],
        "multifamily": [170.0, 190.0, 210.0, 240.0]
    }

    heights_for_costs = [15, 55, 120, np.inf]

    parking_sqft_d = {
        'surface': 300.0,
        'deck': 250.0,
        'underground': 250.0
    }
    parking_cost_d = {
        'surface': 30,
        'deck': 90,
        'underground': 110
    }

    HEIGHTPERSTORY = 10.0
    MAXRETAILHEIGHT = 2.0
    MAXINDUSTRIALHEIGHT = 2.0

    def __init__(self):
        """
        This class encapsulates the configuration options for the square
        foot based pro forma.

        Attributes:
        parcel_sizes : list
            A list of parcel sizes to test.  Interestingly, right now
            the parcel sizes cancel is this style of pro forma computation so
            you can set this to something reasonable for debugging purposes -
            e.g. [10000].  All sizes can be feet or meters as long as they are
            consistently used.
        fars : list
            A list of floor area ratios to use.  FAR is a multiple of
            the parcel size that is the total building bulk that is allowed by
            zoning on the site.  In this case, all of these ratios will be
            tested regardless of zoning and the zoning test will be performed
            later.
        uses : list
            A list of space uses to use within a building.  These are
            mixed into forms.  Generally speaking, you should only have uses
            for which you have an estimate (or observed) values for rents in
            the building.  By default, uses are retail, industrial, office,
            and residential.
        forms : dict
            A dictionary where keys are names for the form and values
            are also dictionaries where keys are uses and values are the
            proportion of that use used in this form.  The values of the
            dictionary should sum to 1.0.  For instance, a form called
            "residential" might have a dict of space allocations equal to
            {"residential": 1.0} while a form called "mixedresidential"
            might have a dictionary of space allocations equal to
            {"retail": .1, "residential" .9] which is 90% residential and
            10% retail.
        parking_rates : dict
            A dict of rates per thousand square feet where keys are the uses
            from the list specified in the attribute above.  The ratios
            are typically in the range 0.5 - 3.0 or similar.  So for
            instance, a key-value pair of "retail": 2.0 would be two parking
            spaces per 1,000 square feet of retail.  This is a per square
            foot pro forma, so the more typically parking ratio of spaces
            per residential unit must be converted to square feet for use in
            this pro forma.
        SQFTPERRATE : float
            The number of square feet per unit for use in the
            parking_rates above.  By default this is set to 1,000 but can be
            overridden.
        parking_configs : list
            An expert parameter and is usually unchanged.  By default
            it is set to ['surface', 'deck', 'underground'] and very semantic
            differences in the computation are performed for each of these
            parking configurations.  Generally speaking it will break things
            to change this array, but an item can be removed if that parking
            configuration should not be tested.
        parking_sqft_d : dict
            A dictionary where keys are the three parking
            configurations listed above and values are square foot uses of
            parking spaces in that configuration.  This is to capture the
            fact that surface parking is usually more space intensive
            than deck or underground parking.
        parking_cost_d : dict
            The parking cost for each parking configuration.  Keys are the
            name of the three parking configurations listed above and values
            are dollars PER SQUARE FOOT for parking in that configuration.
            Used to capture the fact that underground and deck are far more
            expensive than surface parking.
        height_for_costs : list
            A list of "break points" as heights at which construction becomes
            more expensive.  Generally these are the heights at which
            construction materials change from wood, to concrete, to steel.
            Costs are also given as lists by use for each of these break
            points and are considered to be valid up to the break point.  A
            list would look something like [15, 55, 120, np.inf].
        costs : dict
            The keys are uses from the attribute above and the values are a
            list of floating point numbers of same length as the
            height_for_costs attribute.  A key-value pair of
            "residential": [160.0, 175.0, 200.0, 230.0] would say that the
            residential use if $160/sqft up to 15ft in total height for the
            building, $175/sqft up to 55ft, $200/sqft up to 120ft, and
            $230/sqft beyond.  A final value in the height_for_costs
            array of np.inf is typical.
        HEIGHTPERSTORY : float
            The per-story height for the building used to turn an
            FAR into an actual height.
        MAXRETAILHEIGHT : float
            The maximum height of retail buildings to consider.
        MAXINDUSTRIALHEIGHT : float
            The maxmium height of industrial buildings to consider.
        PROFITFACTOR : float
            The ratio of profit a developer expects to make above the break
            even rent.
        EFFICIENCY : float
            The efficiency of the building.  This turns total FAR into the
            amount of space which gets a square foot rent.  The entire building
            gets the cost of course.
        PARCELUSE : float
            The ratio of the building footprint to the parcel size.  Also used
            to turn an FAR into a height to cost properly.
        INTERESTRATE : float
            The interest rate on loans.  Used to turn the total cost of the
            building into a mortgage payment per year.
        PERIODS : float
            Number of years (periods) for the construction loan.
        CAPRATE : float
            The rate an investor is willing to pay for a cash flow per year.
            This means $1/year is equivalent to 1/CAPRATE present dollars.
            This is a macroeconomic input that is widely available on the
            internet.
        """
        pass

    @property
    def costs_arr(self):
        return np.transpose(np.array([
            [160.0, 175.0, 200.0, 230.0],  # retail
            [140.0, 175.0, 200.0, 230.0],  # industrial
            [160.0, 175.0, 200.0, 230.0],  # office
            [170.0, 190.0, 210.0, 240.0],  # multifamily
        ]))

    @property
    def tiled_parcel_sizes(self):
        return np.reshape(np.repeat(self.parcel_sizes, self.fars.size), (-1, 1))


class SqFtProForma:

    def __init__(self, config=None):
        """
        Initialize the square foot based pro forma.

        This pro forma has no representation of units - it does not
        differentiate between the rent attained by 1BR, 2BR, or 3BR and change
        the rents accordingly.  This is largely because it is difficult to get
        information on the unit mix in an existing building in order to compute
        its acquisition cost.  Thus rents and costs per sqft are used for new
        and current buildings which assumes there is a constant return on
        increasing and decreasing unit sizes, an extremely useful simplifying
        assumption above the project scale (i.e. city of regional scale)

        Parameters:
        -----------
        config : The [optional] configuration object which should be an
        instance of SqftProFormaConfig.  The configuration options for this
        pro forma are documented on the configuration object.
        """
        if config is None:
            config = SqFtProFormaConfig()
        self.config = config
        self._generate_lookup()

    # run the developer model on all possible inputs specified in the
    # configuration object - not generally called by the user
    def _generate_lookup(self):
        c = self.config

        # get all the building forms we can use
        keys = c.forms.keys()
        keys.sort()
        df_d = {}
        for name in keys:
            # get the use distribution for each
            uses_distrib = c.forms[name]

            for parking_config in c.parking_configs:

                df = pd.DataFrame(index=c.fars)
                df['far'] = c.fars
                df['pclsz'] = c.tiledparcelsizes

                building_bulk = np.reshape(
                    c.parcel_sizes, (-1, 1)) * np.reshape(c.fars, (1, -1))
                building_bulk = np.reshape(building_bulk, (-1, 1))

                # need to converge in on exactly how much far is available for
                # deck pkg
                if parking_config == 'deck':
                    orig_bulk = building_bulk
                    while 1:
                        parkingstalls = building_bulk * \
                            np.sum(uses_distrib * c.parking_rates) / c.SQFTPERRATE
                        if np.where(
                                np.absolute(
                                    orig_bulk - building_bulk -
                                    parkingstalls *
                                    c.parking_sqft_d[parking_config])
                                > 10.0)[0].size == 0:
                            break
                        building_bulk = orig_bulk - parkingstalls * \
                            c.parking_sqft_d[parking_config]

                df['build'] = building_bulk

                parkingstalls = building_bulk * \
                    np.sum(uses_distrib * c.parking_rates) / c.SQFTPERRATE
                parking_cost = (c.parking_cost_d[parking_config] *
                                parkingstalls *
                                c.parking_sqft_d[parking_config])

                df['spaces'] = parkingstalls

                if parking_config == 'underground':
                    df['parksqft'] = parkingstalls * \
                        c.parking_sqft_d[parking_config]
                    stories = building_bulk / c.tiledparcelsizes
                if parking_config == 'deck':
                    df['parksqft'] = parkingstalls * \
                        c.parking_sqft_d[parking_config]
                    stories = ((building_bulk + parkingstalls *
                                c.parking_sqft_d[parking_config]) /
                               c.tiledparcelsizes)
                if parking_config == 'surface':
                    stories = building_bulk / \
                        (c.tiledparcelsizes - parkingstalls *
                         c.parking_sqft_d[parking_config])
                    df['parksqft'] = parkingstalls * \
                        c.parking_sqft_d[parking_config]
                    stories[np.where(stories < 0.0)] = np.nan

                stories /= c.PARCELUSE
                # acquisition cost!
                cost = c.building_cost(
                    uses_distrib, stories, df) * building_bulk + parking_cost
                df['parkcost'] = parking_cost
                df['cost'] = cost

                yearly_cost_per_sqft = np.pmt(
                    c.INTERESTRATE, c.PERIODS, cost) / (building_bulk * c.EFFICIENCY)
                df['yearly_pmt'] = yearly_cost_per_sqft

                break_even_weighted_rent = c.PROFITFACTOR * \
                    yearly_cost_per_sqft * -1.0
                if name == 'retail':
                    break_even_weighted_rent[
                        np.where(c.fars > c.MAXRETAILHEIGHT)] = np.nan
                if name == 'industrial':
                    break_even_weighted_rent[
                        np.where(c.fars > c.MAXINDUSTRIALHEIGHT)] = np.nan
                df['even_rent'] = break_even_weighted_rent

                df_d[(name, parking_config)] = df

        self.df_d = df_d

        min_even_rents_d = {}
        bignum = 999999

        for name in keys:
            min_even_rents = None
            for parking_config in c.parking_configs:
                even_rents = df_d[(name, parking_config)][
                    'even_rent'].fillna(bignum)
                min_even_rents = even_rents if min_even_rents is None \
                    else np.minimum(min_even_rents, even_rents)

            min_even_rents = min_even_rents.replace(bignum, np.nan)
            # this is the minimum cost per sqft for this form and far
            min_even_rents_d[name] = min_even_rents

        self.min_even_rents_d = min_even_rents_d

    def _building_cost(self, use_mix, stories, df):
        c = self.config
        height = stories * c.HEIGHTPERSTORY

        df['stories'] = stories

        cost = np.searchsorted(c.heightsforcosts, height)
        cost[np.isnan(height)] = 0

        cost = np.dot(np.squeeze(c.costs[cost.astype('int32')]), use_mix)
        cost[np.isnan(stories).flatten()] = np.nan

        cost = np.reshape(cost, (-1, 1))
        df['costsqft'] = cost

        return cost

    # this function does the developer model lookups for all the actual parcels
    # form must be one of the forms specified here
    # rents is a matrix of rents of shape (numparcels x numuses)
    # land_costs is the current yearly rent on each parcel
    # parcel_size is the size of the parcel
    def lookup(self, form, rents, land_costs, parcel_sizes,
               max_fars, max_heights):

        print form, time.ctime()
        c = self.config
        rents = np.dot(rents, c.forms[form])  # get weighted rent for this form

        even_rents = self.min_even_rents_d[form]
        print "sqft cost\n", even_rents

        # min between max_fars and max_heights
        max_heights[np.isnan(max_heights)] = 9999.0
        max_fars[np.isnan(max_fars)] = 0.0
        max_heights = max_heights / c.HEIGHTPERSTORY * c.PARCELUSE
        max_fars = np.minimum(max_heights, max_fars)

        # zero out fars not allowed by zoning
        fars = np.tile(even_rents.index.values, (len(parcel_sizes.index), 1))
        fars[fars > np.reshape(max_fars.values, (-1, 1)) + .01] = np.nan

        # parcel sizes * possible fars
        building_bulks = fars * np.reshape(parcel_sizes.values, (-1, 1))

        # cost to build the new building
        building_costs = building_bulks * \
            np.reshape(even_rents.values, (1, -1)) / c.CAPRATE

        # add cost to buy the current building
        building_costs = building_costs + \
            np.reshape(land_costs.values, (-1, 1)) / c.CAPRATE

        # rent to make for the new building
        building_revenue = building_bulks * \
            np.reshape(rents, (-1, 1)) / c.CAPRATE

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
        c = self.config

        df_d = self.df_d
        keys = df_d.keys()
        keys.sort()
        for key in keys:
            print "\n", key, "\n"
            print df_d[key]
        for key in self.min_even_rents_d.keys():
            print "\n", key, "\n"
            print self.min_even_rents_d[key]

        keys = c.forms.keys()
        keys.sort()
        c = 1
        share = None
        fig = plt.figure(figsize=(12, 3 * len(keys)))
        fig.suptitle('Profitable rents by use', fontsize=40)
        for name in keys:
            sumdf = None
            for parking_config in c.parking_configs:
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
                handles, c.parking_configs, loc='lower right',
                title='Parking type')
            c += 1
        plt.savefig('even_rents.png', bbox_inches=0)
