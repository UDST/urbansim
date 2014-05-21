import numpy as np
import pandas as pd


class SqFtProFormaConfig:

    def _reset_defaults(self):
        self.parcel_sizes = [10000.0]
        self.fars = [.1, .25, .5, .75, 1.0, 1.5, 1.8, 2.0, 3.0, 4.0, 5.0, 7.0, 9.0, 11.0]
        self.uses = ['retail', 'industrial', 'office', 'residential']
        self.forms = {
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

        self.profit_factor = 1.0
        self.efficiency = .7
        self.parcel_use = .8
        self.interest_rate = .05
        self.cap_rate = self.interest_rate
        self.periods = 20

        self.parking_rates = {
            "retail": 2.0,
            "industrial": .6,
            "office": 1.0,
            "residential": 1.0
        }
        self.sqft_per_rate = 1000.0

        self.parking_configs = ['surface', 'deck', 'underground']

        self.costs = {
            "retail": [160.0, 175.0, 200.0, 230.0],
            "industrial": [140.0, 175.0, 200.0, 230.0],
            "office": [160.0, 175.0, 200.0, 230.0],
            "residential": [170.0, 190.0, 210.0, 240.0]
        }

        self.heights_for_costs = [15, 55, 120, np.inf]

        self.parking_sqft_d = {
            'surface': 300.0,
            'deck': 250.0,
            'underground': 250.0
        }
        self.parking_cost_d = {
            'surface': 30,
            'deck': 90,
            'underground': 110
        }

        self.height_per_story = 10.0
        self.max_retail_height = 2.0
        self.max_industrial_height = 2.0

    def __init__(self):
        """
        This class encapsulates the configuration options for the square
        foot based pro forma.

        Attributes
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
        sqft_per_rate : float
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
        height_per_story : float
            The per-story height for the building used to turn an
            FAR into an actual height.
        max_retail_height : float
            The maximum height of retail buildings to consider.
        max_industrial_height : float
            The maxmium height of industrial buildings to consider.
        profit_factor : float
            The ratio of profit a developer expects to make above the break
            even rent.
        efficiency : float
            The efficiency of the building.  This turns total FAR into the
            amount of space which gets a square foot rent.  The entire building
            gets the cost of course.
        parcel_use : float
            The ratio of the building footprint to the parcel size.  Also used
            to turn an FAR into a height to cost properly.
        interest_rate : float
            The interest rate on loans.  Used to turn the total cost of the
            building into a mortgage payment per year.
        periods : float
            Number of years (periods) for the construction loan.
        cap_rate : float
            The rate an investor is willing to pay for a cash flow per year.
            This means $1/year is equivalent to 1/cap_rate present dollars.
            This is a macroeconomic input that is widely available on the
            internet.
        """
        self._reset_defaults()

    def _convert_types(self):
        """
        convert lists and dictionaries that are useful for users to
        np vectors that are usable by machines
        """
        self.fars = np.array(self.fars)
        self.parking_rates = np.array([self.parking_rates[use] for use in self.uses])
        for k, v in self.forms.iteritems():
            self.forms[k] = np.array([self.forms[k].get(use, 0.0) for use in self.uses])
            # normalize if not already
            self.forms[k] /= self.forms[k].sum()
        self.costs = np.transpose(np.array([self.costs[use] for use in self.uses]))

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
        self.config._convert_types()
        self._generate_lookup()

    def _building_cost(self, use_mix, stories):
        """
        Generate building cost for a set of buildings

        Parameters
        ----------
        use_mix : array
            The mix of uses for this form
        stories : series
            A Pandas Series of stories

        Returns
        -------
        The cost per sqft for this unit mix and height
        """
        c = self.config
        # stories to heights
        heights = stories * c.height_per_story
        # cost index for this height
        costs = np.searchsorted(c.heights_for_costs, heights)
        # this will get set to nan later
        costs[np.isnan(heights)] = 0
        # compute cost with matrix multiply
        costs = np.dot(np.squeeze(c.costs[costs.astype('int32')]), use_mix)
        # some heights aren't allowed - cost should be nan
        costs[np.isnan(stories).flatten()] = np.nan
        return costs.flatten()

    def _generate_lookup(self):
        """
        Run the developer model on all possible inputs specified in the
        configuration object - not generally called by the user.  This part
        computes the final cost per sqft of the building to construct and
        then turns it into the yearly rent necessary to make break even on
        that cost.
        """
        c = self.config

        # get all the building forms we can use
        keys = c.forms.keys()
        keys.sort()
        df_d = {}
        for name in keys:
            # get the use distribution for each
            uses_distrib = c.forms[name]

            for parking_config in c.parking_configs:

                # going to make a dataframe to store values to make
                # pro forma results transparent
                df = pd.DataFrame(index=c.fars)
                df['far'] = c.fars
                df['pclsz'] = c.tiled_parcel_sizes

                building_bulk = np.reshape(
                    c.parcel_sizes, (-1, 1)) * np.reshape(c.fars, (1, -1))
                building_bulk = np.reshape(building_bulk, (-1, 1))

                # need to converge in on exactly how much far is available for
                # deck pkg
                if parking_config == 'deck':
                    orig_bulk = building_bulk
                    while 1:
                        parkingstalls = building_bulk * \
                            np.sum(uses_distrib * c.parking_rates) / c.sqft_per_rate
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
                    np.sum(uses_distrib * c.parking_rates) / c.sqft_per_rate
                parking_cost = (c.parking_cost_d[parking_config] *
                                parkingstalls *
                                c.parking_sqft_d[parking_config])

                df['spaces'] = parkingstalls

                if parking_config == 'underground':
                    df['parksqft'] = parkingstalls * \
                        c.parking_sqft_d[parking_config]
                    stories = building_bulk / c.tiled_parcel_sizes
                if parking_config == 'deck':
                    df['parksqft'] = parkingstalls * \
                        c.parking_sqft_d[parking_config]
                    stories = ((building_bulk + parkingstalls *
                                c.parking_sqft_d[parking_config]) /
                               c.tiled_parcel_sizes)
                if parking_config == 'surface':
                    stories = building_bulk / \
                        (c.tiled_parcel_sizes - parkingstalls *
                         c.parking_sqft_d[parking_config])
                    df['parksqft'] = parkingstalls * \
                        c.parking_sqft_d[parking_config]
                    # not all fars support surface parking
                    stories[np.where(stories < 0.0)] = np.nan

                df['total_sqft'] = df.build + df.parksqft
                stories /= c.parcel_use
                df['stories'] = stories
                df['build_cost_sqft'] = self._building_cost(uses_distrib, stories)

                df['build_cost'] = df.build_cost_sqft * df.build
                df['park_cost'] = parking_cost
                df['cost'] = df.build_cost + df.park_cost

                #df['yearly_pmt'] = np.pmt(c.interest_rate, c.periods, df.cost)

                df['ave_cost_sqft'] = (df.cost / df.total_sqft) * c.profit_factor

                if name == 'retail':
                    df['ave_cost_sqft'][c.fars > c.max_retail_height] = np.nan
                if name == 'industrial':
                    df['ave_cost_sqft'][c.fars > c.max_industrial_height] = np.nan
                df_d[(name, parking_config)] = df

        # from here on out we need the min rent for a form and a far
        min_ave_cost_sqft_d = {}
        bignum = 999999

        for name in keys:
            min_ave_cost_sqft = None
            for parking_config in c.parking_configs:
                ave_cost_sqft = df_d[(name, parking_config)][
                    'ave_cost_sqft'].fillna(bignum)
                min_ave_cost_sqft = ave_cost_sqft if min_ave_cost_sqft is None \
                    else np.minimum(min_ave_cost_sqft, ave_cost_sqft)

            min_ave_cost_sqft = min_ave_cost_sqft.replace(bignum, np.nan)
            # this is the minimum cost per sqft for this form and far
            min_ave_cost_sqft_d[name] = min_ave_cost_sqft

        self.dev_d = df_d
        self.min_ave_cost_d = min_ave_cost_sqft_d

    def get_debug_info(self, form, parking_config):
        """
        Get the debug info after running the pro forma for a given form and parking
        configuration

        Parameters
        ----------
        form : string
            The form to get debug info for
        parking_config : string
            The parking configuration to get debug info for

        Returns
        -------
        A dataframe where the index is the far with many columns representing
        intermediate steps in the pro forma computation.  Additional documentation
        will be added at a later date, although many of the columns should be fairly
        self-expanatory.
        """
        return self.dev_d[(form, parking_config)]

    def get_ave_cost_sqft(self, form):
        """
        Get the average cost per sqft for the pro forma for a given form

        Parameters
        ----------
        form : string
            Get a series representing the average cost per sqft for each form in the
            config

        Returns
        -------
        A series where the index is the far and the values are the average cost per
        sqft at which the building is "break even" given the configuration parameters
        that were passed at run time.
        """
        return self.min_ave_cost_d[form]

    def lookup(self, form, df):
        """
        This function does the developer model lookups for all the actual input data.

        Parameters
        ----------
        form : string
            One of the forms specified in the configuration file
        df: dataframe
            Pass in a single data frame which is indexed by parcel_id and has the
            following columns

        Columns
        -------
        rents : dataframe
            A set of columns, one for each of the uses passed in the configuration.
            Values are yearly rents for that use.  Typical column names would be
            "residential", "retail", "industrial" and "office"
        land_costs : series
            A series representing the CURRENT yearly rent for each parcel.  Used to
            compute acquisition costs for the parcel.
        parcel_sizes : series
            A series representing the parcel size for each parcel.
        max_fars : series
            A series representing the maximum far allowed by zoning.  Buildings
            will not be built above these fars.
        max_heights : series
            A series representing the maxmium height allowed by zoning.  Buildings
            will not be built above these heights.  Will pick between the min of
            the far and height, will ignore on of them if one is nan, but will not
            build if both are nan.
        """
        c = self.config

        cost_sqft = self.get_ave_cost_sqft(form)
        cost_sqft_col = np.reshape(cost_sqft.values, (-1, 1))
        cost_sqft_index_col = np.reshape(cost_sqft.index.values, (-1, 1))

        # weighted rent for this form
        df['weighted_rent'] = np.dot(df[c.uses], c.forms[form])

        # min between max_fars and max_heights
        df['max_far_from_heights'] = df.max_heights / c.height_per_story * c.parcel_use
        df['min_max_fars'] = df[['max_far_from_heights', 'max_fars']].min(axis=1).fillna(0)
        df = df.query('min_max_fars > 0 and parcel_sizes > 0')
        print df.min_max_fars

        # all possible fars on all parcels
        fars = np.repeat(cost_sqft_index_col, len(df.index), axis=1)
        print fars

        # zero out fars not allowed by zoning
        fars[fars > df.min_max_fars.values + .01] = np.nan
        print fars

        # parcel sizes * possible fars
        building_bulks = fars * df.parcel_sizes.values
        print building_bulks

        # cost to build the new building
        building_costs = building_bulks * cost_sqft_col
        print building_costs

        # add cost to buy the current building
        building_costs += df.land_costs.values
        print "costs + land"
        print building_costs

        # rent to make for the new building
        building_revenue = building_bulks * df.weighted_rent.values / c.cap_rate
        print "revenue"
        print building_revenue

        # profit for each form
        profit = building_revenue - building_costs
        print profit

        maxprofitind = np.argmax(np.nan_to_num(profit.astype('float')), axis=0)
        print maxprofitind

        def twod_get(indexes, arr):
            return arr[indexes, np.arange(indexes.size)].astype('float')

        outdf = pd.DataFrame({
            'building_size': twod_get(maxprofitind, building_bulks),
            'building_cost': twod_get(maxprofitind, profit),
            'land_cost': twod_get(maxprofitind, profit),
            'building_revenue': twod_get(maxprofitind, building_revenue),
            'max_profit_far': twod_get(maxprofitind, fars),
            'max_profit': twod_get(maxprofitind, profit)
        }, index=df.index)
        print outdf.max_profit_far
        print outdf.max_profit

        return outdf

    def _debug_output(self):
        """
        this code creates the debugging plots to understand
        the behavior of the hypothetical building model
        """
        import matplotlib.pyplot as plt
        c = self.config

        df_d = self.dev_d
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
        cnt = 1
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
                share = plt.subplot(len(keys) / 2, 2, cnt)
            else:
                plt.subplot(len(keys) / 2, 2, cnt, sharex=share, sharey=share)

            handles = plt.plot(far, sumdf)
            plt.ylabel('even_rent')
            plt.xlabel('FAR')
            plt.title('Rents for use type %s' % name)
            plt.legend(
                handles, c.parking_configs, loc='lower right',
                title='Parking type')
            cnt += 1
        plt.savefig('even_rents.png', bbox_inches=0)
