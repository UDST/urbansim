import numpy as np
import pandas as pd
import logging


logger = logging.getLogger(__name__)


class SqFtProFormaConfig(object):
    """
    This class encapsulates the configuration options for the square
    foot based pro forma.

    parcel_sizes : list
        A list of parcel sizes to test.  Interestingly, right now
        the parcel sizes cancel in this style of pro forma computation so
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
        The maximum height of industrial buildings to consider.
    profit_factor : float
        The ratio of profit a developer expects to make above the break
        even rent.  Should be greater than 1.0, e.g. a 10% profit would be
        a profit factor of 1.1.
    building_efficiency : float
        The efficiency of the building.  This turns total FAR into the
        amount of space which gets a square foot rent.  The entire building
        gets the cost of course.
    parcel_coverage : float
        The ratio of the building footprint to the parcel size.  Also used
        to turn an FAR into a height to cost properly.
    cap_rate : float
        The rate an investor is willing to pay for a cash flow per year.
        This means $1/year is equivalent to 1/cap_rate present dollars.
        This is a macroeconomic input that is widely available on the
        internet.

    """

    def __init__(self):
        self._reset_defaults()

    def _reset_defaults(self):
        self.parcel_sizes = [10000.0]
        self.fars = [.1, .25, .5, .75, 1.0, 1.5, 1.8, 2.0, 2.25, 2.5, 2.75,
                     3.0, 3.25, 3.5, 3.75, 4.0, 4.5,
                     5.0, 5.5, 6.0, 6.5, 7.0, 9.0, 11.0]
        self.uses = ['retail', 'industrial', 'office', 'residential']
        self.residential_uses = [False, False, False, True]
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

        self.profit_factor = 1.1
        self.building_efficiency = .7
        self.parcel_coverage = .8
        self.cap_rate = .05

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

        self.height_per_story = 12.0
        self.max_retail_height = 2.0
        self.max_industrial_height = 2.0

    def _convert_types(self):
        """
        convert lists and dictionaries that are useful for users to
        np vectors that are usable by machines

        """
        self.fars = np.array(self.fars)
        self.parking_rates = np.array([self.parking_rates[use] for use in self.uses])
        self.res_ratios = {}
        assert len(self.uses) == len(self.residential_uses)
        for k, v in self.forms.iteritems():
            self.forms[k] = np.array([self.forms[k].get(use, 0.0) for use in self.uses])
            # normalize if not already
            self.forms[k] /= self.forms[k].sum()
            self.res_ratios[k] = pd.Series(self.forms[k])[self.residential_uses].sum()
        self.costs = np.transpose(np.array([self.costs[use] for use in self.uses]))

    @property
    def tiled_parcel_sizes(self):
        return np.reshape(np.repeat(self.parcel_sizes, self.fars.size), (-1, 1))

    def check_is_reasonable(self):
        fars = pd.Series(self.fars)
        assert len(fars[fars > 20]) == 0
        assert len(fars[fars <= 0]) == 0
        for k, v in self.forms.iteritems():
            assert isinstance(v, dict)
            for k2, v2 in self.forms[k].iteritems():
                assert isinstance(k2, str)
                assert isinstance(v2, float)
            for k2, v2 in self.forms[k].iteritems():
                assert isinstance(k2, str)
                assert isinstance(v2, float)
        for k, v in self.parking_rates.iteritems():
            assert isinstance(k, str)
            assert k in self.uses
            assert 0 <= v < 5
        for k, v in self.parking_sqft_d.iteritems():
            assert isinstance(k, str)
            assert k in self.parking_configs
            assert 50 <= v <= 1000
        for k, v in self.parking_sqft_d.iteritems():
            assert isinstance(k, str)
            assert k in self.parking_cost_d
            assert 10 <= v <= 300
        for v in self.heights_for_costs:
            assert isinstance(v, int) or isinstance(v, float)
            if np.isinf(v):
                continue
            assert 0 <= v <= 1000
        for k, v in self.costs.iteritems():
            assert isinstance(k, str)
            assert k in self.uses
            for i in v:
                assert 10 < i < 1000


class SqFtProForma(object):
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

    Parameters
    ----------
    config : `SqFtProFormaConfig`
        The configuration object which should be an
        instance of `SqFtProFormaConfig`.  The configuration options for this
        pro forma are documented on the configuration object.

    """

    def __init__(self, config=None):
        if config is None:
            config = SqFtProFormaConfig()
        config.check_is_reasonable()
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
        array
            The cost per sqft for this unit mix and height.

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
                    building_bulk /= (1.0 + np.sum(uses_distrib * c.parking_rates) *
                                      c.parking_sqft_d[parking_config] /
                                      c.sqft_per_rate)

                df['building_sqft'] = building_bulk

                parkingstalls = building_bulk * \
                    np.sum(uses_distrib * c.parking_rates) / c.sqft_per_rate
                parking_cost = (c.parking_cost_d[parking_config] *
                                parkingstalls *
                                c.parking_sqft_d[parking_config])

                df['spaces'] = parkingstalls

                if parking_config == 'underground':
                    df['park_sqft'] = parkingstalls * \
                        c.parking_sqft_d[parking_config]
                    stories = building_bulk / c.tiled_parcel_sizes
                if parking_config == 'deck':
                    df['park_sqft'] = parkingstalls * \
                        c.parking_sqft_d[parking_config]
                    stories = ((building_bulk + parkingstalls *
                                c.parking_sqft_d[parking_config]) /
                               c.tiled_parcel_sizes)
                if parking_config == 'surface':
                    stories = building_bulk / \
                        (c.tiled_parcel_sizes - parkingstalls *
                         c.parking_sqft_d[parking_config])
                    df['park_sqft'] = 0
                    # not all fars support surface parking
                    stories[stories < 0.0] = np.nan
                    # I think we can assume that stories over 3
                    # do not work with surface parking
                    stories[stories > 5.0] = np.nan

                df['total_built_sqft'] = df.building_sqft + df.park_sqft
                df['parking_sqft_ratio'] = df.park_sqft / df.total_built_sqft
                stories /= c.parcel_coverage
                df['stories'] = np.ceil(stories)
                df['height'] = df.stories * c.height_per_story
                df['build_cost_sqft'] = self._building_cost(uses_distrib, stories)

                df['build_cost'] = df.build_cost_sqft * df.building_sqft
                df['park_cost'] = parking_cost
                df['cost'] = df.build_cost + df.park_cost

                df['ave_cost_sqft'] = (df.cost / df.total_built_sqft) * c.profit_factor

                if name == 'retail':
                    df['ave_cost_sqft'][c.fars > c.max_retail_height] = np.nan
                if name == 'industrial':
                    df['ave_cost_sqft'][c.fars > c.max_industrial_height] = np.nan

                df_d[(name, parking_config)] = df

        self.dev_d = df_d

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
        debug_info : dataframe
            A dataframe where the index is the far with many columns
            representing intermediate steps in the pro forma computation.
            Additional documentation will be added at a later date, although
            many of the columns should be fairly self-expanatory.

        """
        return self.dev_d[(form, parking_config)]

    def get_ave_cost_sqft(self, form, parking_config):
        """
        Get the average cost per sqft for the pro forma for a given form
        Parameters
        ----------
        form : string
            Get a series representing the average cost per sqft for each form in
            the config
        parking_config : string
            The parking configuration to get debug info for
        Returns
        -------
        cost : series
            A series where the index is the far and the values are the average
            cost per sqft at which the building is "break even" given the
            configuration parameters that were passed at run time.
        """
        return self.dev_d[(form, parking_config)].ave_cost_sqft

    def lookup(self, form, df, only_built=True, pass_through=None):
        """
        This function does the developer model lookups for all the actual input data.

        Parameters
        ----------
        form : string
            One of the forms specified in the configuration file
        df: dataframe
            Pass in a single data frame which is indexed by parcel_id and has the
            following columns
        only_built : bool
            Whether to return only those buildings that are profitable and allowed
            by zoning, or whether to return as much information as possible, even if
            unlikely to be built (can be used when development might be subsidized
            or when debugging)
        pass_through : list of strings
            List of field names to take from the input parcel frame and pass
            to the output feasibility frame - is usually used for debugging
            purposes - these fields will be passed all the way through
            developer

        Input Dataframe Columns
        rent : dataframe
            A set of columns, one for each of the uses passed in the configuration.
            Values are yearly rents for that use.  Typical column names would be
            "residential", "retail", "industrial" and "office"
        land_cost : series
            A series representing the CURRENT yearly rent for each parcel.  Used to
            compute acquisition costs for the parcel.
        parcel_size : series
            A series representing the parcel size for each parcel.
        max_far : series
            A series representing the maximum far allowed by zoning.  Buildings
            will not be built above these fars.
        max_height : series
            A series representing the maxmium height allowed by zoning.  Buildings
            will not be built above these heights.  Will pick between the min of
            the far and height, will ignore on of them if one is nan, but will not
            build if both are nan.
        max_dua : series, optional
            A series representing the maximum dwelling units per acre allowed by
            zoning.  If max_dua is passed, the average unit size should be passed
            below to translate from dua to floor space.
        ave_unit_size : series, optional
            This is required if max_dua is passed above, otherwise it is optional.
            This is the same as the parameter to Developer.pick() (it should be the
            same series).

        Returns
        -------
        index : Series, int
            parcel identifiers
        building_sqft : Series, float
            The number of square feet for the building to build.  Keep in mind
            this includes parking and common space.  Will need a helpful function
            to convert from gross square feet to actual usable square feet in
            residential units.
        building_cost : Series, float
            The cost of constructing the building as given by the
            ave_cost_per_sqft from the cost model (for this FAR) and the number
            of square feet.
        total_cost : Series, float
            The cost of constructing the building plus the cost of acquisition of
            the current parcel/building.
        building_revenue : Series, float
            The NPV of the revenue for the building to be built, which is the
            number of square feet times the yearly rent divided by the cap
            rate (with a few adjustment factors including building efficiency).
        max_profit_far : Series, float
            The FAR of the maximum profit building (constrained by the max_far and
            max_height from the input dataframe).
        max_profit :
            The profit for the maximum profit building (constrained by the max_far
            and max_height from the input dataframe).

        """
        c = self.config
        d = {}
        profit_df = pd.DataFrame()
        for parking_config in c.parking_configs:
            # this function gives the max profit development for the given
            # parking config need to iterate over parking configs to pick the
            # max profit config
            outdf = self._lookup_parking_cfg(form, parking_config, df, only_built,
                                             pass_through)
            d[parking_config] = outdf
            profit_df[parking_config] = outdf["max_profit"]

        # get the max_profit idx
        max_profit_ind = profit_df.idxmax(axis=1)

        if len(max_profit_ind) == 0:
            return pd.DataFrame()

        # make a new df of all the attributes from the max profit df
        l = []
        for parking_config in c.parking_configs:
            s = max_profit_ind[max_profit_ind == parking_config]
            # these are the rows that are most profitable with this
            # parking config
            tmpdf = d[parking_config].loc[s.index]
            tmpdf["parking_config"] = parking_config
            l.append(tmpdf)

        df = pd.concat(l)

        return df

    def _lookup_parking_cfg(self, form, parking_config, df, only_built=True,
                            pass_through=None):

        dev_info = self.dev_d[(form, parking_config)]

        cost_sqft_col = np.reshape(dev_info.ave_cost_sqft.values, (-1, 1))
        cost_sqft_index_col = np.reshape(dev_info.index.values, (-1, 1))

        parking_sqft_ratio = np.reshape(dev_info.parking_sqft_ratio.values, (-1, 1))
        heights = np.reshape(dev_info.height.values, (-1, 1))

        # don't really mean to edit the df that's passed in
        df = df.copy()

        c = self.config

        # weighted rent for this form
        df['weighted_rent'] = np.dot(df[c.uses], c.forms[form])

        # min between max_fars and max_heights
        df['max_far_from_heights'] = df.max_height / c.height_per_story * \
            c.parcel_coverage

        # now also minimize with max_dua from zoning - since this pro forma is
        # really geared toward per sqft metrics, this is a bit tricky.  dua
        # is converted to floorspace and everything just works (floor space
        # will get covered back to units in developer.pick() but we need to
        # test the profitability of the floorspace allowed by max_dua here.
        if 'max_dua' in df.columns:
            # if max_dua is in the data frame, ave_unit_size must also be there
            assert 'ave_unit_size' in df.columns
            # so this is the max_dua times the parcel size in acres, which gives
            # the number of units that are allowable on the parcel, times
            # by the average unit size which gives the square footage of
            # those units, divided by the building efficiency which is a
            # factor that indicates that the actual units are not the whole
            # FAR of the building and then divided by the parcel size again
            # in order to get FAR - I recognize that parcel_size actually
            # cancels here as it should, but the calc was hard to get right
            # and it's just so much more transparent to have it in there twice
            df['max_far_from_dua'] = df.max_dua * \
                (df.parcel_size / 43560) * \
                df.ave_unit_size / self.config.building_efficiency / \
                df.parcel_size
            df['min_max_fars'] = df[['max_far_from_heights', 'max_far',
                                     'max_far_from_dua']].min(axis=1)
        else:
            df['min_max_fars'] = df[['max_far_from_heights', 'max_far']].min(axis=1)

        if only_built:
            df = df.query('min_max_fars > 0 and parcel_size > 0')

        fars = np.repeat(cost_sqft_index_col, len(df.index), axis=1)

        # turn fars into nans which are not allowed by zoning
        # (so we can fillna with one of the other zoning constraints)
        fars[fars > df.min_max_fars.values + .01] = np.nan

        # same thing for heights
        heights = np.repeat(heights, len(df.index), axis=1)

        # turn heights into nans which are not allowed by zoning
        # (so we can fillna with one of the other zoning constraints)
        fars[heights > df.max_height.values + .01] = np.nan

        # parcel sizes * possible fars
        building_bulks = fars * df.parcel_size.values

        # cost to build the new building
        building_costs = building_bulks * cost_sqft_col

        # add cost to buy the current building
        total_costs = building_costs + df.land_cost.values

        # rent to make for the new building
        building_revenue = building_bulks * (1-parking_sqft_ratio) * \
            c.building_efficiency * df.weighted_rent.values / c.cap_rate

        # profit for each form
        profit = building_revenue - total_costs

        profit = profit.astype('float')
        profit[np.isnan(profit)] = -np.inf
        maxprofitind = np.argmax(profit, axis=0)

        def twod_get(indexes, arr):
            return arr[indexes, np.arange(indexes.size)].astype('float')

        outdf = pd.DataFrame({
            'building_sqft': twod_get(maxprofitind, building_bulks),
            'building_cost': twod_get(maxprofitind, building_costs),
            'parking_ratio': parking_sqft_ratio[maxprofitind].flatten(),
            'stories': twod_get(maxprofitind, heights) / c.height_per_story,
            'total_cost': twod_get(maxprofitind, total_costs),
            'building_revenue': twod_get(maxprofitind, building_revenue),
            'max_profit_far': twod_get(maxprofitind, fars),
            'max_profit': twod_get(maxprofitind, profit)
        }, index=df.index)

        if pass_through:
            outdf[pass_through] = df[pass_through]

        resratio = c.res_ratios[form]
        nonresratio = 1.0 - resratio
        outdf["residential_sqft"] = outdf.building_sqft * c.building_efficiency * resratio
        outdf["non_residential_sqft"] = outdf.building_sqft * c.building_efficiency * nonresratio

        if only_built:
            outdf = outdf.query('max_profit > 0').copy()
        else:
            outdf = outdf.loc[outdf.max_profit != -np.inf].copy()

        return outdf

    def _debug_output(self):
        """
        this code creates the debugging plots to understand
        the behavior of the hypothetical building model

        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        c = self.config

        df_d = self.dev_d
        keys = df_d.keys()
        keys.sort()
        for key in keys:
            logger.debug("\n" + str(key) + "\n")
            logger.debug(df_d[key])
        for form in self.config.forms:
            logger.debug("\n" + str(key) + "\n")
            logger.debug(self.get_ave_cost_sqft(form, "surface"))

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
                sumdf[parking_config] = df['ave_cost_sqft']
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
