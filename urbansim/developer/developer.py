import pandas as pd
import numpy as np


class Developer(object):
    """
    Pass the dataframe that is returned by feasibility here

    Can also be a dictionary where keys are building forms and values are
    the individual data frames returned by the proforma lookup routine.

    """

    def __init__(self, feasibility):
        if isinstance(feasibility, dict):
            feasibility = pd.concat(feasibility.values(), keys=feasibility.keys(), axis=1)
        self.feasibility = feasibility

    @staticmethod
    def _max_form(f, colname):
        """
        Assumes dataframe with hierarchical columns with first index equal to the
        use and second index equal to the attribute.

        e.g. f.columns equal to::

            mixedoffice   building_cost
                          building_revenue
                          building_size
                          max_profit
                          max_profit_far
                          total_cost
            industrial    building_cost
                          building_revenue
                          building_size
                          max_profit
                          max_profit_far
                          total_cost
        """
        df = f.stack(level=0)[[colname]].stack().unstack(level=1).reset_index(level=1, drop=True)
        return df.idxmax(axis=1)

    def keep_form_with_max_profit(self, forms=None):
        """
        This converts the dataframe, which shows all profitable forms,
        to the form with the greatest profit, so that more profitable
        forms outcompete less profitable forms.

        Parameters
        ----------
        forms: list of strings
            List of forms which compete which other.  Can leave some out.

        Returns
        -------
        Nothing.  Goes from a multi-index to a single index with only the
        most profitable form.
        """
        f = self.feasibility

        if forms is not None:
            f = f[forms]

        mu = self._max_form(f, "max_profit")
        indexes = [tuple(x) for x in mu.reset_index().values]
        df = f.stack(level=0).loc[indexes]
        df.index.names = ["parcel_id", "form"]
        df = df.reset_index(level=1)
        return df

    @staticmethod
    def compute_units_to_build(num_agents, num_units, target_vacancy):
        """
        Compute number of units to build to match target vacancy.

        Parameters
        ----------
        num_agents : int
            number of agents that need units in the region
        num_units : int
            number of units in buildings
        target_vacancy : float (0-1.0)
            target vacancy rate

        Returns
        -------
        number_of_units : int
            the number of units that need to be built
        """
        print "Number of agents: {:,}".format(num_agents)
        print "Number of agent spaces: {:,}".format(int(num_units))
        assert target_vacancy < 1.0
        target_units = int(max(num_agents / (1 - target_vacancy) - num_units, 0))
        print "Current vacancy = {:.2f}".format(1 - num_agents / float(num_units))
        print "Target vacancy = {:.2f}, target of new units = {:,}".\
            format(target_vacancy, target_units)
        return target_units

    def pick(self, form, target_units, parcel_size, ave_unit_size,
             current_units, max_parcel_size=200000, min_unit_size=400,
             drop_after_build=True, residential=True, bldg_sqft_per_job=400.0,
             profit_to_prob_func=None):
        """
        Choose the buildings from the list that are feasible to build in
        order to match the specified demand.

        Parameters
        ----------
        form : string or list
            One or more of the building forms from the pro forma specification -
            e.g. "residential" or "mixedresidential" - these are configuration
            parameters passed previously to the pro forma.  If more than one form
            is passed the forms compete with each other (based on profitability)
            for which one gets built in order to meet demand.
        target_units : int
            The number of units to build.  For non-residential buildings this
            should be passed as the number of job spaces that need to be created.
        parcel_size : series
            The size of the parcels.  This was passed to feasibility as well,
            but should be passed here as well.  Index should be parcel_ids.
        ave_unit_size : series
            The average residential unit size around each parcel - this is
            indexed by parcel, but is usually a disaggregated version of a
            zonal or accessibility aggregation.
        bldg_sqft_per_job : float (default 400.0)
            The average square feet per job for this building form.
        min_unit_size : float
            Values less than this number in ave_unit_size will be set to this
            number.  Deals with cases where units are currently not built.
        current_units : series
            The current number of units on the parcel.  Is used to compute the
            net number of units produced by the developer model.  Many times
            the developer model is redeveloping units (demolishing them) and
            is trying to meet a total number of net units produced.
        max_parcel_size : float
            Parcels larger than this size will not be considered for
            development - usually large parcels should be specified manually
            in a development projects table.
        drop_after_build : bool
            Whether or not to drop parcels from consideration after they
            have been chosen for development.  Usually this is true so as
            to not develop the same parcel twice.
        residential: bool
            If creating non-residential buildings set this to false and
            developer will fill in job_spaces rather than residential_units
        profit_to_prob_func: function
            As there are so many ways to turn the development feasibility
            into a probability to select it for building, the user may pass
            a function which takes the feasibility dataframe and returns
            a series of probabilities.  If no function is passed, the behavior
            of this method will not change

        Returns
        -------
        None if thar are no feasible buildings
        new_buildings : dataframe
            DataFrame of buildings to add.  These buildings are rows from the
            DataFrame that is returned from feasibility.
        """

        if len(self.feasibility) == 0:
            # no feasible buildings, might as well bail
            return

        if form is None:
            df = self.feasibility
        elif isinstance(form, list):
            df = self.keep_form_with_max_profit(form)
        else:
            df = self.feasibility[form]

        # feasible buildings only for this building type
        df = df[df.max_profit_far > 0]
        ave_unit_size[ave_unit_size < min_unit_size] = min_unit_size
        df["ave_unit_size"] = ave_unit_size
        df["parcel_size"] = parcel_size
        df['current_units'] = current_units
        df = df[df.parcel_size < max_parcel_size]

        df['residential_units'] = (df.residential_sqft / df.ave_unit_size).round()
        df['job_spaces'] = (df.non_residential_sqft / bldg_sqft_per_job).round()

        if residential:
            df['net_units'] = df.residential_units - df.current_units
        else:
            df['net_units'] = df.job_spaces - df.current_units
        df = df[df.net_units > 0]

        if len(df) == 0:
            print "WARNING THERE ARE NO FEASIBLE BUILDING TO CHOOSE FROM"
            return

        # print "Describe of net units\n", df.net_units.describe()
        print "Sum of net units that are profitable: {:,}".\
            format(int(df.net_units.sum()))

        if profit_to_prob_func:
            p = profit_to_prob_func(df)
        else:
            df['max_profit_per_size'] = df.max_profit / df.parcel_size
            p = df.max_profit_per_size.values / df.max_profit_per_size.sum()

        if df.net_units.sum() < target_units:
            print "WARNING THERE WERE NOT ENOUGH PROFITABLE UNITS TO " \
                  "MATCH DEMAND"
            build_idx = df.index.values
        elif target_units <= 0:
            build_idx = []
        else:
            # we don't know how many developments we will need, as they differ in net_units.
            # If all developments have net_units of 1 than we need target_units of them.
            # So we choose the smaller of available developments and target_units.
            choices = np.random.choice(df.index.values, size=min(len(df.index), target_units),
                                       replace=False, p=p)
            tot_units = df.net_units.loc[choices].values.cumsum()
            ind = int(np.searchsorted(tot_units, target_units, side="left")) + 1
            build_idx = choices[:ind]

        if drop_after_build:
            self.feasibility = self.feasibility.drop(build_idx)

        new_df = df.loc[build_idx]
        new_df.index.name = "parcel_id"
        return new_df.reset_index()

    @staticmethod
    def merge(old_df, new_df, return_index=False):
        """
        Merge two dataframes of buildings.  The old dataframe is
        usually the buildings dataset and the new dataframe is a modified
        (by the user) version of what is returned by the pick method.

        Parameters
        ----------
        old_df : dataframe
            Current set of buildings
        new_df : dataframe
            New buildings to add, usually comes from this module
        return_index : bool
            If return_index is true, this method will return the new
            index of new_df (which changes in order to create a unique
            index after the merge)

        Returns
        -------
        df : dataframe
            Combined DataFrame of buildings, makes sure indexes don't overlap
        index : pd.Index
            If and only if return_index is True, return the new index for the
            new_df dataframe (which changes in order to create a unique index
            after the merge)
        """
        maxind = np.max(old_df.index.values)
        new_df = new_df.reset_index(drop=True)
        new_df.index = new_df.index + maxind + 1
        concat_df = pd.concat([old_df, new_df], verify_integrity=True)
        concat_df.index.name = 'building_id'

        if return_index:
            return concat_df, new_df.index

        return concat_df
