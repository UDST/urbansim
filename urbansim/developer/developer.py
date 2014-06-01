import pandas as pd
import numpy as np


class Developer:

    def __init__(self, feasibility):
        """
        Pass the dataframe that is returned by feasibility here
        """
        self.f = feasibility

    @staticmethod
    def max_form(f, colname):
        """
        Assumes dataframe with hierarchical columns with first index equal to the
        use and second index equal to the attribtue

        e.g. f.columns equal to:
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

    def keep_form_with_max_profit(self):
        """
        This converts the dataframe, which shows all profitable forms,
        to the form with the greatest profit, so that more profitable
        forms outcompete less profitable forms.
        """
        f = self.dset.feasibility
        mu = self.max_form(f, "max_profit")
        indexes = [tuple(x) for x in mu.reset_index().values]
        df = f.stack(level=0).loc[indexes]
        df.index.names = ["parcel_id", "use"]
        df = df.reset_index(level=1)
        return df

    @staticmethod
    def compute_units_to_build(num_agents, num_units, target_vacancy):
        """
        Compute number of units to build to match target vacancy.

        Parameters:
        num_agents : int
            number of agents that need units in the region
        num_units : int
            number of units in buildings
        target_vacancy : float (0-1.0)
            target vacancy rate

        Returns : int
            the number of units that need to be built
        """
        print "Number of agents: %d" % num_agents
        print "Number of agent spaces: %d" % num_units
        assert target_vacancy < 1.0
        target_units = max(num_agents / (1 - target_vacancy) - num_units, 0)
        print "Current vacancy = %.2f" % (1 - num_agents / num_units)
        print "Target vacancy = %.2f, target of new units = %d" % (target_vacancy, target_units)
        return target_units

    def pick(self, form, target_units, parcel_size, ave_unit_size,
             current_units, max_parcel_size=200000, drop_after_build=True):
        """
        Choose the buildings from the list that are feasible to build in
        order to match the specified demand.

        Parameters
        ----------
        form : string
            One of the building forms from the pro forma specification -
            e.g. "residential" or "mixedresidential" - these are configuration
            parameters pass previously to the pro forma.
        target_units : int
            The number of units to build.  For non-residential buildings this
            should be passed as the number of job spaces that need to be created.
        parcel_size : series
            The size of the parcels.  This was passed to feasibility as well,
            but should be passed here as well.  Index should be parcel_ids.
        ave_unit_size : series
            The average unit size around each parcel - this is indexed
            by parcel, but is usually a disaggregated version of a zonal or
            accessibility aggregation.
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
        """

        df = self.feasibility[form]

        # feasible buildings only for this building type
        df = df[df.max_feasiblefar > 0]
        df["parcel_size"] = parcel_size
        df = df[df.parcel_size < max_parcel_size]
        df['new_sqft'] = df.parcel_size * df.max_feasiblefar
        df['new_units'] = np.round(df.new_sqft / ave_unit_size)
        df['current_units'] = current_units
        df['net_units'] = df.new_units - df.current_units

        print "Describe of net units\n", df.new_units.describe()

        choices = np.random.choice(df.index.values, size=len(df.index),
                                   replace=False,
                                   p=(df.max_profit.values / df.max_profit.sum()))
        net_units = df.net_units.loc[choices]
        tot_units = net_units.values.cumsum()
        ind = np.searchsorted(tot_units, target_units, side="right")
        build_idx = choices[:ind]

        print "Describe of buildings built\n", df.total_units.describe()
        print "Describe of profit\n", df.max_profit.describe()

        if drop_after_build:
            self.feasibility = self.feasibility.drop(build_idx)

        new_df = df.loc[build_idx]
        print new_df.index.name
        new_df.index.name = "parcel_id"
        return new_df.reset_index()

    @staticmethod
    def merge(old_df, new_df):
        """
        Merge two dataframes of buildings.  The old dataframe is
        usually the buildings dataset and the new dataframe is a modified
        (by the user) version of what is returned by the pick method.
        """
        maxind = np.max(old_df.index.values)
        new_df.index = new_df.index + maxind + 1
        concat_df = pd.concat([old_df, new_df], verify_integrity=True)
        print concat_df.index.name
        concat_df.index.name = 'building_id'
        return concat_df
