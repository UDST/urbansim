import numpy as np
from urbansim.utils import misc
from urbansim.models import RegressionModel, MNLLocationChoiceModel, \
    GrowthRateTransition


def hedonic_estimate(df, cfgname):
    """
    Parameters
    ----------
    df : DataFrame
        The dataframe which contains the columns to use for the estimation.
    cfgname : string
        The name of the yaml config file which describes the hedonic model.
    """
    print "Running hedonic estimation\n"
    cfg = misc.config(cfgname)
    hm = RegressionModel.from_yaml(str_or_buffer=cfg)
    print hm.fit(df).summary()
    hm.to_yaml(str_or_buffer=cfg)


def hedonic_simulate(df, cfgname, outdf, outfname):
    """
    Parameters
    ----------
    df : DataFrame
        The dataframe which contains the columns to use for the estimation.
    cfgname : string
        The name of the yaml config file which describes the hedonic model.
    outdf : DataFrame
        The dataframe to write the simulated price/rent to.
    outfname : string
        The column name to write the simulated price/rent to.
    """
    print "Running hedonic simulation\n"
    cfg = misc.config(cfgname)
    hm = RegressionModel.from_yaml(str_or_buffer=cfg)
    price_or_rent = hm.predict(df)
    print price_or_rent.describe()
    outdf[outfname] = price_or_rent.reindex(outdf.index)


def lcm_estimate(choosers, chosen_fname, alternatives, cfgname):
    """
    Parameters
    ----------
    choosers : DataFrame
        A dataframe of rows of agents which have locations assigned.
    chosen_fname : string
        A string indicating the column in the choosers dataframe which
        gives which location the choosers have chosen.
    alternatives : DataFrame
        A dataframe of locations which should include the chosen locations
        from the choosers dataframe as well as some other locations from
        which to sample.  Values in choosers[chosen_fname] should index
        into the alternatives dataframe.
    cfgname : string
        The name of the yaml config file from which to read the location
        choice model.
    """
    print "Running location choice model estimation\n"
    cfg = misc.config(cfgname)
    lcm = MNLLocationChoiceModel.from_yaml(str_or_buffer=cfg)
    lcm.fit(choosers, alternatives, choosers[chosen_fname])
    lcm.report_fit()
    lcm.to_yaml(str_or_buffer=cfg)


def get_vacant_units(choosers, location_fname, locations, supply_fname):
    """
    This is a bit of a nuanced method for this skeleton which computes
    the vacant units from a building dataset for both households and jobs.

    Parameters
    ----------
    choosers : DataFrame
        A dataframe of agents doing the choosing.
    location_fname : string
        A string indicating a column in choosers which indicates the locations
        from the locations dataframe that these agents are located in.
    locations : DataFrame
        A dataframe of locations which the choosers are location in and which
        have a supply.
    supply_fname : string
        A string indicating a column in locations which is an integer value
        representing the number of agents that can be located at that location.
    """
    vacant_units = locations[supply_fname].sub(
        choosers.groupby(location_fname).size(), fill_value=0)
    print "There are %d total available units" % locations[supply_fname].sum()
    print "    and %d total choosers" % len(choosers.index)
    print "    but there are %d overfull buildings" % \
        len(vacant_units[vacant_units < 0].index)
    vacant_units = vacant_units[vacant_units > 0]
    alternatives = locations.loc[np.repeat(vacant_units.index,
                                 vacant_units.values.astype('int'))] \
        .reset_index()
    print "    for a total of %d empty units" % vacant_units.sum()
    print "    in %d buildings total in the region" % len(vacant_units)
    return alternatives


def _print_number_unplaced(df, fieldname="building_id"):
    """
    Just an internal function to use to compute and print info on the number
    of unplaced agents.
    """
    counts = df[fieldname].isnull().value_counts()
    count = 0 if True not in counts else counts[True]
    print "Total currently unplaced: %d" % count


def lcm_simulate(choosers, locations, cfgname, outdf, output_fname):
    """
    Simulate the location choices for the specified choosers

    Parameters
    ----------
    choosers : DataFrame
        A dataframe of agents doing the choosing.
    locations : DataFrame
        A dataframe of locations which the choosers are location in and which
        have a supply.
    cfgname : string
        The name of the yaml config file from which to read the location
        choice model.
    outdf : DataFrame
        The dataframe to write the simulated location to.
    outfname : string
        The column name to write the simulated location to.
    """
    print "Running location choice model simulation\n"
    cfg = misc.config(cfgname)
    lcm = MNLLocationChoiceModel.from_yaml(str_or_buffer=cfg)
    movers = choosers[choosers[output_fname].isnull()]
    new_units = lcm.predict(movers, locations)
    print "Assigned %d choosers to new units" % len(new_units.index)
    outdf[output_fname].loc[new_units.index] = \
        locations.loc[new_units.values][output_fname].values
    _print_number_unplaced(outdf, output_fname)


def simple_relocation(choosers, relocation_rate, fieldname='building_id'):
    """
    Parameters
    ----------
    choosers : DataFrame
        A dataframe of people which might be relocating.
    relocation_rate : float
        A number less than one describing the percent of rows to mark for
        relocation.
    fieldname : string
        The field name in the choosers dataframe to set to np.nan for those
        rows to mark for relocation.
    """
    print "Running relocation\n"
    _print_number_unplaced(choosers, fieldname)
    chooser_ids = np.random.choice(choosers.index, size=relocation_rate *
                                   len(choosers.index), replace=False)
    choosers[fieldname].loc[chooser_ids] = np.nan
    _print_number_unplaced(choosers, fieldname)


def simple_transition(dset, dfname, rate):
    """
    Parameters
    ----------
    choosers : dataset
        The dataset object, in order to write the resulting transitioned
        dataframe
    dfname : string
        The name of the dataframe in the dataset to read and write the the
        dataframe.
    rate : float
        The rate at which to grow the dataframe using a simple growth rate
        transition model.
    """
    transition = GrowthRateTransition(rate)
    df = dset.fetch(dfname)
    print "%d agents before transition" % len(df.index)
    df, added, copied, removed = transition.transition(df, None)
    print "%d agents after transition" % len(df.index)
    dset.save_tmptbl(dfname, df)
