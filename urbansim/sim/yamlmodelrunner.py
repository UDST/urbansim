import numpy as np
import yaml
import pandas as pd
from numpy import random
from urbansim.utils import misc
import urbansim.sim.simulation as sim
from urbansim.models import RegressionModel, SegmentedRegressionModel, \
    MNLLocationChoiceModel, SegmentedMNLLocationChoiceModel, \
    GrowthRateTransition


# this is a single place to deal with nas
def deal_with_nas_for_est_or_sim(df, subset=None):
    if subset is not None:
        flds = filter(lambda x: x in df.columns, subset)
        df = df[flds]
    lenbefore = len(df)
    df = df.dropna(how='any')
    lenafter = len(df)
    if lenafter != lenbefore:
        print "Dropped %d rows because they contained nas" % (lenbefore-lenafter)
    return df


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
    model_type = yaml.load(open(cfg))["model_type"]
    if model_type == "regression":
        hm = RegressionModel.from_yaml(str_or_buffer=cfg)
        df = deal_with_nas_for_est_or_sim(df, hm.columns_used())
        print hm.fit(df, debug=True).summary()
        est_data = {"est_data": hm.est_data}
    if model_type == "segmented_regression":
        hm = SegmentedRegressionModel.from_yaml(str_or_buffer=cfg)
        df = deal_with_nas_for_est_or_sim(df, hm.columns_used())
        hm.min_segment_size = 10
        for k, v in hm.fit(df, debug=True).items():
            print "REGRESSION RESULTS FOR SEGMENT %s\n" % str(k)
            print v.summary()
            print
        est_data = {name: hm._group.models[name].est_data for name in hm._group.models}
    hm.to_yaml(str_or_buffer=cfg)
    return est_data


def hedonic_simulate(df, cfgname, outdf_name, outfname):
    """
    Parameters
    ----------
    df : DataFrame
        The dataframe which contains the columns to use for the estimation.
    cfgname : string
        The name of the yaml config file which describes the hedonic model.
    outdf_name : string
        The name of the dataframe to write the simulated price/rent to.
    outfname : string
        The column name to write the simulated price/rent to.
    """
    print "Running hedonic simulation\n"
    cfg = misc.config(cfgname)
    model_type = yaml.load(open(cfg))["model_type"]
    if model_type == "regression":
        hm = RegressionModel.from_yaml(str_or_buffer=cfg)
        df = deal_with_nas_for_est_or_sim(df, hm.columns_used())
    if model_type == "segmented_regression":
        hm = SegmentedRegressionModel.from_yaml(str_or_buffer=cfg)
        df = deal_with_nas_for_est_or_sim(df, hm.columns_used())
        hm.min_segment_size = 10
    price_or_rent = hm.predict(df)
    print price_or_rent.describe()
    print
    s = sim.get_table(outdf_name).get_column(outfname)
    s.loc[price_or_rent.index.values] = price_or_rent
    sim.add_column(outdf_name, outfname, s)


def _to_frame_get_fields(model_type, model, output_fname, df):
    add_flds = [output_fname]
    if model_type == "segmented_locationchoice":
        add_flds += [model.segmentation_col]
    flds = model.columns_used()+add_flds
    print "The following fields are used by this model:", flds
    print
    df = df.to_frame(flds)
    return deal_with_nas_for_est_or_sim(df)


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
    model_type = yaml.load(open(cfg))["model_type"]
    if model_type == "locationchoice":
        lcm = MNLLocationChoiceModel.from_yaml(str_or_buffer=cfg)
        choosers = _to_frame_get_fields(model_type, lcm, chosen_fname, choosers)
        alternatives = deal_with_nas_for_est_or_sim(alternatives, lcm.columns_used())
        lcm.fit(choosers, alternatives, choosers[chosen_fname])
        lcm.report_fit()
    elif model_type == "segmented_locationchoice":
        lcm = SegmentedMNLLocationChoiceModel.from_yaml(str_or_buffer=cfg)
        choosers = _to_frame_get_fields(model_type, lcm, chosen_fname, choosers)
        alternatives = deal_with_nas_for_est_or_sim(alternatives, lcm.columns_used())
        lcm.fit(choosers, alternatives, choosers[chosen_fname])
        for k, v in lcm._group.models.items():
            print "LCM RESULTS FOR SEGMENT %s\n" % str(k)
            v.report_fit()
            print
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
        choosers[location_fname].value_counts(), fill_value=0)
    print "There are %d total available units" % locations[supply_fname].sum()
    print "    and %d total choosers" % len(choosers.index)
    print "    but there are %d overfull buildings" % \
        len(vacant_units[vacant_units < 0])
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
    counts = (df[fieldname] == -1).value_counts()
    count = 0 if True not in counts else counts[True]
    print "Total currently unplaced: %d" % count


def lcm_simulate(choosers, locations, cfgname, outdf_name, output_fname,
                 location_ratio=2.0):
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
    outdf_name : string
        The name of the dataframe to write the simulated location to.
    outfname : string
        The column name to write the simulated location to.
    location_ratio : float
        Above the location ratio (default of 2.0) of locations to choosers, the
        locations will be sampled to meet this ratio (for performance reasons).
    """
    print "Running location choice model simulation\n"
    outdf = sim.get_table(outdf_name)
    cfg = misc.config(cfgname)
    model_type = yaml.load(open(cfg))["model_type"]

    if model_type == "locationchoice":
        lcm = MNLLocationChoiceModel.from_yaml(str_or_buffer=cfg)
    elif model_type == "segmented_locationchoice":
        lcm = SegmentedMNLLocationChoiceModel.from_yaml(str_or_buffer=cfg)

    choosers = _to_frame_get_fields(model_type, lcm, output_fname, choosers)

    movers = choosers[choosers[output_fname] == -1]

    locations = deal_with_nas_for_est_or_sim(locations, lcm.columns_used()+[output_fname])

    if len(locations) > len(movers) * location_ratio:
        print "Location ratio exceeded: %d locations and only %d choosers" % \
              (len(locations), len(movers))
        idxes = random.choice(locations.index, size=len(movers) * location_ratio,
                              replace=False)
        locations = locations.loc[idxes]
        print "  after sampling %d locations are available\n" % len(locations)

    new_units = lcm.predict(movers, locations, debug=True)
    print "Assigned %d choosers to new units" % len(new_units.index)
    if len(new_units) == 0:
        return
    s = sim.get_table(outdf_name).get_column(output_fname)
    s.loc[new_units.index] = \
        locations.loc[new_units.values][output_fname].values
    sim.add_column(outdf_name, output_fname,  s)
    _print_number_unplaced(outdf, output_fname)

    if model_type == "locationchoice":
        sim_pdf = {"sim_pdf": lcm.sim_pdf}
    elif model_type == "segmented_locationchoice":
        sim_pdf = {name: lcm._group.models[name].sim_pdf for name in lcm._group.models}

    # go back to the buildings from units
    sim_pdf = pd.concat(sim_pdf.values(), keys=sim_pdf.keys(), axis=1)
    sim_pdf.index = locations.loc[sim_pdf.index][output_fname].values
    sim_pdf = sim_pdf.groupby(level=0).first()

    return sim_pdf


def simple_relocation(choosers, relocation_rate, fieldname='building_id'):
    """
    Parameters
    ----------
    choosers_name : string
        A name of the dataframe of people which might be relocating.
    relocation_rate : float
        A number less than one describing the percent of rows to mark for
        relocation.
    fieldname : string
        The field name in the choosers dataframe to set to -1 for those
        rows to mark for relocation.
    """
    choosers_name = choosers
    choosers = sim.get_table(choosers)
    print "Total agents: %d" % len(choosers[fieldname])
    _print_number_unplaced(choosers, fieldname)
    chooser_ids = np.random.choice(choosers.index, size=int(relocation_rate *
                                   len(choosers)), replace=False)
    s = choosers[fieldname]
    print "Assinging for relocation..."
    s.loc[chooser_ids] = -1
    sim.add_column(choosers_name, fieldname, s)
    _print_number_unplaced(choosers, fieldname)


def simple_transition(dfname, rate):
    """
    Parameters
    ----------
    dfname : string
        The name of the dataframe in the dataset to read and write the the
        dataframe.
    rate : float
        The rate at which to grow the dataframe using a simple growth rate
        transition model.
    """
    transition = GrowthRateTransition(rate)
    tbl = sim.get_table(dfname)
    df = tbl.to_frame(tbl.local_columns)
    print "%d agents before transition" % len(df.index)
    df, added, copied, removed = transition.transition(df, None)
    print "%d agents after transition" % len(df.index)
    sim.add_table(dfname, df)