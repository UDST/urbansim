__author__ = 'JMartinez'
import os

import numpy as np
import orca
import pandas as pd
from urbansim.models import RegressionModel, SegmentedRegressionModel, \
    MNLDiscreteChoiceModel, SegmentedMNLDiscreteChoiceModel, \
    GrowthRateTransition, RelocationModel, TabularTotalsTransition, supplydemand
from urbansim.developer import sqftproforma, developer
from urbansim.models.transition import DRCOGHouseholdTransitionModel
from urbansim.utils import misc
from urbansim.utils import yamlio
from sqlalchemy import create_engine




def get_run_filename():
    return os.path.join(misc.runs_dir(), "run%d.h5" % misc.get_run_number())


def change_store(store_name):
    orca.add_injectable(
        "store",
        pd.HDFStore(os.path.join(misc.data_dir(), store_name), mode="r"))


def change_scenario(scenario):
    assert scenario in orca.get_injectable("scenario_inputs"), \
        "Invalid scenario name"
    print "Changing scenario to '%s'" % scenario
    orca.add_injectable("scenario", scenario)


def conditional_upzone(scenario, attr_name, upzone_name):
    scenario_inputs = orca.get_injectable("scenario_inputs")
    zoning_baseline = orca.get_table(
        scenario_inputs["baseline"]["zoning_table_name"])
    attr = zoning_baseline[attr_name]
    if scenario != "baseline":
        zoning_scenario = orca.get_table(
            scenario_inputs[scenario]["zoning_table_name"])
        upzone = zoning_scenario[upzone_name].dropna()
        attr = pd.concat([attr, upzone], axis=1).max(skipna=True, axis=1)
    return attr


def enable_logging():
    from urbansim.utils import logutil
    logutil.set_log_level(logutil.logging.INFO)
    logutil.log_to_stream()


def deal_with_nas(df):
    df_cnt = len(df)
    fail = False

    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.columns:
        s_cnt = df[col].count()
        if df_cnt != s_cnt:
            fail = True
            print "Found %d nas or inf (out of %d) in column %s" % \
                  (df_cnt-s_cnt, df_cnt, col)

    assert not fail, "NAs were found in dataframe, please fix"
    return df


def fill_nas_from_config(dfname, df):
    df_cnt = len(df)
    fillna_config = orca.get_injectable("fillna_config")
    fillna_config_df = fillna_config[dfname]
    for fname in fillna_config_df:
        filltyp, dtyp = fillna_config_df[fname]
        s_cnt = df[fname].count()
        fill_cnt = df_cnt - s_cnt
        if filltyp == "zero":
            val = 0
        elif filltyp == "mode":
            val = df[fname].dropna().value_counts().idxmax()
        elif filltyp == "median":
            val = df[fname].dropna().quantile()
        else:
            assert 0, "Fill type not found!"
        print "Filling column {} with value {} ({} values)".\
            format(fname, val, fill_cnt)
        df[fname] = df[fname].fillna(val).astype(dtyp)
    return df


def to_frame(tables, cfg, additional_columns=[]):
    cfg = yaml_to_class(cfg).from_yaml(str_or_buffer=cfg)
    tables = [t for t in tables if t is not None]
    columns = misc.column_list(tables, cfg.columns_used()) + additional_columns
    if len(tables) > 1:
        df = orca.merge_tables(target=tables[0].name,
                               tables=tables, columns=columns)
    else:
        df = tables[0].to_frame(columns)
    try:
        df = deal_with_nas(df)
    except:
        df.fillna(0, inplace=True)
    return df


def yaml_to_class(cfg):
    import yaml
    model_type = yaml.load(open(cfg))["model_type"]
    return {
        "regression": RegressionModel,
        "segmented_regression": SegmentedRegressionModel,
        "discretechoice": MNLDiscreteChoiceModel,
        "segmented_discretechoice": SegmentedMNLDiscreteChoiceModel
    }[model_type]


def hedonic_estimate(cfg, tbl, nodes):
    cfg = misc.config(cfg)
    df = to_frame([tbl, nodes], cfg)
    return yaml_to_class(cfg).fit_from_cfg(df, cfg)


def hedonic_simulate(cfg, buildings, parcels, zones, out_fname):
    cfg = misc.config(cfg)
    df = to_frame([buildings, parcels, zones], cfg)
    price_or_rent, _ = yaml_to_class(cfg).predict_from_cfg(df, cfg)
    buildings.update_col_from_series(out_fname, price_or_rent)


def lcm_estimate(cfg, choosers, chosen_fname, buildings, nodes):
    cfg = misc.config(cfg)
    choosers = to_frame([choosers], cfg, additional_columns=[chosen_fname])
    alternatives = to_frame([buildings, nodes], cfg)
    return yaml_to_class(cfg).fit_from_cfg(choosers,
                                           chosen_fname,
                                           alternatives,
                                           cfg)

def elcm_simulate(cfg, choosers, buildings, parcels, zones, out_fname,
                 supply_fname, vacant_fname):

    """
    Simulate the location choices for the specified choosers
    Parameters
    ----------
    cfg : string
        The name of the yaml config file from which to read the location
        choice model.
    choosers : DataFrame
        A dataframe of agents doing the choosing.
    buildings : DataFrame
        A dataframe of buildings which the choosers are locating in and which
        have a supply.
    nodes : DataFrame
        A land use dataset to give neighborhood info around the buildings -
        will be joined to the buildings.
    out_dfname : string
        The name of the dataframe to write the simulated location to.
    out_fname : string
        The column name to write the simulated location to.
    supply_fname : string
        The string in the buildings table that indicates the amount of
        available units there are for choosers, vacant or not.
    vacant_fname : string
        The string in the buildings table that indicates the amount of vacant
        units there will be for choosers.
    """
    cfg = misc.config(cfg)

    chooser_cols = [out_fname, 'zone_id', 'county_id', 'employees']

    #choosers_df = to_frame([choosers, buildings, parcels, zones], cfg, additional_columns=chooser_cols)
    #TODO add join parameters to orca.merge_tables
    choosers_df = to_frame([choosers], cfg, additional_columns=[out_fname])
    locations_df = to_frame([buildings, parcels, zones], cfg, additional_columns=[supply_fname, vacant_fname, 'county_id', 'zone_id'])
    #update choosers_df county_id to match that of transition model
    choosers_df.loc[:, 'county_id'] = orca.get_table('updated_emp').county_id

    available_units = buildings[supply_fname]
    vacant_units = buildings[vacant_fname]

    print "There are %d total available units" % available_units.sum()
    print "    and %d total choosers" % len(choosers)
    print "    but there are %d overfull buildings" % \
          len(vacant_units[vacant_units < 0])

    vacant_units = vacant_units[vacant_units > 0]
    units = locations_df

    print "    for a total of %d temporarily empty units" % vacant_units.sum()
    print "    in %d buildings total in the region" % len(vacant_units)

    movers = choosers_df[choosers_df[out_fname] == -1]

    if len(movers) > vacant_units.sum():
        print "WARNING: Not enough locations for movers"
        print "    reducing locations to size of movers for performance gain"
        movers = movers.head(vacant_units.sum())

    new_units, _ = yaml_to_class(cfg).predict_from_cfg(movers, units, cfg)

    # new_units returns nans when there aren't enough units,
    # get rid of them and they'll stay as -1s
    new_units = new_units.dropna()

    # go from units back to buildings
    #new_buildings = pd.Series(units.loc[new_units.values][out_fname].values,
    #                          index=new_units.index)

    print locations_df.county_id.loc[new_units].value_counts()
    choosers.update_col_from_series(out_fname, new_units.groupby(level=0).first())
    _print_number_unplaced(choosers, out_fname)

    vacant_units = buildings[vacant_fname]
    vacant_units = vacant_units[vacant_units > 0]
    print "    and there are now %d empty units" % vacant_units.sum()
    vacant_units = buildings[vacant_fname]
    print "    and %d overfull buildings" % len(vacant_units[vacant_units < 0])



def lcm_simulate(cfg, choosers, buildings, parcels, zones, out_fname,
                 supply_fname, vacant_fname):
    """
    Simulate the location choices for the specified choosers
    Parameters
    ----------
    cfg : string
        The name of the yaml config file from which to read the location
        choice model.
    choosers : DataFrame
        A dataframe of agents doing the choosing.
    buildings : DataFrame
        A dataframe of buildings which the choosers are locating in and which
        have a supply.
    nodes : DataFrame
        A land use dataset to give neighborhood info around the buildings -
        will be joined to the buildings.
    out_dfname : string
        The name of the dataframe to write the simulated location to.
    out_fname : string
        The column name to write the simulated location to.
    supply_fname : string
        The string in the buildings table that indicates the amount of
        available units there are for choosers, vacant or not.
    vacant_fname : string
        The string in the buildings table that indicates the amount of vacant
        units there will be for choosers.
    """
    cfg = misc.config(cfg)
    yaml_dict = yamlio.yaml_to_dict(str_or_buffer=cfg)

    chooser_cols = [out_fname, 'zone_id', 'county_id']
    choosers_df = to_frame([choosers], cfg, additional_columns=[out_fname])
    locations_df = to_frame([buildings, parcels, zones], cfg, additional_columns=[supply_fname, vacant_fname, 'county_id', 'zone_id'])
    #update choosers_df county_id to match that of transition model
    choosers_df.loc[:, 'county_id'] = orca.get_table('updated_hh').county_id

    available_units = buildings[supply_fname]
    vacant_units = buildings[vacant_fname]

    print "There are %d total available units" % available_units.sum()
    print "    and %d total choosers" % len(choosers)
    print "    but there are %d overfull buildings" % \
          len(vacant_units[vacant_units < 0])

    vacant_units = vacant_units[vacant_units > 0]
    units = locations_df.loc[np.repeat(vacant_units.index.values,
                             vacant_units.values.astype('int'))].reset_index()

    print "    for a total of %d temporarily empty units" % vacant_units.sum()
    print "    in %d buildings total in the region" % len(vacant_units)

    movers = choosers_df[choosers_df[out_fname] == -1]

    if len(movers) > vacant_units.sum():
        print "WARNING: Not enough locations for movers"
        print "    reducing locations to size of movers for performance gain"
        movers = movers.head(vacant_units.sum())

    new_units, _ = yaml_to_class(cfg).predict_from_cfg(movers, units, cfg)

    # new_units returns nans when there aren't enough units,
    # get rid of them and they'll stay as -1s
    new_units = new_units.dropna()

    # go from units back to buildings
    new_buildings = pd.Series(units.loc[new_units.values][out_fname].values,
                              index=new_units.index)

    print locations_df.county_id.loc[new_buildings].value_counts()
    choosers.update_col_from_series(out_fname, new_buildings)
    _print_number_unplaced(choosers, out_fname)

    vacant_units = buildings[vacant_fname]
    vacant_units = vacant_units[vacant_units > 0]
    print "    and there are now %d empty units" % vacant_units.sum()
    vacant_units = buildings[vacant_fname]
    print "    and %d overfull buildings" % len(vacant_units[vacant_units < 0])


def simple_relocation(choosers, relocation_rate, fieldname):
    print "Total agents: %d" % len(choosers)
    _print_number_unplaced(choosers, fieldname)

    print "Assinging for relocation..."
    chooser_ids = np.random.choice(choosers.index, size=int(relocation_rate *
                                   len(choosers)), replace=False)
    choosers.update_col_from_series(fieldname,
                                    pd.Series(-1, index=chooser_ids))

    _print_number_unplaced(choosers, fieldname)

def relocation_model(choosers, control_table, field_name):
    print "Total agents: %d" % len(choosers)
    _print_number_unplaced(choosers, field_name)

    print "Assigning for relocation..."
    hh_relo_model = RelocationModel(control_table.to_frame())
    movers = hh_relo_model.find_movers(choosers.to_frame())

    print "%d agents relocating" % len(movers)
    choosers.update_col_from_series(field_name, pd.Series(-1, index=movers))




def emp_relocation_model(choosers, control_table, field_name):
    print "Total agents: %d" % len(choosers)
    _print_number_unplaced(choosers, field_name)

    print "Assigning for relocation..."
    emp_relo_model = RelocationModel(control_table.to_frame(), rate_column='job_relocation_probability')
    movers = emp_relo_model.find_movers(choosers.to_frame())

    print "%d agents relocating" % len(movers)
    choosers.update_col_from_series(field_name, pd.Series(-1, index=movers))

def simple_transition(tbl, rate, location_fname):
    transition = GrowthRateTransition(rate)
    df = tbl.to_frame(tbl.local_columns)

    print "%d agents before transition" % len(df.index)
    df, added, copied, removed = transition.transition(df, None)
    print "%d agents after transition" % len(df.index)

    df.loc[added, location_fname] = -1
    orca.add_table(tbl.name, df)

def hh_transition(households, tbl, location_fname, year):
    migration = orca.get_table('migration_data').to_frame()
    pdf = migration['net_migration'] / migration.net_migration.sum()

    tran = DRCOGHouseholdTransitionModel(tbl.to_frame(), 'total_number_of_households',
                                         prob_dist = [pdf], migration_data=migration)

    cols = orca.get_table('households').local_columns
    add_cols = ['zone_id','county_id']
    cols = cols + add_cols
    df = orca.merge_tables('households', tables=['households','buildings','parcels'],columns=cols)

    print "%d households before transition" % len(df.index)
    df, added, copied, removed = tran.transition(df, year, [pdf])
    print "%d households after transition" % len(df.index)

    df.loc[added, location_fname] = -1
    orca.add_table('households', df.loc[:, orca.get_table('households').local_columns])
    orca.add_table('updated_hh', df, cache=True, cache_scope='iteration')

def emp_transition(tbl, location_fname, year):
    #tran = TabularFilteredTotalsTransition(tbl.to_frame(), 'total_number_of_jobs', ['sector_id_six','home_based_status'],
    #                                       accounting_column='employees')
    tran = TabularTotalsTransition(tbl.to_frame(), 'total_number_of_jobs', accounting_column='employees')
    cols = orca.get_table('establishments').local_columns
    add_cols = ['zone_id','county_id','sector_id_six']
    cols = cols + add_cols
    df = orca.merge_tables('establishments', tables=['establishments','buildings','parcels'],columns=cols)

    print "%d establishments with %d employees before transition" % (len(df.index), df.employees.sum())
    df, added, copied, removed = tran.transition(df, year)
    print "%d establishments with %d employees after transition" % (len(df.index), df.employees.sum())

    df.loc[added, location_fname] = -1
    orca.add_table('establishments', df.loc[:, orca.get_table('establishments').local_columns])
    orca.add_table('updated_emp', df, cache=True, cache_scope='iteration')


def _print_number_unplaced(df, fieldname):
    print "Total currently unplaced: %d" % \
          df[fieldname].value_counts().get(-1, 0)


def run_feasibility(parcels, parcel_price_callback,
                    parcel_use_allowed_callback, residential_to_yearly=True):
    """
    Execute development feasibility on all parcels
    Parameters
    ----------
    parcels : DataFrame Wrapper
        The data frame wrapper for the parcel data
    parcel_price_callback : function
        A callback which takes each use of the pro forma and returns a series
        with index as parcel_id and value as yearly_rent
    parcel_use_allowed_callback : function
        A callback which takes each form of the pro forma and returns a series
        with index as parcel_id and value and boolean whether the form
        is allowed on the parcel
    residential_to_yearly : boolean (default true)
        Whether to use the cap rate to convert the residential price from total
        sales price per sqft to rent per sqft
    Returns
    -------
    Adds a table called feasibility to the sim object (returns nothing)
    """
    pf = sqftproforma.SqFtProForma()

    df = orca.merge_tables('parcels', tables=['parcels', 'fars'])

    # add prices for each use
    for use in pf.config.uses:
        df[use] = parcel_price_callback(use)

    #There are 179 zones without buildings. These zones will get county average prices for each use
    #No buildings
    df_dropna = df.replace([np.inf, -np.inf], np.nan).dropna()
    subset = df.loc[(df.residential.isnull())&(df.retail.isnull())&(df.industrial.isnull())&(df.office.isnull())]
    df.loc[subset.index, 'residential'] = df_dropna.groupby('county_id').residential.mean()[subset.county_id].values
    df.loc[subset.index, 'retail'] = df_dropna.groupby('county_id').retail.mean()[subset.county_id].values
    df.loc[subset.index, 'industrial'] = df_dropna.groupby('county_id').industrial.mean()[subset.county_id].values
    df.loc[subset.index, 'office'] = df_dropna.groupby('county_id').office.mean()[subset.county_id].values

    #No residential buildings
    subset = df.loc[(df.residential.isnull())]
    df.loc[subset.index, 'residential'] = df_dropna.groupby('county_id').residential.mean()[subset.county_id].values
    subset = df.loc[(np.isinf(df.residential.values))]
    df.loc[subset.index, 'residential'] = df_dropna.groupby('county_id').residential.mean()[subset.county_id].values


    #No retail buildings
    subset = df.loc[(df.retail.isnull())]
    df.loc[subset.index, 'retail'] = df_dropna.groupby('county_id').retail.mean()[subset.county_id].values

    #No industrial buildings
    subset = df.loc[(df.industrial.isnull())]
    df.loc[subset.index, 'industrial'] = df_dropna.groupby('county_id').industrial.mean()[subset.county_id].values

    #No office buildings
    subset = df.loc[(df.office.isnull())]
    df.loc[subset.index, 'office'] = df_dropna.groupby('county_id').office.mean()[subset.county_id].values

    #elbert county has no non_res buildings so it gets regional averages
    subset = df.loc[(df.retail.isnull())]
    df.loc[subset.index, 'retail'] = df.retail.mean()

    subset = df.loc[(df.industrial.isnull())]
    df.loc[subset.index, 'industrial'] = df.industrial.mean()

    subset = df.loc[(df.office.isnull())]
    df.loc[subset.index, 'office'] = df.office.mean()

    #deal with fars
    #environmental constraints
    df['far'] = df['far']*(1-df.prop_constrained)
    #ugb policies
    #if restricting fars on ugb policies this would go here

    #rename far column to max_far
    df.rename(columns={'far':'max_far'}, inplace=True)

    #max height series
    df['max_height'] = np.nan



    # convert from cost to yearly rent
    if residential_to_yearly:
        df["residential"] *= pf.config.cap_rate

    print "Describe of the yearly rent by use"
    print df[pf.config.uses].describe()

    d = {}
    for form in pf.config.forms:
        print "Computing feasibility for form %s" % form
        d[form] = pf.lookup(form, df[parcel_use_allowed_callback(form)])

    far_predictions = pd.concat(d.values(), keys=d.keys(), axis=1)

    orca.add_table("feasibility", far_predictions)


def run_developer(forms, agents, buildings, supply_fname, parcel_size,
                  ave_unit_size, total_units, feasibility, year=None,
                  target_vacancy=.1, form_to_btype_callback=None,
                  add_more_columns_callback=None, max_parcel_size=200000,
                  residential=True, bldg_sqft_per_job=400.0):
    """
    Run the developer model to pick and build buildings
    Parameters
    ----------
    forms : string or list of strings
        Passed directly dev.pick
    agents : DataFrame Wrapper
        Used to compute the current demand for units/floorspace in the area
    buildings : DataFrame Wrapper
        Used to compute the current supply of units/floorspace in the area
    supply_fname : string
        Identifies the column in buildings which indicates the supply of
        units/floorspace
    parcel_size : Series
        Passed directly to dev.pick
    ave_unit_size : Series
        Passed directly to dev.pick - average residential unit size
    total_units : Series
        Passed directly to dev.pick - total current residential_units /
        job_spaces
    feasibility : DataFrame Wrapper
        The output from feasibility above (the table called 'feasibility')
    year : int
        The year of the simulation - will be assigned to 'year_built' on the
        new buildings
    target_vacancy : float
        The target vacancy rate - used to determine how much to build
    form_to_btype_callback : function
        Will be used to convert the 'forms' in the pro forma to
        'building_type_id' in the larger model
    add_more_columns_callback : function
        Takes a dataframe and returns a dataframe - is used to make custom
        modifications to the new buildings that get added
    max_parcel_size : float
        Passed directly to dev.pick - max parcel size to consider
    residential : boolean
        Passed directly to dev.pick - switches between adding/computing
        residential_units and job_spaces
    bldg_sqft_per_job : float
        Passed directly to dev.pick - specified the multiplier between
        floor spaces and job spaces for this form (does not vary by parcel
        as ave_unit_size does)
    Returns
    -------
    Writes the result back to the buildings table (returns nothing)
    """

    dev = developer.Developer(feasibility.to_frame())

    target_units = dev.\
        compute_units_to_build(len(agents),
                               buildings[supply_fname].sum(),
                               target_vacancy)

    print "{:,} feasible buildings before running developer".format(
          len(dev.feasibility))

    new_buildings = dev.pick(forms,
                             target_units,
                             parcel_size,
                             ave_unit_size,
                             total_units,
                             max_parcel_size=max_parcel_size,
                             drop_after_build=True,
                             residential=residential,
                             bldg_sqft_per_job=bldg_sqft_per_job)

    orca.add_table("feasibility", dev.feasibility)

    if new_buildings is None:
        return

    if year is not None:
        new_buildings["year_built"] = year

    if not isinstance(forms, list):
        # form gets set only if forms is a list
        new_buildings["form"] = forms

    if form_to_btype_callback is not None:
        new_buildings["building_type_id"] = new_buildings["form"].\
            apply(form_to_btype_callback)

    new_buildings["stories"] = new_buildings.stories.apply(np.ceil)

    if add_more_columns_callback is not None:
        new_buildings = add_more_columns_callback(new_buildings)

    print "Adding {:,} buildings with {:,} {}".\
        format(len(new_buildings),
               int(new_buildings[supply_fname].sum()),
               supply_fname)

    print "{:,} feasible buildings after running developer".format(
          len(dev.feasibility))

    all_buildings = dev.merge(buildings.to_frame(buildings.local_columns),
                              new_buildings[buildings.local_columns])

    orca.add_table("buildings", all_buildings)

def supply_demand(cfg, choosers, alternatives, buildings, alt_seg, price_col, reg_col=None):
    lcm = yaml_to_class(cfg).from_yaml(str_or_buffer=cfg)
    choosers_frame = choosers.to_frame()
    alts_frame = alternatives.to_frame()
    new_price, zone_ratios = supplydemand.supply_and_demand(lcm, choosers_frame, alts_frame, alt_seg, price_col, reg_col=reg_col)
    buildings.update_col_from_series(price_col, new_price)

def export_indicators(zones, year):
    engine = create_engine('postgresql://postgres:postgres@localhost:5432/postgres', echo=False)
    #TODO add county table to h5 file
    counties = pd.read_csv('c:/urbansim/data/counties.csv', index_col=0)

    buildings = orca.merge_tables('buildings', tables=['buildings','parcels'],
                             columns=['unit_price_residential','unit_price_non_residential','residential_units',
                                      'non_residential_units','building_type_id', 'zone_id','county_id'])

    establishments = orca.merge_tables('establishments', tables=['establishments','buildings','parcels'],
                                       columns=['employees', 'sector_id_six','zone_id', 'county_id'])

    households = orca.merge_tables('households', tables=['households','buildings','parcels'], columns=
                                   ['persons','age_of_head','income', 'zone_id', 'county_id'])

    ##zone_summary
    zone_summary = pd.DataFrame(index=zones.index)
    zone_summary['sim_year'] = year - 1
    zone_summary['pop_sim'] = households.persons.groupby(households.zone_id).sum()
    zone_summary['hh_sim'] = households.age_of_head.groupby(households.zone_id).size()
    zone_summary['emp_sim'] = establishments.employees.groupby(establishments.zone_id).sum()
    zone_summary['unit_price_residential'] = buildings.unit_price_residential.groupby(buildings.zone_id).mean()
    zone_summary['unit_price_non_residential'] = buildings.unit_price_non_residential.groupby(buildings.zone_id).mean()
    zone_summary['ru_sim'] = buildings.residential_units.groupby(buildings.zone_id).sum()
    zone_summary['nr_sim'] = buildings.non_residential_units.groupby(buildings.zone_id).sum()
    zone_summary['buildings'] = buildings.building_type_id.groupby(buildings.zone_id).size()
    zone_summary['median_income_sim'] = households.income.groupby(households.zone_id).median()
    zone_summary['emp1_sim'] = establishments.employees.loc[establishments.sector_id_six == 1].groupby(establishments.zone_id).sum()
    zone_summary['emp2_sim'] = establishments.employees.loc[establishments.sector_id_six == 2].groupby(establishments.zone_id).sum()
    zone_summary['emp3_sim'] = establishments.employees.loc[establishments.sector_id_six == 3].groupby(establishments.zone_id).sum()
    zone_summary['emp4_sim'] = establishments.employees.loc[establishments.sector_id_six == 4].groupby(establishments.zone_id).sum()
    zone_summary['emp5_sim'] = establishments.employees.loc[establishments.sector_id_six == 5].groupby(establishments.zone_id).sum()
    zone_summary['emp6_sim'] = establishments.employees.loc[establishments.sector_id_six == 6].groupby(establishments.zone_id).sum()




    ##county_summary
    county_summary = pd.DataFrame(index=counties.index)
    county_summary['county_name'] = counties.values
    county_summary['sim_year'] = year
    county_summary['pop_sim'] = households.persons.groupby(households.county_id).sum()
    county_summary['hh_sim'] = households.age_of_head.groupby(households.county_id).size()
    county_summary['emp_sim'] = establishments.employees.groupby(establishments.county_id).sum()
    county_summary['unit_price_residential_sim'] = buildings.unit_price_residential.groupby(buildings.county_id).mean()
    county_summary['unit_price_non_residential_sim'] = buildings.unit_price_non_residential.groupby(buildings.county_id).mean()
    county_summary['ru_sim'] = buildings.residential_units.groupby(buildings.county_id).sum()
    county_summary['nr_sim'] = buildings.non_residential_units.groupby(buildings.county_id).sum()
    county_summary['buildings'] = buildings.building_type_id.groupby(buildings.county_id).size()
    county_summary['median_income_sim'] = households.income.groupby(households.county_id).median()
    county_summary['emp1_sim'] = establishments.employees.loc[establishments.sector_id_six == 1].groupby(establishments.county_id).sum()
    county_summary['emp2_sim'] = establishments.employees.loc[establishments.sector_id_six == 2].groupby(establishments.county_id).sum()
    county_summary['emp3_sim'] = establishments.employees.loc[establishments.sector_id_six == 3].groupby(establishments.county_id).sum()
    county_summary['emp4_sim'] = establishments.employees.loc[establishments.sector_id_six == 4].groupby(establishments.county_id).sum()
    county_summary['emp5_sim'] = establishments.employees.loc[establishments.sector_id_six == 5].groupby(establishments.county_id).sum()
    county_summary['emp6_sim'] = establishments.employees.loc[establishments.sector_id_six == 6].groupby(establishments.county_id).sum()

    orca.add_table('zone_summary', zone_summary, cache=False)
    orca.add_table('county_summary', county_summary, cache=False)

    #one_summary.to_sql('zone_summary_new', engine, if_exists='append')
    #county_summary.to_sql('county_summary_new', engine, if_exists='append')
    

