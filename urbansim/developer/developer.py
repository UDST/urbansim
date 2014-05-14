import pandas as pd
import numpy as np

TARGETVACANCY = .1  # really should be a target per TAZ
MAXPARCELSIZE = 200000  # really need some subdivision
AVERESUNITSIZE = 1300  # really should be an accessibility variable
AVENONRESUNITSIZE = 250  # really should be an accessibility variable


def developer_run(dset, year=2010):
    res_developer_run(dset, year)
    nonres_developer_run(dset, year)


def res_developer_run(dset, year=2010):
    exec_developer(dset, year, "households", "residential_units",
                   [1, 2, 3])


def nonres_developer_run(dset, year=2010):
    exec_developer(dset, year, "jobs", "non_residential_units",
                   [4, 7, 8, 9, 10, 11, 12, 14], nonres=True)


def exec_developer(dset, year, agents, unit_fname, btypes,
                   nonres=False):
    numagents = len(dset.fetch(agents).index)
    numunits = dset.buildings[unit_fname].sum()
    print "Running developer for %s" % agents
    print "Number of agents: %d" % numagents
    print "Number of agent spaces: %d" % numunits
    assert TARGETVACANCY < 1.0
    targetunits = max(numagents / (1 - TARGETVACANCY) - numunits, 0)
    print "Current vacancy = %.2f" % (1 - numagents / numunits)
    print "Target vacancy = %.2f, target of new units = %d" % \
        (TARGETVACANCY, targetunits)

    #df = pd.read_csv(
    #    os.path.join(misc.data_dir(), 'far_predictions.csv'),
    #    index_col='parcel_id')
    df = dset.feasibility

    fnames = ["type%d_profit" % i for i in btypes]
    fnames_d = dict((fnames[i], btypes[i]) for i in range(len(fnames)))

    # pick the max profit building type so a parcel only gets
    # developed once
    df["max_profit"] = df[fnames].max(axis=1)
    df["max_btype"] = df[fnames].idxmax(axis=1).map(fnames_d)

    # this is a little bit wonky too - need to take
    # the far from the max profit above
    df["max_feasiblefar"] = df[
        ["type%d_feasiblefar" % i for i in btypes]].max(axis=1)

    # feasible buildings only for this building type
    df = df[df.max_feasiblefar > 0]
    df = df[df.parcelsize < MAXPARCELSIZE]
    df['newsqft'] = df.parcelsize * df.max_feasiblefar
    AVEUNITSIZE = AVENONRESUNITSIZE if nonres else AVERESUNITSIZE
    df['newunits'] = np.ceil(df.newsqft / AVEUNITSIZE).astype('int')

    df['total_units'] = dset.buildings.groupby('parcel_id')[unit_fname].sum()
    df['netunits'] = df.newunits - df.total_units
    print "Describe of netunits\n", df.describe()

    choices = np.random.choice(df.index.values, size=len(df.index),
                               replace=False,
                               p=(df.max_profit.values / df.max_profit.sum()))
    netunits = df.netunits.loc[choices]
    totunits = netunits.values.cumsum()
    ind = np.searchsorted(totunits, targetunits, side="right")
    build = choices[:ind]

    # from here on just keep the ones to build
    df = df.loc[build]

    print "Describe of buildings built\n", df.total_units.describe()
    print "Describe of profit\n", df.max_profit.describe()
    print "Buildings by type\n", df.max_btype.value_counts()
    print "Units by type\n", \
        df.groupby('max_btype').netunits.sum().order(ascending=False)

    # df = df.join(dset.parcels).reset_index()
    # format new buildings to concatenate with old buildings
    NOTHING = np.zeros(len(df.index))
    df['parcel_id'] = df.index
    df['building_type'] = df.max_btype
    df['residential_units'] = NOTHING if nonres else df.newunits
    df['non_residential_units'] = df.newunits if nonres else NOTHING
    df['building_sqft'] = df.newsqft
    df['stories'] = df.max_feasiblefar
    df['building_type_id'] = df.max_btype
    df['year_built'] = np.ones(len(df.index)) * year
    df['general_type'] = df.building_type_id.map(dset.BUILDING_TYPE_MAP)
    df['unit_sqft'] = AVEUNITSIZE
    df['lot_size'] = dset.parcels.shape_area * 10.764
    df['unit_lot_size'] = df.lot_size / df.residential_units
    for f in ['x', 'y', '_node_id']:
        df[f] = dset.parcels[f]
    maxind = np.max(dset.buildings.index.values)
    df.index = df.index + maxind + 1
    dset.buildings = pd.concat([dset.buildings, df], verify_integrity=True)
    dset.buildings.index.name = 'building_id'

    # need to remove parcels from consideration in feasibility
    dset.save_tmptbl("feasibility", dset.feasibility.drop(build))

