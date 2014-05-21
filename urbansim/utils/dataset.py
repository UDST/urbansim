import copy
import os
import time
import warnings

import numpy as np
import pandas as pd
import simplejson

from urbansim.utils import misc

warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)

# this is the central location to do all the little data format issues
# that will be needed by all models


class Dataset(object):

    def __init__(self, filename):
        self.store = pd.HDFStore(filename, "r")
        # keep track of in memory pandas data frames so as not to load multiple
        # times form disk
        self.d = {}
        self.coeffs = pd.DataFrame()  # keep all coefficients in memory
        self.attrs = {}  # keep all computed outputs in memory
        self.savetbls = []  # names of tables to write to output hdf5

    def close(self):
        self.store.close()

    def list_tbls(self):
        return list(set([x[1:] for x in self.store.keys()] + self.d.keys()))

    def save_toinputfile(self, name, df):
        self.store[name] = df

    def save_tmptbl(self, name, tbl):
        # used to pass a different table instead of the one that comes from the
        # database
        self.d[name] = tbl

    def save_table(self, name):
        if name not in self.savetbls:
            self.savetbls.append(name)

    def save_output(self, filename):
        outstore = pd.HDFStore(filename)
        for key in self.attrs.keys():
            df = self.attrs[key]
            df = misc.df64bitto32bit(df)
            print key + "\n", df.describe()
            outstore[key] = df

        for key in self.savetbls:
            df = self.fetch(key)
            df = misc.df64bitto32bit(df)
            print key + "\n", df.describe()
            outstore[key] = df

        outstore.close()

    def fetch_csv(self, name, **kwargs):
        if name in self.d:
            return self.d[name]
        tbl = pd.read_csv(os.path.join(misc.data_dir(), name), **kwargs)
        self.d[name] = tbl
        return tbl

    def fetch(self, name, **kwargs):
        if name in self.d:
            return self.d[name]

        print "Fetching %s" % name
        f = "fetch_%s" % name
        if hasattr(self, f):
            tbl = getattr(self, f)(**kwargs)
        else:
            tbl = self.store[name]

        if hasattr(self, "modify_table"):
            tbl = self.modify_table(name, tbl, **kwargs)

        self.d[name] = tbl

        return tbl

    def __getattr__(self, name):
        # this is a little hackish - basically if you called fetch we don't
        # want to call fetch again
        if "fetch_" in name:
            raise Exception()
        return self.fetch(name)

    # this is a shortcut function to join the table with dataset.fetch(tblname)
    # using the foreign_key in order to add fieldname to the source table
    def join_for_field(self, table, tblname, foreign_key, fieldname):
        if isinstance(table, str):
            table = self.fetch(table)
        if foreign_key is None:  # join on index
            # return
            # pd.merge(table, self.fetch(tblname)[[fieldname]],
            #          left_index=True, right_index=True)
            table[fieldname] = self.fetch(
                tblname)[fieldname].loc[table.index].values
        else:
            # return
            # pd.merge(table, self.fetch(tblname)[[fieldname]],
            #          left_on=foreign_key, right_index=True)
            table[fieldname] = self.fetch(
                tblname)[fieldname].loc[table[foreign_key]].values
        return table

    def compute_range(self, attr, dist, agg=np.sum):
        travel_data = self.fetch('travel_data').reset_index(level=1)
        travel_data = travel_data[travel_data.travel_time < dist]
        travel_data["attr"] = attr[travel_data.to_zone_id].values
        return travel_data.groupby(level=0).attr.apply(agg)

    def add_xy(self, df):

        assert 'building_id' in df

        cols = ['x', 'y']
        for col in cols:
            if col in df.columns:
                del df[col]

        df = pd.merge(df, self.buildings[cols],
                      left_on='building_id', right_index=True)
        return df
