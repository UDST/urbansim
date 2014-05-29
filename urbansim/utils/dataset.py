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
        self.scenario = "baseline"

    def list_tbls(self):
        return list(set([x[1:] for x in self.store.keys()] + self.d.keys()))

    def save_tmptbl(self, name, tbl):
        self.d[name] = tbl

    def save_output(self, filename, savetbls=None, debug=False):

        if savetbls is None:
            savetbls = self.d.keys()

        outstore = pd.HDFStore(filename, "w")

        for key in savetbls:
            df = self.fetch(key)
            df = misc.df64bitto32bit(df)
            print key
            if debug:
                print df.describe()
            outstore[key] = df

        outstore.close()

    def fetch(self, name, **kwargs):
        if name in self.d:
            return self.d[name]

        print "Fetching %s" % name
        f = "fetch_%s" % name
        if hasattr(self, f):
            tbl = getattr(self, f)(**kwargs)
        else:
            tbl = self.store[name]

        self.d[name] = tbl

        return tbl

    def __getattr__(self, name):
        # this is a little hackish - basically if you called fetch we don't
        # want to call fetch again
        if "fetch_" in name:
            raise Exception()
        return self.fetch(name)

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


class CustomDataFrame:
    def __init__(self):
        pass

    def build_df(obj, flds=None):
        if flds is None:
            flds = obj.flds
        columns = [getattr(obj, fld) for fld in flds]
        df = pd.concat(columns, axis=1)
        df.columns = flds
        return df
