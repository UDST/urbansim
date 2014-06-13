import warnings

import numpy as np
import pandas as pd

from urbansim.utils import misc
reindex = misc.reindex

warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)

# this is the central location to do all the little data format issues
# that will be needed by all models


class Dataset(object):

    def __init__(self, filename, scenario="baseline"):
        self.store = pd.HDFStore(filename, "r")
        # keep track of in memory pandas data frames so as
        # not to load multiple times form disk
        self.d = {}
        self.scenario = scenario
        self.clear_views()
        self.debug = False

    def view(self, name):
        return self.views[name]

    def clear_views(self):
        self.views = {}

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
            print "Writing %s to disk" % key
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


class CustomDataFrame(object):
    def __init__(self, dset, name):
        self.dset = dset
        self.name = name

    @property
    def df(self):
        return self.dset.fetch(self.name)

    def build_df(obj, flds=None):
        flds = None
        if flds is None:
            if obj.flds is None:
                return obj.df
            else:
                flds = obj.flds
        columns = [getattr(obj, fld) for fld in flds]
        df = pd.concat(columns, axis=1)
        df.columns = flds
        return df

    def __getattr__(self, name):
        try:
            return super(CustomDataFrame, "__getattr__")(name)
        except:
            attr = getattr(self.df, name)
            if self.dset.debug is True:
                print "Returning primary attribute: %s of %s" % (name, self.name)
            return attr


def variable(func):
    @property
    def _decorator(self):
        if hasattr(self, "_property_cache") and func in self._property_cache:
            val = self._property_cache[func]
            if self.dset.debug is True:
                print "Returning from cache: %s of %s" % \
                      (func.__name__, self.name)
            return val

        s = func(self)

        if self.dset.debug is True:
            print "Computing: %s of %s as" % (func.__name__, self.name)
            print "    %s" % s
        try:
            r = eval(s, globals(), self.dset.views)
        except Exception as e:
            print "Variable computation failed!!"
            print s
            print e, "\n\n\n"

        r[np.isinf(r)] = np.nan

        if not hasattr(self, "_property_cache"):
            self._property_cache = {}
        self._property_cache[func] = r

        return r
    return _decorator
