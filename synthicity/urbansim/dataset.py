import copy
import os
import time
import warnings

import numpy as np
import pandas as pd
import simplejson

from synthicity.utils import misc

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

    def copy_coeffs(self, filename=None):
        if filename:
            coeffstore = pd.HDFStore(filename)
        else:
            coeffstore = self.store
        self.coeffs = coeffstore['coefficients']
        coeffstore.close()

    def save_coeffs(self, filename):
        outstore = pd.HDFStore(filename)
        print self.coeffs
        outstore['coefficients'] = self.coeffs
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

    def load_attr(self, name, year):
        return self.attrs[name][year]

    def store_attr(self, name, year, value):
        value = misc.series64bitto32bit(value)
        if name in self.attrs and year in self.attrs[name]:
            del self.attrs[name][year]
        df = self.attrs.get(name, pd.DataFrame(index=value.index))
        value = pd.DataFrame({year: value}, index=value.index)
        df = pd.concat([df, value], axis=1)
        self.attrs[name] = df

    def load_coeff(self, name, jsonformat=True):
        if jsonformat:
            d = simplejson.loads(
                open(os.path.join(misc.coef_dir(), name + '.json')).read())
            return np.array(d["coeffs"])
        else:
            return self.coeffs[(name, 'coeffs')].dropna()

    def load_fnames(self, name, jsonformat=True):
        if jsonformat:
            d = simplejson.loads(
                open(os.path.join(misc.coef_dir(), name + '.json')).read())
            return d["fnames"]
        else:
            return self.coeffs[(name, 'fnames')].dropna()

    def load_coeff_series(self, name):
        return pd.Series(
            self.load_coeff(name).values, index=self.load_fnames(name).values)

    def store_coeff(self, name, value, fnames=None, jsonformat=True):
        if jsonformat:
            d = {"coeffs": [round(x, 3) for x in list(value)]}
            if fnames is not None:
                d["fnames"] = list(fnames)
            open(os.path.join(misc.coef_dir(), name + '.json'),
                 'w').write(simplejson.dumps(d, indent=4))
        else:
            colname1 = (name, 'coeffs')
            colname2 = (name, 'fnames')
            if colname1 in self.coeffs:
                del self.coeffs[colname1]
            if colname2 in self.coeffs:
                del self.coeffs[colname2]

            d = {colname1: value}
            if fnames is not None:
                d[colname2] = fnames
            self.coeffs = pd.concat([self.coeffs, pd.DataFrame(d)], axis=1)

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

    def relocation_rates(self, agents, relocation, rate_fname):

        agents["relocation_rate"] = np.zeros(
            len(agents.index), dtype="float32")
        for row in relocation.iterrows():
            row = row[1]  # discard index
            a = []
            for fname in row.index:
                if fname.endswith("_max"):
                    shrtfname = fname[:-4]
                    assert shrtfname in agents.columns
                    val = row[fname]
                    if val == -1:
                        val = np.finfo('d').max
                    a.append(agents[shrtfname] < val)
                elif fname.endswith("_min"):
                    shrtfname = fname[:-4]
                    assert shrtfname in agents.columns
                    val = row[fname]
                    if val == -1:
                        val = np.finfo('d').min
                    a.append(agents[shrtfname] > val)
                elif fname != rate_fname:  # field needs to be in there
                    assert fname in agents.columns
                    val = row[fname]
                    a.append(agents[fname] == val)

            agents.relocation_rate.values[
                np.prod(a, axis=0).astype('bool')] = row[rate_fname]

        print "Histogram of relocation rates:"
        print agents.relocation_rate.value_counts()
        movers = agents[
            np.random.sample(len(agents.index)) < agents.relocation_rate].index
        print "%d agents are moving" % len(movers)
        if len(movers) == 0:
            raise Exception(
                "Stopping execution - no movers, which is probably a bug")
        return movers

    def add_xy(self, df):

        assert 'building_id' in df

        for col in ['_node_id', 'x', 'y']:
            if col in df.columns:
                del df[col]
            df = self.join_for_field(df, 'buildings', 'building_id', col)

        return df

    def choose(self, p, mask, alternatives, segment, new_homes, minsize=None):
        p = copy.copy(p)

        if minsize is not None:
            p[alternatives.supply < minsize] = 0
        else:
            p[mask] = 0  # already chosen
        print "Choosing from %d nonzero alts" % np.count_nonzero(p)

        try:
            indexes = np.random.choice(
                len(alternatives.index), len(segment.index),
                replace=False, p=p / p.sum())
        except:
            print ("WARNING: not enough options to fit agents, "
                   "will result in unplaced agents")
            return mask, new_homes
        new_homes.ix[segment.index] = alternatives.index.values[indexes]

        if minsize is not None:
            alternatives["supply"].ix[
                alternatives.index.values[indexes]] -= minsize
        else:
            mask[indexes] = 1

        return mask, new_homes
