import numpy as np, pandas as pd
import time, os
from synthicity.utils import misc
import warnings

warnings.filterwarnings('ignore',category=pd.io.pytables.PerformanceWarning)

# this is the central location to do all the little data format issues that will be needed by all models

class Dataset(object):

  def __init__(self,filename):
      self.store = pd.HDFStore(filename)
      self.d = {} # keep track of in memory pandas data frames so as not to load multiple times form disk
      self.coeffs = pd.DataFrame() # keep all coefficients in memory
      self.attrs = {} # keep all computed outputs in memory
      self.savetbls = [] # names of tables to write to output hdf5

  def close(self):
    self.store.close()

  def save_toinputfile(self,name,df):
    self.store[name] = df

  def save_tmptbl(self,name,tbl):
    # used to pass a different table instead of the one that comes from the database
    self.d[name] = tbl

  def save_table(self,name):
    if name not in self.savetbls: self.savetbls.append(name)

  def save_output(self,filename):
    outstore = pd.HDFStore(filename)
    for key in self.attrs.keys():
      df = self.attrs[key]
      df = misc.df64bitto32bit(df)
      print key+"\n", df.describe()
      outstore[key] = df
    
    for key in self.savetbls:
      df = self.fetch(key)
      df = misc.df64bitto32bit(df)
      print key+"\n", df.describe()
      outstore[key] = df

    outstore.close()

  def copy_coeffs(self,filename=None):
    if filename: coeffstore = pd.HDFStore(filename)
    else: coeffstore = self.store
    self.coeffs = coeffstore['coefficients'] 
    coeffstore.close()

  def save_coeffs(self,filename):
    outstore = pd.HDFStore(filename)
    print self.coeffs
    outstore['coefficients'] = self.coeffs 
    outstore.close()

  def fetch(self,name,**kwargs):
    if name in self.d: return self.d[name]
   
    print "Fetching %s" % name
    f = "fetch_%s"%name
    if hasattr(self,f): tbl = getattr(self,f)(**kwargs)
    else: tbl = self.store[name]
    
    if hasattr(self,"modify_table"): tbl = self.modify_table(name,tbl,**kwargs) 
  
    self.d[name] = tbl
    
    return tbl
    
  def __getattr__(self,name):
    # this is a little hackish - basically if you called fetch we don't want to call fetch again
    if "fetch_" in name: raise Exception()
    return self.fetch(name)

  def load_attr(self,name,year):
    return self.attrs[name][year]

  def store_attr(self,name,year,value):
    value = misc.series64bitto32bit(value)
    if name in self.attrs and year in self.attrs[name]: del self.attrs[name][year]
    df = self.attrs.get(name,pd.DataFrame(index=value.index))
    df[year] = value
    self.attrs[name] = df

  def load_coeff(self,name):
    return self.coeffs[(name,'coeffs')].dropna()
  
  def load_fnames(self,name):
    return self.coeffs[(name,'fnames')].dropna()
  
  def load_coeff_series(self,name):
    return pd.Series(self.load_coeff(name).values,index=self.load_fnames(name).values)

  def store_coeff(self,name,value,fnames=None):
    colname1 = (name,'coeffs')
    colname2 = (name,'fnames')
    if colname1 in self.coeffs: del self.coeffs[colname1]
    if colname2 in self.coeffs: del self.coeffs[colname2]

    d = {colname1:value}
    if fnames is not None: d[colname2] = fnames
    self.coeffs = pd.concat([self.coeffs,pd.DataFrame(d)],axis=1)
  
  # this is a shortcut function to join the table with dataset.fetch(tblname) 
  # using the foreign_key in order to add fieldname to the source table
  def join_for_field(self,table,tblname,foreign_key,fieldname):
    if type(table) == type(''): table = self.fetch(table)
    if foreign_key == None: # join on index
        return pd.merge(table,self.fetch(tblname)[[fieldname]],left_index=True,right_index=True)
    return pd.merge(table,self.fetch(tblname)[[fieldname]],left_on=foreign_key,right_index=True)
  
  def compute_range(self,attr,dist,agg=np.sum):
    travel_data = self.fetch('travel_data').reset_index(level=1)
    travel_data = travel_data[travel_data.travel_time<dist]
    travel_data["attr"] = attr[travel_data.to_zone_id].values
    return travel_data.groupby(level=0).attr.apply(agg) 
  
  def relocation_rates(self,agents,relocation,rate_fname):

    agents["relocation_rate"] = np.zeros(len(agents.index),dtype="float32")
    for row in relocation.iterrows():
      row = row[1] # discard index
      a = []
      for fname in row.index:
        if fname.endswith("_max"):
          shrtfname = fname[:-4]
          assert shrtfname in agents.columns
          val = row[fname]
          if val == -1: val = np.finfo('d').max
          a.append(agents[shrtfname] < val)
        elif fname.endswith("_min"):
          shrtfname = fname[:-4]
          assert shrtfname in agents.columns
          val = row[fname]
          if val == -1: val = np.finfo('d').min
          a.append(agents[shrtfname] > val)
        elif fname != rate_fname: # field needs to be in there
          assert fname in agents.columns
          val = row[fname] 
          a.append(agents[fname] == val)

      agents.relocation_rate.values[np.prod(a,axis=0).astype('bool')] = row[rate_fname]
    
    print "Histogram of relocation rates:"
    print agents.relocation_rate.value_counts()
    movers = agents[np.random.sample(len(agents.index)) < agents.relocation_rate].index
    print "%d agents are moving" % len(movers)
    if len(movers) == 0: raise Exception("Stopping execution - no movers, which is probably a bug")
    return movers
